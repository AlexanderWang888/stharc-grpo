"""5 个业务无关 reward 组件，全部基于 :class:`TrajectorySummary`。

1. TaskCompletedReward      —— 任务是否完成（final_answer 非空 + 无 env error + 有 is_final step）
2. ToolCallSuccessReward    —— 非 final step 中 parse_ok 比例
3. JudgeReward              —— 调外部 judge LLM 打分（同步 HTTP，OpenAI 兼容）
4. TeacherSimilarityReward  —— 与 summary.teacher_trace 的 token jaccard
5. AiToCEfficientReward     —— 乘法奖励：task_completed × step_efficiency（推荐用于 ai_to_c GRPO）

设计原则：
- compute(ctx) 必须**同步**（AgentLoop 会整体 to_thread 跑）
- 异常自吞，return RewardResult(0.0, info={"error": ...})（让 WeightedReward 统一处理）
- 数值通常 ∈ [0, 1]
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

from .base import RewardComponent, RewardContext, RewardResult
from .completion import is_valid_task_completed
from .gsb_absolute import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ReAct 格式校验（供 AiToCGatedPairwiseReward 格式门使用）
# ---------------------------------------------------------------------------

# no-think 模式模板注入的空块：<think>\n\n</think>，vLLM 强制前缀，不算违规
_FMT_EMPTY_COT  = re.compile(r'<(?:think|thinking)>\s*</(?:think|thinking)>', re.IGNORECASE)
# 去掉空块后若仍有 </think> / </thinking> → 模型输出了思考内容再关标签，违规
_FMT_CLOSE_ONLY = re.compile(r'</(?:think|thinking)>', re.IGNORECASE)
# 同一步 Thought: 出现次数
_FMT_THOUGHT_RE = re.compile(r'(?m)^Thought\s*:')


def _react_format_ok(response_text: str) -> bool:
    """检查单步 response_text 是否符合 ReAct 格式。

    合法：去掉 no-think 模板空块后，Thought: 恰好出现一次，无裸 </think>/<\\/thinking>。
    违规返回 False，格式正常返回 True。
    """
    text = _FMT_EMPTY_COT.sub("", response_text)
    if _FMT_CLOSE_ONLY.search(text):
        return False
    if len(_FMT_THOUGHT_RE.findall(text)) > 1:
        return False
    return True


# ---------------------------------------------------------------------------
# 1) TaskCompletedReward
# ---------------------------------------------------------------------------


class TaskCompletedReward(RewardComponent):
    """任务完成度（0/1 二值）。

    判定（**全部**满足才 1）：
      - summary.final_answer 去空白后非空（长度 ≥ min_answer_len）
      - summary.has_error 为 False
      - summary.has_final_step 为 True（最后一步显式 finish / give_link_card）

    info 里输出每个子条件的 0/1，便于 wandb 排查为啥总不给分。
    """

    name = "task_completed"

    def __init__(
        self,
        name: Optional[str] = None,
        enabled: bool = True,
        min_answer_len: int = 1,
        require_final_step: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, enabled=enabled, **kwargs)
        self.min_answer_len = int(min_answer_len)
        self.require_final_step = bool(require_final_step)

    def compute(self, ctx: RewardContext) -> RewardResult:
        s = ctx.summary
        ans_ok = len((s.final_answer or "").strip()) >= self.min_answer_len
        no_err = not s.has_error
        completed, comp_info = is_valid_task_completed(s, min_answer_len=self.min_answer_len)
        fin_ok = completed if self.require_final_step else True

        v = 1.0 if (ans_ok and no_err and fin_ok) else 0.0
        return RewardResult(
            value=v,
            info={
                "answer_ok": int(ans_ok),
                "no_error": int(no_err),
                "has_final_step": comp_info.get("has_final_step", int(s.has_final_step)),
                "has_valid_final": comp_info.get("has_valid_final", 0),
                "deprecated_terminal": comp_info.get("deprecated_terminal", 0),
                "answer_len": comp_info.get("answer_len", len((s.final_answer or "").strip())),
            },
        )


# ---------------------------------------------------------------------------
# 2) ToolCallSuccessReward
# ---------------------------------------------------------------------------


class ToolCallSuccessReward(RewardComponent):
    """非 final 步的 parse_ok 比例。

    - 无非 final 步时返回 ``empty_value``（默认 1.0；不惩罚那些「一步出 final」的样本）
    - 同时输出 raw count / total 便于诊断
    """

    name = "tool_call_success"

    def __init__(
        self,
        name: Optional[str] = None,
        enabled: bool = True,
        empty_value: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, enabled=enabled, **kwargs)
        self.empty_value = float(empty_value)

    def compute(self, ctx: RewardContext) -> RewardResult:
        s = ctx.summary
        action_steps = [st for st in s.steps if not st.is_final]
        total = len(action_steps)
        if total == 0:
            return RewardResult(
                value=self.empty_value,
                info={"num_action_steps": 0, "num_parse_ok": 0},
            )
        ok = sum(1 for st in action_steps if st.parse_ok)
        v = ok / total
        return RewardResult(
            value=v,
            info={"num_action_steps": total, "num_parse_ok": ok},
        )


# ---------------------------------------------------------------------------
# 3) JudgeReward
# ---------------------------------------------------------------------------


# 默认 prompt：让 judge 输出严格 JSON {"score": 0~1}
_DEFAULT_JUDGE_SYSTEM = (
    "你是保险领域的回答质量打分器。你将看到用户问题和模型回答。"
    "请综合考虑【正确性、信息完整性、保险合规性、可执行性】打 0~1 分（精度 0.05），"
    "**只输出**一行严格的 JSON：{\"score\": 0.xx}，不要任何额外说明。"
)


class JudgeReward(RewardComponent):
    """外部 judge LLM 打分（同步 HTTP，OpenAI 兼容 /chat/completions）。

    Args:
        judge_url: judge 端点（如 "http://judge-llm:8000/v1"，不含 /chat/completions）
        judge_model: judge model 名
        api_key: 可选 Bearer token
        temperature / max_tokens / timeout: 调用参数
        system_prompt: 打分指引
        max_answer_chars: 给 judge 的 answer 最多多少字符（防止超长）
        default_on_error: 异常时返回的分数（默认 0.0）

    判定逻辑：
        - 解析 judge 输出里的 JSON `{"score": x}`（兼容数字裸文本兜底）
        - 拿到 score 后 clip 到 [0, 1]
    """

    name = "judge"

    def __init__(
        self,
        name: Optional[str] = None,
        enabled: bool = True,
        judge_url: Optional[str] = None,
        judge_model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 64,
        timeout: int = 60,
        system_prompt: Optional[str] = None,
        max_answer_chars: int = 4000,
        default_on_error: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, enabled=enabled, **kwargs)
        if enabled and (not judge_url or not judge_model):
            raise ValueError("JudgeReward enabled=True 时必须给出 judge_url 和 judge_model")
        self.judge_url = (judge_url or "").rstrip("/")
        self.judge_model = judge_model or ""
        self.api_key = api_key
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = int(timeout)
        self.system_prompt = system_prompt or _DEFAULT_JUDGE_SYSTEM
        self.max_answer_chars = int(max_answer_chars)
        self.default_on_error = float(default_on_error)

    # ------------------------------------------------------------------

    def compute(self, ctx: RewardContext) -> RewardResult:
        s = ctx.summary
        query = (s.query or "").strip()
        answer = (s.final_answer or "").strip()
        if not answer:
            return RewardResult(value=0.0, info={"skipped": "empty_answer"})

        if len(answer) > self.max_answer_chars:
            answer = answer[: self.max_answer_chars] + "...[truncated]"

        user = f"【用户问题】\n{query}\n\n【模型回答】\n{answer}\n\n请按规范输出 JSON 评分。"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]

        try:
            raw = self._call_judge(messages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[judge] call failed: %s", exc)
            return RewardResult(
                value=self.default_on_error,
                info={"error": f"{type(exc).__name__}: {exc}"},
            )

        score, parse_ok = _parse_judge_score(raw)
        score = max(0.0, min(1.0, float(score)))
        return RewardResult(
            value=score,
            info={"raw": raw[:200], "parse_ok": int(parse_ok)},
        )

    # ------------------------------------------------------------------

    def _call_judge(self, messages: List[Dict[str, str]]) -> str:
        import requests  # 延迟 import，避免单测环境不需要

        url = f"{self.judge_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.judge_model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return str(
            data.get("choices", [{}])[0].get("message", {}).get("content", "")
        )


_SCORE_JSON_RE = re.compile(r'\{[^{}]*"score"\s*:\s*([0-9]*\.?[0-9]+)[^{}]*\}')
_SCORE_FALLBACK_RE = re.compile(r"([0-9]*\.?[0-9]+)")


def _parse_judge_score(text: str) -> tuple[float, bool]:
    """从 judge 输出中提取 score。返回 (score, parse_ok)。"""
    if not isinstance(text, str):
        return 0.0, False
    text = text.strip()

    # 优先：完整 JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "score" in obj:
            return float(obj["score"]), True
    except Exception:  # noqa: BLE001
        pass

    # 次优：正则抓 `{"score": x}`
    m = _SCORE_JSON_RE.search(text)
    if m:
        try:
            return float(m.group(1)), True
        except Exception:  # noqa: BLE001
            pass

    # 兜底：找到第一个数字（避免误判 prompt 里的数字）
    m2 = _SCORE_FALLBACK_RE.search(text)
    if m2:
        try:
            v = float(m2.group(1))
            return v, False  # 标记 parse_ok=False 但值仍可用
        except Exception:  # noqa: BLE001
            pass

    return 0.0, False


# ---------------------------------------------------------------------------
# 4) TeacherSimilarityReward
# ---------------------------------------------------------------------------


class TeacherSimilarityReward(RewardComponent):
    """与 summary.teacher_trace 的 token jaccard 相似度。

    - 不依赖外部模型；用 unicode 字符 + 空白分词的近似 jaccard
    - teacher_trace 为 None 或空 → 跳过（返回 ``empty_value``，默认 0.0）
    - 主要用作辅助 reward（防 reward hacking 后期监控用），权重通常较低

    Args:
        empty_value: 没有 teacher_trace 时的返回值（默认 0.0）
        min_token: 至少多少 token 才有效（避免极短样本）
        token_pattern: 分词正则（默认中英混合：连续字母数字 OR 单个 CJK 字符）
    """

    name = "teacher_similarity"

    _DEFAULT_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")

    def __init__(
        self,
        name: Optional[str] = None,
        enabled: bool = True,
        empty_value: float = 0.0,
        min_token: int = 3,
        token_pattern: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, enabled=enabled, **kwargs)
        self.empty_value = float(empty_value)
        self.min_token = int(min_token)
        self._pat = re.compile(token_pattern) if token_pattern else self._DEFAULT_PATTERN

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return self._pat.findall(text)

    def compute(self, ctx: RewardContext) -> RewardResult:
        s = ctx.summary
        teacher = s.teacher_trace or ""
        student = s.response_text or s.final_answer or ""

        t_tokens = set(self._tokenize(teacher))
        st_tokens = set(self._tokenize(student))

        if len(t_tokens) < self.min_token or len(st_tokens) < self.min_token:
            return RewardResult(
                value=self.empty_value,
                info={
                    "teacher_tokens": len(t_tokens),
                    "student_tokens": len(st_tokens),
                    "skipped": "below_min_token",
                },
            )

        inter = t_tokens & st_tokens
        union = t_tokens | st_tokens
        v = len(inter) / max(1, len(union))
        return RewardResult(
            value=v,
            info={
                "teacher_tokens": len(t_tokens),
                "student_tokens": len(st_tokens),
                "intersect": len(inter),
            },
        )


# ---------------------------------------------------------------------------
# 5) StepCountPenaltyReward
# ---------------------------------------------------------------------------


class StepCountPenaltyReward(RewardComponent):
    """基于 ReAct 轮次的效率奖励（线性衰减）。

    鼓励模型用尽可能少的工具调用步完成任务：

        num_action_steps <= target_steps  →  score = 1.0
        num_action_steps >= max_steps     →  score = 0.0
        中间                               →  线性衰减

    公式：score = clip(1 - (n - target_steps) / (max_steps - target_steps), 0, 1)

    Args:
        target_steps: 不超过该步数时得满分（默认 3）
        max_steps: 达到或超过该步数时得 0 分（默认 7）
        empty_value: 轨迹出错（has_error=True）时返回的值（默认 0.0）

    注意：
        - has_error=True 时直接返回 empty_value，避免与 TaskCompletedReward 双重惩罚
        - num_action_steps == 0（无工具调用直接 finish）返回 1.0
    """

    name = "step_count_penalty"

    def __init__(
        self,
        name: Optional[str] = None,
        enabled: bool = True,
        target_steps: int = 3,
        max_steps: int = 7,
        empty_value: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, enabled=enabled, **kwargs)
        if target_steps >= max_steps:
            raise ValueError(
                f"target_steps({target_steps}) must be < max_steps({max_steps})"
            )
        self.target_steps = int(target_steps)
        self.max_steps = int(max_steps)
        self.empty_value = float(empty_value)

    def compute(self, ctx: RewardContext) -> RewardResult:
        s = ctx.summary
        if s.has_error:
            return RewardResult(
                value=self.empty_value,
                info={"num_action_steps": s.num_action_steps, "skipped": "has_error"},
            )
        n = s.num_action_steps
        if n <= self.target_steps:
            score = 1.0
        elif n >= self.max_steps:
            score = 0.0
        else:
            score = 1.0 - (n - self.target_steps) / (self.max_steps - self.target_steps)
        return RewardResult(
            value=score,
            info={
                "num_action_steps": n,
                "target_steps": self.target_steps,
                "max_steps": self.max_steps,
            },
        )


# ---------------------------------------------------------------------------
# 6) AiToCEfficientReward  （推荐：乘法 reward = task_completed × step_efficiency）
# ---------------------------------------------------------------------------


class AiToCEfficientReward(RewardComponent):
    """ai_to_c GRPO 专用：任务完成 × 步数效率 = 最终 reward（乘法结构）。

    逻辑：
        未完成（has_final_step=False 或 has_error）→ 0.0
        完成后按步数给效率分：
            n <= target_steps  →  1.0
            n >= max_steps     →  0.0
            中间               →  线性衰减

    公式：reward = 0 if not completed else clip(1 - (n - target) / (max - target), 0, 1)

    Args:
        target_steps: 不超过该步数时给满分（默认 3）
        max_steps: 达到或超过该步数时给 0 分（默认 7）

    info 输出（透传给 reward_extra_info / wandb）：
        task_completed  —— 0 / 1
        step_score      —— 步数效率分 ∈ [0, 1]（任务未完成时为 0.0）
        num_action_steps —— 实际工具调用步数
    """

    name = "ai_to_c_efficient"

    def __init__(
        self,
        name: Optional[str] = None,
        enabled: bool = True,
        target_steps: int = 3,
        max_steps: int = 7,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, enabled=enabled, **kwargs)
        if target_steps >= max_steps:
            raise ValueError(
                f"target_steps({target_steps}) must be < max_steps({max_steps})"
            )
        self.target_steps = int(target_steps)
        self.max_steps = int(max_steps)

    def compute(self, ctx: RewardContext) -> RewardResult:
        s = ctx.summary
        n = s.num_action_steps

        # 任务未完成：错误 / 无合法终止步 / 空 answer / 弃用终止工具。
        completed, comp_info = is_valid_task_completed(s)
        if not completed:
            return RewardResult(
                value=0.0,
                info={
                    "task_completed": 0,
                    "step_score": 0.0,
                    "num_action_steps": n,
                    **{k: v for k, v in comp_info.items() if k not in ("answer_ok",)},
                    "empty_answer": int(comp_info.get("answer_ok", 0) == 0),
                },
            )

        if n <= self.target_steps:
            step_score = 1.0
        elif n >= self.max_steps:
            step_score = 0.0
        else:
            step_score = 1.0 - (n - self.target_steps) / (self.max_steps - self.target_steps)

        return RewardResult(
            value=step_score,
            info={
                "task_completed": 1,
                "step_score": step_score,
                "num_action_steps": n,
                "empty_answer": 0,
                "has_valid_final": 1,
                "deprecated_terminal": 0,
            },
        )


# ---------------------------------------------------------------------------
# 7) GsbAbsoluteJudgeReward —— GSB 维度绝对打分（单轨迹）
# ---------------------------------------------------------------------------


def _get_anthropic_config() -> tuple[str, List[str], str]:
    """从环境变量读取 Anthropic 配置，与 agentic_train_offline.patches.config_train 对齐。

    优先直接 import config_train（ai2c 在 PYTHONPATH 时可用），否则退化到环境变量读取。
    返回 (base_url, api_key_pool, model)。
    """
    import os
    try:
        from agentic_train_offline.patches.config_train import (  # type: ignore[import]
            ANTHROPIC_BASE_URL,
            ANTHROPIC_API_KEY_POOL,
            ANTHROPIC_MODEL,
        )
        return ANTHROPIC_BASE_URL, ANTHROPIC_API_KEY_POOL, ANTHROPIC_MODEL
    except ImportError:
        pass

    base_url = os.environ.get(
        "ANTHROPIC_BASE_URL", "http://ai-api.jdcloud.com/anthropic/v1"
    ).strip()
    key_pool_raw = os.environ.get("ANTHROPIC_API_KEY_POOL", "").strip()
    key_pool = [k.strip() for k in key_pool_raw.split(",") if k.strip()]
    if not key_pool:
        single = os.environ.get("ANTHROPIC_API_KEY", "pk-068e790b-e338-4a74-a647-78d0e997240c").strip()
        key_pool = [single]
    model = os.environ.get("ANTHROPIC_MODEL", "Claude-Sonnet-4.6").strip()
    return base_url, key_pool, model


import itertools as _itertools
import threading as _threading

_gsb_key_cycle: Any = None
_gsb_key_lock = _threading.Lock()


def _next_gsb_api_key() -> str:
    global _gsb_key_cycle
    with _gsb_key_lock:
        if _gsb_key_cycle is None:
            _, pool, _ = _get_anthropic_config()
            _gsb_key_cycle = _itertools.cycle(pool)
        return next(_gsb_key_cycle)


class _PermanentRejectError(Exception):
    """内容被下游 API 永久拒绝（如敏感词过滤 FAILED_PRECONDITION），重试无意义。

    由 _call_anthropic 在检测到 400 响应体含 'sensitive' / 'FAILED_PRECONDITION'
    时抛出；由 _call_with_retry / _call_judge_with_retry 捕获后立即停止重试。
    """


class GsbAbsoluteJudgeReward(RewardComponent):
    """基于 GSB 评测维度的 LLM 绝对打分。

    走和子 agent 一样的外部 Anthropic Messages API（http://ai-api.jdcloud.com/anthropic/v1），
    无需独立 judge vLLM 服务，API Key / URL / Model 与 config_train 完全对齐。

    输出 score ∈ [0, 1]，info 中附带各维度子分。
    调用或解析失败时重试，默认最多 50 次；全部失败后返回 default_on_error（默认 0）。

    截断策略：
        - query 保留末尾 max_query_chars 字符（多轮对话末尾最相关）
        - answer 保留末尾 max_answer_chars 字符
    400 Bad Request 最多重试 bad_request_max_retries 次（默认 10，含 usage limits 等），
    耗尽后返回 default_on_error，同时在 info 中标记 bad_request=1 便于排查。
    """

    name = "gsb_judge"

    def __init__(
        self,
        name: Optional[str] = None,
        enabled: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: int = 120,
        max_query_chars: int = 8000,
        max_answer_chars: int = 8000,
        default_on_error: float = 0.0,
        max_retries: int = 50,
        retry_delay: float = 3.0,
        bad_request_max_retries: int = 10,
        judge_io_log_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, enabled=enabled, **kwargs)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = int(timeout)
        self.max_query_chars = int(max_query_chars)
        self.max_answer_chars = int(max_answer_chars)
        self.default_on_error = float(default_on_error)
        self.max_retries = int(max_retries)
        self.retry_delay = float(retry_delay)
        self.bad_request_max_retries = int(bad_request_max_retries)
        # 若未显式指定，读环境变量 GSB_JUDGE_IO_LOG_DIR
        self._io_log_dir: Optional[str] = judge_io_log_dir or os.environ.get("GSB_JUDGE_IO_LOG_DIR") or None
        self._io_log_lock = threading.Lock()

    def score_answer(
        self,
        query: str,
        answer: str,
        num_action_steps: int = 0,
    ) -> tuple[float, Dict[str, float], bool]:
        """对单条 answer 进行 5 维度绝对打分。供 GsbPairwiseJudgeReward 复用。

        返回 (score ∈ [0,1], dims, ok)；ok=False 表示 judge 调用失败。
        """
        if not answer.strip():
            return 0.0, {}, True
        if len(query) > self.max_query_chars:
            query = "...[truncated]\n" + query[-self.max_query_chars:]
        if len(answer) > self.max_answer_chars:
            answer = "...[truncated]\n" + answer[-self.max_answer_chars:]
        user = build_judge_user_prompt(query, answer, num_action_steps=num_action_steps)
        raw, score, dims, parse_ok, attempts, retries_exhausted, bad_request = (
            self._call_judge_with_retry(system=JUDGE_SYSTEM_PROMPT, user=user)
        )
        ok = not retries_exhausted and not bad_request
        score = max(0.0, min(1.0, float(score)))
        return score, dims, ok

    def compute(self, ctx: RewardContext) -> RewardResult:
        s = ctx.summary
        # 优先从第一步的完整 messages 中重建对话上下文（含历史多轮），
        # 退化到 s.query（仅最后一条 user 消息）。
        query = _build_conversation_context(s)
        answer = (s.final_answer or "").strip()
        if not answer:
            return RewardResult(value=0.0, info={"skipped": "empty_answer"})

        # 保留末尾（多轮对话越靠后越相关；final_answer 末尾含结论）
        if len(query) > self.max_query_chars:
            query = "...[truncated]\n" + query[-self.max_query_chars:]
        if len(answer) > self.max_answer_chars:
            answer = "...[truncated]\n" + answer[-self.max_answer_chars:]

        user = build_judge_user_prompt(
            query, answer, num_action_steps=s.num_action_steps
        )

        raw, score, dims, parse_ok, attempts, retries_exhausted, bad_request = (
            self._call_judge_with_retry(system=JUDGE_SYSTEM_PROMPT, user=user)
        )

        if bad_request:
            return RewardResult(
                value=self.default_on_error,
                info={
                    "bad_request": 1,
                    "retry_attempts": attempts,
                    "parse_ok": 0,
                    "error": "bad_request_400",
                    "query_chars": len(query),
                    "answer_chars": len(answer),
                    "user_prompt_chars": len(user),
                },
            )

        if retries_exhausted:
            return RewardResult(
                value=self.default_on_error,
                info={
                    "retries_exhausted": 1,
                    "retry_attempts": attempts,
                    "parse_ok": 0,
                    "error": "max_retries_exhausted",
                    "raw": raw or "",
                },
            )

        score = max(0.0, min(1.0, float(score)))
        info: Dict[str, Any] = {
            "raw": raw,
            "parse_ok": int(parse_ok),
            "retries_exhausted": 0,
            "bad_request": 0,
            "retry_attempts": attempts,
        }
        for k, v in dims.items():
            info[f"dim_{k}"] = v
        return RewardResult(value=score, info=info)

    def _call_judge_with_retry(
        self, system: str, user: str
    ) -> tuple[str, float, Dict[str, float], bool, int, bool, bool]:
        """返回 (raw, score, dims, parse_ok, attempts_used, retries_exhausted, bad_request)。

        400 分两类处理：
          - 敏感词（_PermanentRejectError）：立即停止，标记 bad_request=True，不再重试
          - 配额/服务限制（555420等，普通 HTTPError 含"400"）：固定 2s 延迟短重试，
            最多 bad_request_max_retries 次
        429 限流：指数退避（retry_delay * 2^attempt，上限 60s）
        其他错误：固定 retry_delay 后重试
        """
        last_error = ""
        bad_request_attempts = 0
        for attempt in range(1, self.max_retries + 1):
            is_rate_limited = False
            is_bad_request = False
            try:
                raw = self._call_anthropic(system=system, user=user)
                score, dims, parse_ok = _parse_gsb_score(raw)
                if parse_ok:
                    return raw, score, dims, True, attempt, False, False
                last_error = "parse_failed"
                logger.warning(
                    "[gsb_judge] parse failed on attempt %d/%d, raw=%r",
                    attempt, self.max_retries, raw,
                )
            except _PermanentRejectError as exc:
                # 敏感词被拦截，重试无意义，立即终止
                logger.warning("[gsb_judge] %s", exc)
                return "", self.default_on_error, {}, False, attempt, False, True
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                exc_str = str(exc)
                is_rate_limited = "429" in exc_str
                is_bad_request = "400" in exc_str

                if is_bad_request:
                    bad_request_attempts += 1
                    if bad_request_attempts >= self.bad_request_max_retries:
                        logger.error(
                            "[gsb_judge] 400(配额/服务)，%d 次重试已耗尽: %s | "
                            "user_prompt_chars=%d",
                            self.bad_request_max_retries, exc, len(user),
                        )
                        return "", self.default_on_error, {}, False, attempt, False, True
                    logger.warning(
                        "[gsb_judge] 400(配额/服务) attempt %d/%d: %s | user_prompt_chars=%d",
                        bad_request_attempts, self.bad_request_max_retries, exc, len(user),
                    )
                else:
                    logger.warning(
                        "[gsb_judge] call failed on attempt %d/%d: %s",
                        attempt, self.max_retries, exc,
                    )

            if attempt < self.max_retries:
                if is_rate_limited:
                    # 429 限流：指数退避，上限 60s
                    delay = min(self.retry_delay * (2 ** (attempt - 1)), 60.0)
                    logger.warning(
                        "[gsb_judge] rate-limited (429), backing off %.1fs before retry %d",
                        delay, attempt + 1,
                    )
                elif is_bad_request:
                    # 400 配额/服务限制：固定 2s 短重试（不需要指数退避）
                    delay = 2.0
                    logger.warning(
                        "[gsb_judge] 400(配额/服务), retry in %.1fs (attempt %d/%d)",
                        delay, bad_request_attempts, self.bad_request_max_retries,
                    )
                else:
                    delay = min(self.retry_delay, 30.0)
                time.sleep(delay)

        logger.warning(
            "[gsb_judge] all %d retries exhausted, last_error=%s",
            self.max_retries, last_error,
        )
        return "", self.default_on_error, {}, False, self.max_retries, True, False

    def _call_anthropic(self, system: str, user: str) -> str:
        """调用 Anthropic Messages API，返回原始文本。

        400 错误细分：
          - 敏感词拦截（FAILED_PRECONDITION / "sensitive"）→ 抛 _PermanentRejectError，不重试
          - 其他 400（如 555420 配额限制）→ 正常抛 HTTPError，由重试循环短延迟重试
        """
        import requests as _requests

        base_url, _, model = _get_anthropic_config()
        url = f"{base_url.rstrip('/')}/messages"
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_next_gsb_api_key()}",
            "Trace-Id": "gsb-reward-judge",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": max(0.0, min(1.0, self.temperature)),
        }
        logger.debug(
            "[gsb_judge] request | model=%s | system=%s | user=%s",
            model, system, user,
        )
        resp = _requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        if not resp.ok:
            try:
                err_body: Any = resp.json()
            except Exception:
                err_body = resp.text
            self._write_io_log(
                payload,
                {"__error__": True, "status_code": resp.status_code, "body": err_body},
            )
            if resp.status_code == 400:
                err_str = json.dumps(err_body, ensure_ascii=False) if isinstance(err_body, dict) else str(err_body)
                if "sensitive" in err_str.lower() or "FAILED_PRECONDITION" in err_str:
                    raise _PermanentRejectError(
                        f"[gsb_judge] 敏感词拦截，不重试 | chars={len(user)} | detail={err_str[:200]}"
                    )
        resp.raise_for_status()
        data = resp.json()
        raw = str(data["content"][0]["text"])
        logger.debug("[gsb_judge] response | raw=%s", raw)
        self._write_io_log(payload, data)
        return raw

    def _write_io_log(self, payload: Dict[str, Any], response: Any) -> None:
        """把原始请求体和完整响应 JSON 追加写入 JSONL 日志。"""
        if not self._io_log_dir:
            return
        try:
            os.makedirs(self._io_log_dir, exist_ok=True)
            log_path = os.path.join(self._io_log_dir, "gsb_judge_io.jsonl")
            # 去掉 Authorization，避免 key 落盘
            safe_payload = {k: v for k, v in payload.items() if k != "Authorization"}
            entry = json.dumps(
                {
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "request": safe_payload,
                    "response": response,
                },
                ensure_ascii=False,
            )
            # 多进程安全：fcntl 文件锁 + 线程锁双保险
            with self._io_log_lock:
                with open(log_path, "a", encoding="utf-8") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    try:
                        f.write(entry + "\n")
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[gsb_judge] io_log write failed: %s", exc)


def _build_conversation_context(s: Any) -> str:
    """从 TrajectorySummary 的第一步 messages 中重建完整对话上下文。

    优先使用 s.steps[0].messages（第一次调用主脑时的完整 messages，含历史多轮），
    跳过 system 消息，将每条 user/assistant 消息拼成易读的对话文本。
    若 steps 为空则退化为 s.query（仅最后一条 user 消息）。
    """
    try:
        if s.steps and s.steps[0].messages:
            lines: List[str] = []
            for msg in s.steps[0].messages:
                role = (msg.get("role") or "").strip()
                content = (msg.get("content") or "").strip()
                if not content or role == "system":
                    continue
                label = "用户" if role == "user" else "助手"
                lines.append(f"{label}：{content}")
            if lines:
                return "\n".join(lines)
    except Exception:  # noqa: BLE001
        pass
    return (s.query or "").strip()


_GSB_KNOWN_DIMS = ("intent", "accuracy", "completeness", "practicality", "expression")
_GSB_DIM_RE = re.compile(
    r'"(' + "|".join(_GSB_KNOWN_DIMS) + r')"\s*:\s*([0-9]*\.?[0-9]+)'
)


def _compute_score_from_dims(dims: Dict[str, float]) -> float:
    """对 5 个已知维度做等权平均，缺失维度不计入分母。"""
    values = [dims[k] for k in _GSB_KNOWN_DIMS if k in dims]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _parse_gsb_score(text: str) -> tuple[float, Dict[str, float], bool]:
    """从 judge 输出中提取 (score, dimensions, parse_ok)。

    score 由代码对 5 个维度等权平均计算，不依赖 judge 输出的 score 字段。

    解析顺序（逐步宽松）：
    1. json.loads 完整解析，提取 dimensions 子字典
    2. 正则取首个 {...} 块再 json.loads
    3. 定向正则：只要能从文本中找到全部 5 个维度分数即视为成功
    4. 宽松正则：找到部分维度但不足 5 个时，parse_ok=False，触发重试
    """
    if not isinstance(text, str):
        return 0.0, {}, False
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()

    # 1. 完整 JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "dimensions" in obj:
            dims = _extract_dimensions(obj["dimensions"])
            if all(k in dims for k in _GSB_KNOWN_DIMS):
                return _compute_score_from_dims(dims), dims, True
    except Exception:  # noqa: BLE001
        pass

    # 2. 取首个 {...} 块再解析
    brace_m = re.search(r"\{[\s\S]*\}", text)
    if brace_m:
        try:
            obj = json.loads(brace_m.group(0))
            if isinstance(obj, dict) and "dimensions" in obj:
                dims = _extract_dimensions(obj["dimensions"])
                if all(k in dims for k in _GSB_KNOWN_DIMS):
                    return _compute_score_from_dims(dims), dims, True
        except Exception:  # noqa: BLE001
            pass

    # 3. 定向正则：从文本中直接提取所有维度分数
    dim_matches: Dict[str, float] = {
        dim: float(v)
        for dim, v in _GSB_DIM_RE.findall(text)
    }
    if all(k in dim_matches for k in _GSB_KNOWN_DIMS):
        return _compute_score_from_dims(dim_matches), dim_matches, True

    # 4. 宽松兜底：找到部分维度但不全，parse_ok=False 触发重试
    if dim_matches:
        return _compute_score_from_dims(dim_matches), dim_matches, False

    return 0.0, {}, False


def _extract_dimensions(raw: Any) -> Dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# 8) AiToCGatedJudgeReward —— 门控复合：step_efficiency × gsb_judge
# ---------------------------------------------------------------------------


class AiToCGatedJudgeReward(RewardComponent):
    """ai_to_c GRPO：任务完成门控 × 步数效率 × GSB judge 质量分。

    公式：
        未完成（含弃用 transfer-human/planner）→ 0.0，且 skip judge 调用
        完成 → reward = step_score × gsb_score
            gsb_score = 5 个维度（各 0/1）的等权平均，取值 {0, 0.2, 0.4, 0.6, 0.8, 1.0}

    Judge 走与子 agent 完全相同的外部 Anthropic API，无需单独起 judge vLLM。
    通过环境变量 ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY_POOL / ANTHROPIC_MODEL 控制。

    Args:
        target_steps / max_steps: 步数效率参数（同 AiToCEfficientReward）
        judge_* : 透传给 GsbAbsoluteJudgeReward（temperature/max_tokens/timeout 等）
    """

    name = "ai_to_c_gated_judge"

    def __init__(
        self,
        name: Optional[str] = None,
        enabled: bool = True,
        target_steps: int = 3,
        max_steps: int = 7,
        judge_temperature: float = 0.0,
        judge_max_tokens: int = 4096,
        judge_timeout: int = 120,
        max_query_chars: int = 8000,
        max_answer_chars: int = 8000,
        default_on_error: float = 0.0,
        judge_max_retries: int = 50,
        judge_retry_delay: float = 3.0,
        judge_io_log_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, enabled=enabled, **kwargs)
        if target_steps >= max_steps:
            raise ValueError(
                f"target_steps({target_steps}) must be < max_steps({max_steps})"
            )
        self.target_steps = int(target_steps)
        self.max_steps = int(max_steps)
        self._efficient = AiToCEfficientReward(
            target_steps=target_steps,
            max_steps=max_steps,
            enabled=True,
        )
        self._judge = GsbAbsoluteJudgeReward(
            temperature=judge_temperature,
            max_tokens=judge_max_tokens,
            timeout=judge_timeout,
            max_query_chars=max_query_chars,
            max_answer_chars=max_answer_chars,
            default_on_error=default_on_error,
            max_retries=judge_max_retries,
            retry_delay=judge_retry_delay,
            judge_io_log_dir=judge_io_log_dir,
            enabled=enabled,
        )

    def compute(self, ctx: RewardContext) -> RewardResult:
        s = ctx.summary
        n = s.num_action_steps

        completed, comp_info = is_valid_task_completed(s)
        if not completed:
            return RewardResult(
                value=0.0,
                info={
                    "task_completed": 0,
                    "step_score": 0.0,
                    "gsb_score": 0.0,
                    "num_action_steps": n,
                    "judge_skipped": 1,
                    **comp_info,
                },
            )

        if n <= self.target_steps:
            step_score = 1.0
        elif n >= self.max_steps:
            step_score = 0.0
        else:
            step_score = 1.0 - (n - self.target_steps) / (self.max_steps - self.target_steps)

        judge_res = self._judge.compute(ctx)
        gsb_score = float(judge_res.value)
        judge_failed = bool(judge_res.info.get("retries_exhausted", 0)) or bool(
            judge_res.info.get("bad_request", 0)
        )
        if judge_failed:
            total = float("nan")
        else:
            total = step_score * gsb_score

        info: Dict[str, Any] = {
            "task_completed": 1,
            "step_score": step_score,
            "gsb_score": gsb_score,
            "judge_masked": int(judge_failed),
            "num_action_steps": n,
            "judge_skipped": 0,
            "judge_called": 1,
            "has_valid_final": 1,
            "deprecated_terminal": 0,
            "retries_exhausted": int(judge_res.info.get("retries_exhausted", 0)),
            "bad_request": int(judge_res.info.get("bad_request", 0)),
            "retry_attempts": int(judge_res.info.get("retry_attempts", 0)),
        }
        for k, v in (judge_res.info or {}).items():
            info[f"judge_{k}"] = v
        return RewardResult(value=total, info=info)


# ---------------------------------------------------------------------------
# 9) GsbPairwiseJudgeReward —— 相对对比打分（模型 vs baseline，gsb_pairwise prompt）
# ---------------------------------------------------------------------------

# G/S/B → model_win ∈ {+1, 0, -1}（以模型为系统B视角）
_GSB_TO_WIN: Dict[str, int] = {"G": 1, "S": 0, "B": -1}
_GSB_PARSE_RE = re.compile(r'"gsb"\s*:\s*"([GSBgsb])"')


def _parse_gsb_response(raw: str) -> tuple[str, str]:
    """从 judge 输出中提取 (gsb, reason)，兜底返回 ("S", ...)。"""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        obj = json.loads(text)
        gsb = str(obj.get("gsb", "")).strip().upper()
        if gsb in ("G", "S", "B"):
            return gsb, str(obj.get("reason", "")).strip()
    except Exception:  # noqa: BLE001
        pass
    m = _GSB_PARSE_RE.search(text)
    if m:
        return m.group(1).upper(), ""
    brace_m = re.search(r"\{[\s\S]*?\}", text)
    if brace_m:
        try:
            obj = json.loads(brace_m.group(0))
            gsb = str(obj.get("gsb", "")).strip().upper()
            if gsb in ("G", "S", "B"):
                return gsb, str(obj.get("reason", "")).strip()
        except Exception:  # noqa: BLE001
            pass
    return "S", f"parse_failed: {raw[:100]}"


# 4 个评分维度（与 gsb_pairwise.py prompt 对齐，已删除 accuracy）
_DIM_KEYS = ("intent", "completeness", "practicality", "expression")
_VALID_DIM_SCORES = frozenset({-2, -1, 0, 1, 2})


def _normalize_json_scores(text: str) -> str:
    """将 scores 字段中 +N 形式的正数替换为合法 JSON 数字（去掉 + 号）。

    标准 JSON 不支持 +1 / +2 这类带正号的数字，但 judge LLM 有时会输出它们。
    只处理 "scores" 对象内的数值，避免误替换 reason 字段中的 + 号。
    """
    def _fix_scores_block(match: re.Match) -> str:
        block = match.group(0)
        block = re.sub(r":\s*\+(\d+)", lambda m2: ": " + m2.group(1), block)
        return block

    text = re.sub(r'"scores"\s*:\s*\{[^}]*\}', _fix_scores_block, text)
    return text


def _parse_dim_scores(raw: str) -> tuple[Optional[float], Dict[str, int], str]:
    """从新版 judge 输出中提取 5 维度 B-A 差值并取均值，返回 (avg_score, dim_scores, reason)。

    avg_score ∈ [-2.0, 2.0]，解析失败返回 (None, {}, err_msg)。
    dim_scores: 各维度 B-A 差值 ∈ {-2,-1,0,1,2}。
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    # 去掉 scores 块中的 +N（标准 JSON 不支持正号前缀）
    text = _normalize_json_scores(text)

    def _extract(obj: Any) -> tuple[Optional[float], Dict[str, int], str]:
        scores_raw = obj.get("scores")
        if not isinstance(scores_raw, dict):
            return None, {}, "missing_scores_dict"
        vals: list[int] = []
        dim_scores: Dict[str, int] = {}
        for dim in _DIM_KEYS:
            v = scores_raw.get(dim)
            if v is None:
                return None, {}, f"missing_dim_{dim}"
            try:
                vi = int(v)
            except (TypeError, ValueError):
                return None, {}, f"invalid_dim_{dim}={v!r}"
            if vi not in _VALID_DIM_SCORES:
                return None, {}, f"out_of_range_dim_{dim}={vi}"
            vals.append(vi)
            dim_scores[dim] = vi
        reason = str(obj.get("reason", "")).strip()
        return sum(vals) / len(vals), dim_scores, reason

    try:
        obj = json.loads(text)
        avg, dim_scores, reason = _extract(obj)
        if avg is not None:
            return avg, dim_scores, reason
    except Exception:  # noqa: BLE001
        pass

    brace_m = re.search(r"\{[\s\S]*\}", text)
    if brace_m:
        try:
            obj = json.loads(brace_m.group(0))
            avg, dim_scores, reason = _extract(obj)
            if avg is not None:
                return avg, dim_scores, reason
        except Exception:  # noqa: BLE001
            pass

    # 最终兜底：按维度逐一用 regex 抠整数，支持 +N / -N / N 格式
    scores_by_re: dict[str, int] = {}
    for dim in _DIM_KEYS:
        m = re.search(rf'"{dim}"\s*:\s*([+-]?\d+)', text)
        if m:
            try:
                v = int(m.group(1))
                if v in _VALID_DIM_SCORES:
                    scores_by_re[dim] = v
            except (TypeError, ValueError):
                pass
    if len(scores_by_re) == len(_DIM_KEYS):
        vals = [scores_by_re[d] for d in _DIM_KEYS]
        return sum(vals) / len(vals), scores_by_re, "[reason_skipped]"

    return None, {}, f"parse_failed: {raw[:100]}"


def _merge_dim_advantages(
    fwd_dims: Dict[str, int],
    rev_dims: Dict[str, int],
    *,
    reverse_used: bool,
) -> Dict[str, float]:
    """合并正反双跑的各维度差值，得到 model-baseline 视角的 per-dim advantage。"""
    if not fwd_dims:
        return {}
    if not reverse_used:
        return {dim: float(fwd_dims[dim]) for dim in _DIM_KEYS if dim in fwd_dims}
    out: Dict[str, float] = {}
    for dim in _DIM_KEYS:
        if dim in fwd_dims and dim in rev_dims:
            out[dim] = (float(fwd_dims[dim]) - float(rev_dims[dim])) / 2.0
    return out


class GsbPairwiseJudgeReward(RewardComponent):
    """相对对比打分：模型答案 vs extra_info["baseline_answer"]，使用 gsb_pairwise prompt。

    judge 对 5 个维度各输出 B 相对 A 的差值（{-2,-1,0,1,2}），取均值得 raw_score ∈ [-2, 2]。

    正向（A=baseline, B=model）和反向（A=model, B=baseline）各跑一次，
    取平均消除位置偏差：
        score_fwd        = avg_dims(B-A) where B=model    → model-baseline 视角
        score_rev_raw    = avg_dims(B-A) where B=baseline → baseline-model 视角
        score_rev_flipped = -score_rev_raw                → model-baseline 视角
        model_advantage  = (score_fwd + score_rev_flipped) / 2  ∈ [-2, 2]
        pair_score       = (model_advantage + 2) / 4            ∈ [0, 1]

    reverse=False 时退化为单向：
        pair_score = (score_fwd + 2) / 4 ∈ [0, 1]
    反向调用失败时自动降级为单向。

    no_baseline_fallback：
        "nan"  → 缺 baseline 时返回 NaN（GRPO mask，默认）
        "skip" → 返回 0.5（视为平局）
    """

    name = "gsb_pairwise"

    def __init__(
        self,
        name: Optional[str] = None,
        enabled: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: int = 120,
        max_query_chars: int = 8000,
        max_answer_chars: int = 8000,
        max_retries: int = 50,
        retry_delay: float = 3.0,
        bad_request_max_retries: int = 10,
        reverse: bool = True,
        no_baseline_fallback: str = "nan",
        judge_io_log_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, enabled=enabled, **kwargs)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = int(timeout)
        self.max_query_chars = int(max_query_chars)
        self.max_answer_chars = int(max_answer_chars)
        self.max_retries = int(max_retries)
        self.retry_delay = float(retry_delay)
        self.bad_request_max_retries = int(bad_request_max_retries)
        self.reverse = bool(reverse)
        import math as _math
        if isinstance(no_baseline_fallback, float) and _math.isnan(no_baseline_fallback):
            no_baseline_fallback = "nan"
        no_baseline_fallback = str(no_baseline_fallback)
        if no_baseline_fallback not in ("nan", "skip"):
            raise ValueError("no_baseline_fallback must be 'nan' or 'skip'")
        self.no_baseline_fallback = no_baseline_fallback
        self._io_log_dir: Optional[str] = judge_io_log_dir or os.environ.get("GSB_JUDGE_IO_LOG_DIR") or None
        self._io_log_lock = threading.Lock()

    def compute(self, ctx: RewardContext) -> RewardResult:
        s = ctx.summary
        query = _build_conversation_context(s)
        model_answer = (s.final_answer or "").strip()

        baseline_answer = ""
        if isinstance(ctx.extra_info, dict):
            baseline_answer = str(ctx.extra_info.get("baseline_answer") or "").strip()

        if not baseline_answer:
            if self.no_baseline_fallback == "nan":
                return RewardResult(
                    value=float("nan"),
                    info={"skipped": "no_baseline_answer", "pair_score": float("nan")},
                )
            return RewardResult(
                value=0.5,
                info={"skipped": "no_baseline_answer_fallback_skip", "pair_score": 0.5},
            )

        if not model_answer:
            return RewardResult(value=0.0, info={"skipped": "empty_model_answer", "pair_score": 0.0})

        if len(query) > self.max_query_chars:
            query = "...[truncated]\n" + query[-self.max_query_chars:]
        if len(model_answer) > self.max_answer_chars:
            model_answer = "...[truncated]\n" + model_answer[-self.max_answer_chars:]
        if len(baseline_answer) > self.max_answer_chars:
            baseline_answer = "...[truncated]\n" + baseline_answer[-self.max_answer_chars:]

        from .gsb_pairwise import (
            JUDGE_SYSTEM_PROMPT as _PAIR_SYS,
            build_pairwise_judge_prompt,
            build_pairwise_judge_prompt_reversed,
        )

        # 正向：A=baseline, B=model → score = model-baseline 视角
        score_fwd, fwd_dims, fwd_reason, fwd_attempts, fwd_failed = self._call_with_retry(
            _PAIR_SYS, build_pairwise_judge_prompt(query, baseline_answer, model_answer)
        )
        if fwd_failed:
            return RewardResult(
                value=float("nan"),
                info={
                    "judge_called": 1,
                    "judge_failed": 1,
                    "fwd_attempts": fwd_attempts,
                    "pair_score": float("nan"),
                },
            )

        if not self.reverse:
            pair_score = (score_fwd + 2.0) / 4.0      # [-2,2] → [0,1]
            dim_adv = _merge_dim_advantages(fwd_dims, {}, reverse_used=False)
            info: Dict[str, Any] = {
                "judge_called": 1,
                "pair_score": pair_score,
                "score_fwd": score_fwd,
                "fwd_reason": fwd_reason[:200],
                "fwd_attempts": fwd_attempts,
                "reverse_used": 0,
            }
            for dim, val in dim_adv.items():
                info[f"dim_{dim}"] = val
            return RewardResult(value=pair_score, info=info)

        # 反向：A=model, B=baseline → score = baseline-model 视角，取反得 model-baseline
        score_rev_raw, rev_dims, rev_reason, rev_attempts, rev_failed = self._call_with_retry(
            _PAIR_SYS, build_pairwise_judge_prompt_reversed(query, baseline_answer, model_answer)
        )
        if rev_failed:
            pair_score = (score_fwd + 2.0) / 4.0
            dim_adv = _merge_dim_advantages(fwd_dims, {}, reverse_used=False)
            info = {
                "judge_called": 1,
                "pair_score": pair_score,
                "score_fwd": score_fwd,
                "fwd_reason": fwd_reason[:200],
                "rev_failed": 1,
                "rev_attempts": rev_attempts,
                "reverse_used": 0,
            }
            for dim, val in dim_adv.items():
                info[f"dim_{dim}"] = val
            return RewardResult(value=pair_score, info=info)

        score_rev_flipped = -score_rev_raw             # 转为 model-baseline 视角
        model_advantage = (score_fwd + score_rev_flipped) / 2.0   # ∈ [-2, 2]
        pair_score = (model_advantage + 2.0) / 4.0                # ∈ [0, 1]
        dim_adv = _merge_dim_advantages(fwd_dims, rev_dims, reverse_used=True)

        info = {
            "judge_called": 1,
            "pair_score": pair_score,
            "model_advantage": model_advantage,
            "score_fwd": score_fwd,
            "score_rev_raw": score_rev_raw,
            "score_rev_flipped": score_rev_flipped,
            "fwd_reason": fwd_reason[:200],
            "rev_reason": rev_reason[:200],
            "fwd_attempts": fwd_attempts,
            "rev_attempts": rev_attempts,
            "reverse_used": 1,
        }
        for dim, val in dim_adv.items():
            info[f"dim_{dim}"] = val
        return RewardResult(value=pair_score, info=info)

    def _call_with_retry(
        self, system: str, user_prompt: str
    ) -> tuple[float, Dict[str, int], str, int, bool]:
        """返回 (avg_dim_score, dim_scores, reason, attempts, failed)。

        avg_dim_score ∈ [-2.0, 2.0]：5 维度 B-A 差值的均值；failed=True 时值无意义。

        400 分两类处理：
          - 敏感词（_PermanentRejectError）：立即停止，failed=True，不再重试
          - 配额/服务限制（555420等，普通 HTTPError 含"400"）：固定 2s 延迟短重试，
            最多 bad_request_max_retries 次
        429 限流：指数退避（retry_delay * 2^attempt，上限 60s）
        其他错误：固定 retry_delay 后重试
        """
        last_err = ""
        bad_request_attempts = 0
        for attempt in range(1, self.max_retries + 1):
            is_rate_limited = False
            is_bad_request = False
            try:
                raw = self._call_anthropic(system, user_prompt)
                avg_score, dim_scores, reason = _parse_dim_scores(raw)
                if avg_score is not None:
                    return avg_score, dim_scores, reason, attempt, False
                last_err = f"parse_failed: {raw[:80]}"
                logger.warning("[gsb_pairwise] parse failed attempt %d/%d raw=%r", attempt, self.max_retries, raw)
            except _PermanentRejectError as exc:
                # 敏感词被拦截，重试无意义，立即终止
                logger.warning("[gsb_pairwise] %s", exc)
                return 0.0, {}, "", attempt, True
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
                exc_str = str(exc)
                is_rate_limited = "429" in exc_str
                is_bad_request = "400" in exc_str
                if is_bad_request:
                    bad_request_attempts += 1
                    if bad_request_attempts >= self.bad_request_max_retries:
                        logger.error(
                            "[gsb_pairwise] 400(配额/服务)，%d 次重试已耗尽: %s",
                            self.bad_request_max_retries, exc,
                        )
                        return 0.0, {}, "", attempt, True
                    logger.warning(
                        "[gsb_pairwise] 400(配额/服务) attempt %d/%d: %s",
                        bad_request_attempts, self.bad_request_max_retries, exc,
                    )
                else:
                    logger.warning("[gsb_pairwise] call failed attempt %d/%d: %s", attempt, self.max_retries, exc)
            if attempt < self.max_retries:
                if is_rate_limited:
                    # 429 限流：指数退避，上限 60s
                    delay = min(self.retry_delay * (2 ** (attempt - 1)), 60.0)
                    logger.warning(
                        "[gsb_pairwise] rate-limited (429), backing off %.1fs before retry %d",
                        delay, attempt + 1,
                    )
                elif is_bad_request:
                    # 400 配额/服务限制：固定 2s 短重试（不需要指数退避）
                    delay = 2.0
                    logger.warning(
                        "[gsb_pairwise] 400(配额/服务), retry in %.1fs (attempt %d/%d)",
                        delay, bad_request_attempts, self.bad_request_max_retries,
                    )
                else:
                    delay = min(self.retry_delay, 30.0)
                time.sleep(delay)
        logger.warning("[gsb_pairwise] all %d retries exhausted: %s", self.max_retries, last_err)
        return 0.0, {}, "", self.max_retries, True

    def _call_anthropic(self, system: str, user_prompt: str) -> str:
        """调用 Anthropic Messages API，返回原始文本。

        400 错误细分：
          - 敏感词拦截（FAILED_PRECONDITION / "sensitive"）→ 抛 _PermanentRejectError，不重试
          - 其他 400（如 555420 配额限制）→ 正常抛 HTTPError，由重试循环短延迟重试
        """
        import requests as _requests

        base_url, _, model = _get_anthropic_config()
        url = f"{base_url.rstrip('/')}/messages"
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_next_gsb_api_key()}",
            "Trace-Id": "gsb-pairwise-reward",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": max(0.0, min(1.0, self.temperature)),
        }
        resp = _requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        if not resp.ok:
            try:
                err_body: Any = resp.json()
            except Exception:
                err_body = resp.text
            self._write_io_log(payload, {"__error__": True, "status_code": resp.status_code, "body": err_body})
            if resp.status_code == 400:
                err_str = json.dumps(err_body, ensure_ascii=False) if isinstance(err_body, dict) else str(err_body)
                if "sensitive" in err_str.lower() or "FAILED_PRECONDITION" in err_str:
                    raise _PermanentRejectError(
                        f"[gsb_pairwise] 敏感词拦截，不重试 | chars={len(user_prompt)} | detail={err_str[:200]}"
                    )
        resp.raise_for_status()
        data = resp.json()
        raw = str(data["content"][0]["text"])
        self._write_io_log(payload, data)
        return raw

    def _write_io_log(self, payload: Dict[str, Any], response: Any) -> None:
        if not self._io_log_dir:
            return
        try:
            os.makedirs(self._io_log_dir, exist_ok=True)
            log_path = os.path.join(self._io_log_dir, "gsb_pairwise_io.jsonl")
            safe_payload = {k: v for k, v in payload.items() if k != "Authorization"}
            entry = json.dumps(
                {"ts": time.time(), "pid": os.getpid(), "request": safe_payload, "response": response},
                ensure_ascii=False,
            )
            with self._io_log_lock:
                with open(log_path, "a", encoding="utf-8") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    try:
                        f.write(entry + "\n")
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[gsb_pairwise] io_log write failed: %s", exc)


# ---------------------------------------------------------------------------
# 10) AiToCGatedPairwiseReward —— 完成门控 × step_efficiency × pair_score
# ---------------------------------------------------------------------------


class AiToCGatedPairwiseReward(RewardComponent):
    """ai_to_c GRPO：ReAct 格式门 × 任务完成门控 × 步数效率 × 无效步折扣 × GSB 相对对比分。

    门控顺序（任意一关失败即返回 0.0，跳过后续 judge）：
        1. 格式门：任意步 response_text 出现 multi_thought 或裸 </think> → reward=0.0
        2. 完成门：is_final=False → reward=0.0
        3. 步数效率 × 无效步折扣 × GSB pairwise 打分

    公式（通过前两关后）：
        reward = step_score × pair_score × efficiency_factor
            step_score ∈ [0, 1]（use_step_score=False 时固定 1.0）
            pair_score ∈ {0, 0.25, 0.5, 0.75, 1.0}（reverse=True，正反双跑）
            efficiency_factor = clip(1 - λ·min(num_wasted_steps, cap), 0, 1)
                num_wasted_steps: builder 规则识别的空跑步数
                    （resolve 真空跑、search 无结果、obs_error 类参数/前置校验错）
                λ = waste_penalty_lambda（默认 0.15）
                cap = waste_penalty_cap（默认 2，最多罚到第 2 个空跑步）
                worst case: 1 - 0.15×2 = 0.70（不会归零）

    Judge 使用 gsb_pairwise prompt（系统A vs 系统B 相对对比）。
    extra_info["baseline_answer"] 缺失时：
        no_baseline_fallback="nan" → NaN → GRPO mask（默认）
        no_baseline_fallback="skip" → reward=0.5×step_score（视为平局）

    info 字段：
        format_violation: 0/1（始终记录，供 SwanLab 监控格式失败率）
        num_wasted_steps: 规则识别的无效步数
        efficiency_factor: 无效步折扣因子 ∈ [0, 1]

    Args:
        target_steps / max_steps: 步数效率参数（同 AiToCEfficientReward）
        reverse: True=正反双跑消除位置偏差（默认 True）
        no_baseline_fallback: "nan" 或 "skip"
        waste_penalty_lambda: 每个无效步的折扣率（默认 0.15）
        waste_penalty_cap: 最多惩罚到第几个无效步（默认 2，超出 cap 不再增加惩罚）
        judge_*: 透传给 GsbPairwiseJudgeReward
    """

    name = "ai_to_c_gated_pairwise"

    def __init__(
        self,
        name: Optional[str] = None,
        enabled: bool = True,
        target_steps: int = 3,
        max_steps: int = 7,
        use_completion_gate: bool = True,
        use_step_score: bool = True,
        reverse: bool = True,
        no_baseline_fallback: str = "nan",
        judge_temperature: float = 0.0,
        judge_max_tokens: int = 512,
        judge_timeout: int = 120,
        max_query_chars: int = 8000,
        max_answer_chars: int = 8000,
        judge_max_retries: int = 50,
        judge_retry_delay: float = 3.0,
        judge_io_log_dir: Optional[str] = None,
        waste_penalty_lambda: float = 0.15,
        waste_penalty_cap: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, enabled=enabled, **kwargs)
        if target_steps >= max_steps:
            raise ValueError(f"target_steps({target_steps}) must be < max_steps({max_steps})")
        self.target_steps = int(target_steps)
        self.max_steps = int(max_steps)
        self.use_completion_gate = bool(use_completion_gate)
        self.use_step_score = bool(use_step_score)
        self.waste_penalty_lambda = float(waste_penalty_lambda)
        self.waste_penalty_cap = int(waste_penalty_cap)
        self._judge = GsbPairwiseJudgeReward(
            temperature=judge_temperature,
            max_tokens=judge_max_tokens,
            timeout=judge_timeout,
            max_query_chars=max_query_chars,
            max_answer_chars=max_answer_chars,
            max_retries=judge_max_retries,
            retry_delay=judge_retry_delay,
            reverse=reverse,
            no_baseline_fallback=no_baseline_fallback,
            judge_io_log_dir=judge_io_log_dir,
            enabled=enabled,
        )

    def compute(self, ctx: RewardContext) -> RewardResult:
        import math
        s = ctx.summary
        n = s.num_action_steps

        # --- 格式门：任意步违规 → reward=0.0，跳过 judge（始终记录 format_violation）---
        format_violation = int(
            any(not _react_format_ok(step.response_text) for step in s.steps)
        )
        if format_violation:
            return RewardResult(
                value=0.0,
                info={
                    "format_violation": 1,
                    "task_completed": 0,
                    "step_score": 0.0,
                    "pair_score": 0.0,
                    "num_action_steps": n,
                    "judge_skipped": 1,
                },
            )

        # --- 任务完成门控（use_completion_gate=False 时跳过，直接调 judge）---
        completed, comp_info = is_valid_task_completed(s)
        if self.use_completion_gate and not completed:
            return RewardResult(
                value=0.0,
                info={
                    "format_violation": 0,
                    "task_completed": 0,
                    "step_score": 0.0,
                    "pair_score": 0.0,
                    "num_action_steps": n,
                    "judge_skipped": 1,
                    **comp_info,
                },
            )

        # --- 步数效率分（use_step_score=False 时固定为 1.0，不参与乘法）---
        if self.use_step_score:
            if n <= self.target_steps:
                step_score = 1.0
            elif n >= self.max_steps:
                step_score = 0.0
            else:
                step_score = 1.0 - (n - self.target_steps) / (self.max_steps - self.target_steps)
        else:
            step_score = 1.0  # 不使用步数效率，视为满分

        judge_res = self._judge.compute(ctx)
        pair_score = float(judge_res.value)  # NaN if judge failed / no baseline

        judge_failed = math.isnan(pair_score)

        # --- 无效步折扣（efficiency_factor）---
        # 公式：1 - λ · min(num_wasted_steps, cap)，结果 clip 到 [0, 1]
        # num_wasted_steps 由 builder 通过规则识别（resolve 真空跑、search 无结果等）
        num_wasted = int(getattr(s, "num_wasted_steps", 0) or 0)
        efficiency_factor = max(
            0.0,
            min(1.0, 1.0 - self.waste_penalty_lambda * min(num_wasted, self.waste_penalty_cap)),
        )

        if judge_failed:
            total = float("nan")
        else:
            total = step_score * pair_score * efficiency_factor

        info: Dict[str, Any] = {
            "format_violation": 0,
            "task_completed": int(completed),
            "use_step_score": int(self.use_step_score),
            "use_completion_gate": int(self.use_completion_gate),
            **({"step_score": step_score} if self.use_step_score else {}),
            "pair_score": pair_score,
            "num_wasted_steps": num_wasted,
            "efficiency_factor": efficiency_factor,
            "judge_masked": int(judge_failed),
            "num_action_steps": n,
            "judge_skipped": 0,
            "judge_called": 1,
            "has_valid_final": 1,
            "deprecated_terminal": 0,
        }
        for k, v in (judge_res.info or {}).items():
            info[f"judge_{k}"] = v
        return RewardResult(value=total, info=info)


# ---------------------------------------------------------------------------
# 11) AccuracyTrajectoryJudgeReward —— 全轨迹内容准确性 judge
# ---------------------------------------------------------------------------

_ACCURACY_SCORE_RE = re.compile(r'"accuracy"\s*:\s*([01])')


def _parse_accuracy_score(raw: str) -> tuple:
    """解析 accuracy judge 输出：{"accuracy": 0 or 1, "reason": "..."}。

    三层兜底：完整 JSON → 首个 {} 块 → 正则。
    返回 (score_int, reason_str) 或 (None, '') 表示解析失败。
    """
    if not isinstance(raw, str):
        return None, ""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()

    def _extract(obj: Dict[str, Any]):
        v = obj.get("accuracy")
        if v is None:
            return None, ""
        try:
            vi = int(v)
        except (TypeError, ValueError):
            return None, ""
        if vi not in (0, 1):
            return None, ""
        reason = str(obj.get("reason", "")).strip()
        return vi, reason

    # 1. 完整 JSON
    try:
        obj = json.loads(text)
        score, reason = _extract(obj)
        if score is not None:
            return score, reason
    except Exception:  # noqa: BLE001
        pass

    # 2. 首个 {} 块
    m = re.search(r"\{[\s\S]*?\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            score, reason = _extract(obj)
            if score is not None:
                return score, reason
        except Exception:  # noqa: BLE001
            pass

    # 3. 正则兜底
    m = _ACCURACY_SCORE_RE.search(text)
    if m:
        return int(m.group(1)), ""

    return None, ""


class AccuracyTrajectoryJudgeReward(RewardComponent):
    """全轨迹内容准确性 judge reward。

    将完整 ReAct 轨迹（工具调用+观测+最终回答）一次性发给 judge，
    专项评估产品信息准确性：
      - 产品名称、保额、保障责任、免责条款是否真实、无混淆
      - 是否存在捏造产品特性、错误引用条款
      - 具体产品问答是否有工具查询依据

    score = float(accuracy_0_1)  ∈ {0, 1}

    API 暂时失败时返回 default_on_error（默认 0.5），而非 NaN，
    避免因暂时性 API 故障 mask 掉有效的 pairwise 信号。

    IO 日志：优先写 ACCURACY_JUDGE_IO_LOG_DIR，fallback GSB_JUDGE_IO_LOG_DIR，
            文件名 accuracy_trajectory_io.jsonl（与 gsb_pairwise_io.jsonl 分开）。

    Args:
        max_traj_chars: 轨迹文本截断长度（默认 10000，保留末尾）
        max_answer_chars: 最终回答截断长度（默认 4000）
        max_tokens: judge 输出 max tokens（只需输出短 JSON，256 足够）
        default_on_error: API 暂时失败时的回退分数（默认 0.5，中性值）
    """

    name = "accuracy_trajectory"

    def __init__(
        self,
        name: Optional[str] = None,
        enabled: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout: int = 120,
        max_traj_chars: int = 10000,
        max_answer_chars: int = 4000,
        max_retries: int = 50,
        retry_delay: float = 3.0,
        bad_request_max_retries: int = 10,
        default_on_error: float = 0.5,
        judge_io_log_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, enabled=enabled, **kwargs)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = int(timeout)
        self.max_traj_chars = int(max_traj_chars)
        self.max_answer_chars = int(max_answer_chars)
        self.max_retries = int(max_retries)
        self.retry_delay = float(retry_delay)
        self.bad_request_max_retries = int(bad_request_max_retries)
        self.default_on_error = float(default_on_error)
        self._io_log_dir: Optional[str] = (
            judge_io_log_dir
            or os.environ.get("ACCURACY_JUDGE_IO_LOG_DIR")
            or os.environ.get("GSB_JUDGE_IO_LOG_DIR")
            or None
        )
        self._io_log_lock = threading.Lock()

    def compute(self, ctx: RewardContext) -> RewardResult:
        s = ctx.summary

        final_answer = (s.final_answer or "").strip()
        if not final_answer:
            return RewardResult(
                value=0.0,
                info={"skipped": "empty_answer", "accuracy_raw": 0, "accuracy_score": 0.0},
            )

        from .accuracy_trajectory import (
            ACCURACY_JUDGE_SYSTEM_PROMPT,
            build_accuracy_judge_prompt,
            extract_knowledge_from_steps,
        )

        # 1. 从轨迹中提取知识库工具 observation
        knowledge = extract_knowledge_from_steps(s.steps, max_chars=self.max_traj_chars)
        if not knowledge:
            # 轨迹中未调用任何知识库工具，accuracy 维度无参考依据，跳过评估
            return RewardResult(
                value=self.default_on_error,
                info={
                    "skipped": "no_knowledge_tools",
                    "accuracy_raw": None,
                    "accuracy_score": self.default_on_error,
                    "fallback": 1,
                },
            )

        # 2. 截断最终回答
        fa = final_answer
        if len(fa) > self.max_answer_chars:
            fa = "...[truncated]\n" + fa[-self.max_answer_chars:]

        # 3. 构建 prompt 并调用 judge
        user_prompt = build_accuracy_judge_prompt(s.query or "", knowledge, fa)
        raw_score, reason, attempts, failed = self._call_with_retry(
            ACCURACY_JUDGE_SYSTEM_PROMPT, user_prompt
        )

        if failed:
            # API 暂时失败 → 中性分，不 mask 样本（保留 pairwise 信号）
            return RewardResult(
                value=self.default_on_error,
                info={
                    "judge_failed": 1,
                    "judge_attempts": attempts,
                    "accuracy_raw": None,
                    "accuracy_score": self.default_on_error,
                    "fallback": 1,
                },
            )

        # 4. 二值分直接用：0 → 0.0，1 → 1.0
        score = float(raw_score)
        return RewardResult(
            value=score,
            info={
                "accuracy_raw": raw_score,
                "accuracy_score": score,
                "judge_reason": reason[:200],
                "judge_attempts": attempts,
                "judge_called": 1,
                "fallback": 0,
            },
        )

    def _call_with_retry(
        self, system: str, user_prompt: str
    ) -> tuple:
        """返回 (score_0_1: int, reason: str, attempts: int, failed: bool)。

        重试逻辑与 GsbPairwiseJudgeReward._call_with_retry 完全一致：
          - 敏感词（_PermanentRejectError）：立即停止
          - 429 限流：指数退避，上限 60s
          - 400（配额/服务）：固定 2s 短重试，最多 bad_request_max_retries 次
          - 其他：固定 retry_delay 后重试
        """
        last_err = ""
        bad_request_attempts = 0
        for attempt in range(1, self.max_retries + 1):
            is_rate_limited = False
            is_bad_request = False
            try:
                raw = self._call_anthropic(system, user_prompt)
                score, reason = _parse_accuracy_score(raw)
                if score is not None:
                    return score, reason, attempt, False
                last_err = f"parse_failed: {raw[:80]}"
                logger.warning(
                    "[accuracy] parse failed attempt %d/%d raw=%r",
                    attempt, self.max_retries, raw,
                )
            except _PermanentRejectError as exc:
                logger.warning("[accuracy] 敏感词拦截，停止重试: %s", exc)
                return 3, "", attempt, True
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
                exc_str = str(exc)
                is_rate_limited = "429" in exc_str
                is_bad_request = "400" in exc_str
                if is_bad_request:
                    bad_request_attempts += 1
                    if bad_request_attempts >= self.bad_request_max_retries:
                        logger.error(
                            "[accuracy] 400(配额/服务)，%d 次重试已耗尽: %s",
                            self.bad_request_max_retries, exc,
                        )
                        return 3, "", attempt, True
                    logger.warning(
                        "[accuracy] 400(配额/服务) attempt %d/%d: %s",
                        bad_request_attempts, self.bad_request_max_retries, exc,
                    )
                else:
                    logger.warning(
                        "[accuracy] call failed attempt %d/%d: %s",
                        attempt, self.max_retries, exc,
                    )
            if attempt < self.max_retries:
                if is_rate_limited:
                    delay = min(self.retry_delay * (2 ** (attempt - 1)), 60.0)
                    logger.warning(
                        "[accuracy] rate-limited (429), backing off %.1fs before retry %d",
                        delay, attempt + 1,
                    )
                elif is_bad_request:
                    delay = 2.0
                else:
                    delay = min(self.retry_delay, 30.0)
                time.sleep(delay)
        logger.warning("[accuracy] 全部 %d 次重试耗尽: %s", self.max_retries, last_err)
        return 3, "", self.max_retries, True

    def _call_anthropic(self, system: str, user_prompt: str) -> str:
        """调用 Anthropic Messages API，返回原始文本。（同 GsbPairwiseJudgeReward）"""
        import requests as _requests

        base_url, _, model = _get_anthropic_config()
        url = f"{base_url.rstrip('/')}/messages"
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_next_gsb_api_key()}",
            "Trace-Id": "accuracy-trajectory-reward",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": max(0.0, min(1.0, self.temperature)),
        }
        resp = _requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        if not resp.ok:
            try:
                err_body: Any = resp.json()
            except Exception:
                err_body = resp.text
            self._write_io_log(
                payload,
                {"__error__": True, "status_code": resp.status_code, "body": err_body},
            )
            if resp.status_code == 400:
                err_str = (
                    json.dumps(err_body, ensure_ascii=False)
                    if isinstance(err_body, dict)
                    else str(err_body)
                )
                if "sensitive" in err_str.lower() or "FAILED_PRECONDITION" in err_str:
                    raise _PermanentRejectError(
                        f"[accuracy] 敏感词拦截，不重试 | chars={len(user_prompt)} | detail={err_str[:200]}"
                    )
        resp.raise_for_status()
        data = resp.json()
        raw = str(data["content"][0]["text"])
        self._write_io_log(payload, data)
        return raw

    def _write_io_log(self, payload: Dict[str, Any], response: Any) -> None:
        if not self._io_log_dir:
            return
        try:
            os.makedirs(self._io_log_dir, exist_ok=True)
            log_path = os.path.join(self._io_log_dir, "accuracy_trajectory_io.jsonl")
            safe_payload = {k: v for k, v in payload.items() if k != "Authorization"}
            entry = json.dumps(
                {"ts": time.time(), "pid": os.getpid(), "request": safe_payload, "response": response},
                ensure_ascii=False,
            )
            with self._io_log_lock:
                with open(log_path, "a", encoding="utf-8") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    try:
                        f.write(entry + "\n")
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[accuracy] io_log write failed: %s", exc)


# ---------------------------------------------------------------------------
# 12) WastedStepPenaltyReward —— 无效步惩罚（独立组件）
# ---------------------------------------------------------------------------


class WastedStepPenaltyReward(RewardComponent):
    """无效步惩罚 reward（独立组件版）。

    将原先内嵌在 AiToCGatedPairwiseReward 中的 efficiency_factor 逻辑
    抽取为独立组件，可单独加权或与其他 reward 组合。

    公式（与 AiToCGatedPairwiseReward 内部完全一致）：
        num_wasted = s.num_wasted_steps  （builder 规则识别，无 LLM 调用）
        penalty    = clip(1 - λ × min(num_wasted, cap), 0, 1)

    num_wasted_steps 涵盖：
        - resolve_products 真空跑（resolved=[], pending=0）
        - search_product_clause / search_product_qa 无检索结果
        - obs_error 类（参数缺失、前置校验等模型可控错误）

    默认参数（与 AiToCGatedPairwiseReward 完全一致）：
        lambda_=0.15，cap=2 → 最坏情况 1 - 0.15×2 = 0.70

    注意：若已使用 AiToCGatedPairwiseReward（内含 efficiency_factor），
    不建议再叠加本组件，避免双重惩罚无效步。

    Args:
        lambda_: 每个无效步的折扣率（默认 0.15）
        cap: 最多惩罚到第几个无效步（默认 2，超出不再累加）
    """

    name = "wasted_step_penalty"

    def __init__(
        self,
        name: Optional[str] = None,
        enabled: bool = True,
        lambda_: float = 0.15,
        cap: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, enabled=enabled, **kwargs)
        self.lambda_ = float(lambda_)
        self.cap = int(cap)

    def compute(self, ctx: RewardContext) -> RewardResult:
        s = ctx.summary
        num_wasted = int(getattr(s, "num_wasted_steps", 0) or 0)
        penalty = max(
            0.0,
            min(1.0, 1.0 - self.lambda_ * min(num_wasted, self.cap)),
        )
        return RewardResult(
            value=penalty,
            info={
                "num_wasted_steps": num_wasted,
                "wasted_step_penalty": penalty,
                "lambda": self.lambda_,
                "cap": self.cap,
            },
        )


# ---------------------------------------------------------------------------
# 13) AiToCCombinedReward —— 门控 × (pairwise + accuracy) × efficiency_factor
# ---------------------------------------------------------------------------


class AiToCCombinedReward(RewardComponent):
    """ai_to_c GRPO 统一 reward：门控 × 5维等权质量分 × 无效步折扣。

    解决 AiToCGatedPairwiseReward + AccuracyTrajectoryJudgeReward 分离时的两个问题：
      1. efficiency_factor 应作用于组合总分，而非只乘在 pairwise 上
      2. accuracy judge 应受同一格式门/完成门约束，门控失败时整体返回 0.0

    流程：
        ① 格式门：任意步 response_text 违规 → reward=0.0，跳过所有 judge
        ② 完成门：is_final=False（use_completion_gate=True 时）→ reward=0.0
        ③ 调用 pairwise judge（正反双跑），得到 4 个维度的 per-dim advantage
        ④ pairwise NaN（无 baseline）→ 整体 NaN → GRPO mask
        ⑤ 调用 accuracy judge，得到二值准确性分
        ⑥ 5 维等权聚合：
               dim_i_norm = (dim_i_advantage + 2) / 4  ∈ [0, 1]
               raw = (intent_norm + completeness_norm + practicality_norm
                      + expression_norm + acc_score) / 5
               total = raw × efficiency_factor
           efficiency_factor = clip(1 - λ·min(num_wasted_steps, cap), 0, 1)

    公式说明：
        intent / completeness / practicality / expression：
            pairwise judge 输出的 per-dim B-A 差值 ∈ {-2,-1,0,1,2}，
            经 (x+2)/4 归一化后各占权重 0.2
        acc_score     ∈ {0, 1}（accuracy 二值评分，API 失败 fallback=acc_default_on_error）
            占权重 0.2，与 4 个 pairwise 维度真正等权
        efficiency_factor ∈ [0.70, 1.0]（λ=0.15，cap=2，最坏不归零）
        total         ∈ [0, 1]（clip 在外层 WeightedReward 处理）

    Args:
        use_completion_gate: 是否开启完成门（默认 True）
        use_step_score: 是否启用步数效率分（默认 False，固定 step_score=1.0）
        target_steps / max_steps: 步数效率参数（use_step_score=True 时生效）
        reverse: pairwise 正反双跑消位置偏差（默认 True）
        no_baseline_fallback: "nan"=缺 baseline 时 NaN→mask；"skip"=0.5 平局
        use_accuracy: 是否启用 accuracy judge（默认 True）。设为 False 时跳过 accuracy
            judge 调用（不消耗 API），总分退化为仅 4 个 pairwise 维度等权平均
            （等价于 acc_skipped=1 的分支），关闭但不删除该 reward 组件。
        judge_max_tokens: pairwise judge 最大输出 token（默认 512）
        acc_max_tokens: accuracy judge 最大输出 token（默认 256）
        acc_default_on_error: accuracy API 失败时的 fallback 分（默认 0.5）
        waste_penalty_lambda / waste_penalty_cap: 无效步折扣参数
        judge_*: 两个 judge 共用的超时/重试参数
    """

    name = "ai_to_c_combined"

    def __init__(
        self,
        name: Optional[str] = None,
        enabled: bool = True,
        # 门控
        use_completion_gate: bool = True,
        use_step_score: bool = False,
        target_steps: int = 3,
        max_steps: int = 7,
        # pairwise 参数
        reverse: bool = True,
        no_baseline_fallback: str = "nan",
        judge_max_tokens: int = 512,
        max_query_chars: int = 8000,
        max_answer_chars: int = 8000,
        # accuracy 参数
        use_accuracy: bool = True,
        acc_max_tokens: int = 256,
        acc_max_traj_chars: int = 10000,
        acc_max_answer_chars: int = 4000,
        acc_default_on_error: float = 0.5,
        # 共用 judge 参数
        judge_temperature: float = 0.0,
        judge_timeout: int = 120,
        judge_max_retries: int = 50,
        judge_retry_delay: float = 3.0,
        judge_io_log_dir: Optional[str] = None,
        # 无效步折扣
        waste_penalty_lambda: float = 0.15,
        waste_penalty_cap: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, enabled=enabled, **kwargs)
        if use_step_score and target_steps >= max_steps:
            raise ValueError(f"target_steps({target_steps}) must be < max_steps({max_steps})")
        self.use_completion_gate = bool(use_completion_gate)
        self.use_step_score = bool(use_step_score)
        self.target_steps = int(target_steps)
        self.max_steps = int(max_steps)
        self.waste_penalty_lambda = float(waste_penalty_lambda)
        self.waste_penalty_cap = int(waste_penalty_cap)
        self.use_accuracy = bool(use_accuracy)

        self._pairwise = GsbPairwiseJudgeReward(
            temperature=judge_temperature,
            max_tokens=judge_max_tokens,
            timeout=judge_timeout,
            max_query_chars=max_query_chars,
            max_answer_chars=max_answer_chars,
            max_retries=judge_max_retries,
            retry_delay=judge_retry_delay,
            reverse=reverse,
            no_baseline_fallback=no_baseline_fallback,
            judge_io_log_dir=judge_io_log_dir,
            enabled=enabled,
        )
        self._accuracy = AccuracyTrajectoryJudgeReward(
            temperature=judge_temperature,
            max_tokens=acc_max_tokens,
            timeout=judge_timeout,
            max_traj_chars=acc_max_traj_chars,
            max_answer_chars=acc_max_answer_chars,
            max_retries=judge_max_retries,
            retry_delay=judge_retry_delay,
            default_on_error=acc_default_on_error,
            judge_io_log_dir=judge_io_log_dir,
            enabled=enabled,
        )

    def compute(self, ctx: RewardContext) -> RewardResult:
        import math
        s = ctx.summary
        n = s.num_action_steps

        # ① 格式门
        format_violation = int(
            any(not _react_format_ok(step.response_text) for step in s.steps)
        )
        if format_violation:
            return RewardResult(
                value=0.0,
                info={
                    "format_violation": 1,
                    "task_completed": 0,
                    "pair_score": 0.0,
                    "acc_score": 0.0,
                    "efficiency_factor": 1.0,
                    "num_action_steps": n,
                    "judge_skipped": 1,
                },
            )

        # ② 完成门
        completed, comp_info = is_valid_task_completed(s)
        if self.use_completion_gate and not completed:
            return RewardResult(
                value=0.0,
                info={
                    "format_violation": 0,
                    "task_completed": 0,
                    "pair_score": 0.0,
                    "acc_score": 0.0,
                    "efficiency_factor": 1.0,
                    "num_action_steps": n,
                    "judge_skipped": 1,
                    **comp_info,
                },
            )

        # ③ 步数效率分（use_step_score=False 时固定 1.0）
        if self.use_step_score:
            if n <= self.target_steps:
                step_score = 1.0
            elif n >= self.max_steps:
                step_score = 0.0
            else:
                step_score = 1.0 - (n - self.target_steps) / (self.max_steps - self.target_steps)
        else:
            step_score = 1.0

        # ④ 无效步折扣（旧版 TrajectorySummary 无该字段时视为 0）
        num_wasted = int(getattr(s, "num_wasted_steps", 0) or 0)
        efficiency_factor = max(
            0.0,
            min(1.0, 1.0 - self.waste_penalty_lambda * min(num_wasted, self.waste_penalty_cap)),
        )

        # ⑤ pairwise judge（正反双跑）
        pair_res = self._pairwise.compute(ctx)
        pair_score = float(pair_res.value)

        # 无 baseline → NaN → GRPO mask（accuracy 也不调用，节省 API）
        if math.isnan(pair_score):
            _nan_info: Dict[str, Any] = {
                "format_violation": 0,
                "task_completed": int(completed),
                "pair_score": float("nan"),
                "acc_score": float("nan"),
                "efficiency_factor": efficiency_factor,
                "num_wasted_steps": num_wasted,
                "num_action_steps": n,
                "judge_masked": 1,
            }
            # 与成功路径一致，统一加 "pair/" 前缀，确保 judge_called / judge_failed 等字段
            # 能被 agent_loop.py 的 reward/ai_to_c_combined/pair/judge_called 等 key 正确读取
            for k, v in (pair_res.info or {}).items():
                _nan_info[f"pair/{k}"] = v
            return RewardResult(value=float("nan"), info=_nan_info)

        # ⑥ accuracy judge（use_accuracy=False 时关闭：不调用 judge，不消耗 API，
        #    退化为 acc_skipped=1 分支，即仅 4 个 pairwise 维度等权平均）
        if self.use_accuracy:
            acc_res = self._accuracy.compute(ctx)
            acc_score = float(acc_res.value)
            acc_skipped = (acc_res.info or {}).get("skipped") == "no_knowledge_tools"
            acc_res_info = acc_res.info or {}
        else:
            acc_score = 0.0
            acc_skipped = True
            acc_res_info = {"skipped": "disabled_by_config"}

        # ⑦ 组合得分：
        #   - accuracy 有效（轨迹含知识工具）：5 维等权，各占 0.2
        #   - accuracy 跳过（轨迹无知识工具）：仅 4 个 pairwise 维度均分，accuracy 不参与
        _DIM_ADV_KEYS = ("dim_intent", "dim_completeness", "dim_practicality", "dim_expression")
        pair_info = pair_res.info or {}
        dim_norms: list = []
        for k in _DIM_ADV_KEYS:
            adv = pair_info.get(k)
            if adv is not None and not math.isnan(float(adv)):
                dim_norms.append((float(adv) + 2.0) / 4.0)
            else:
                dim_norms.append(pair_score)  # 单个dim缺失时退回到已聚合的pair_score
        if acc_skipped:
            raw = sum(dim_norms) / 4.0
        else:
            raw = (sum(dim_norms) + acc_score) / 5.0
        total = raw * step_score * efficiency_factor

        info: Dict[str, Any] = {
            "format_violation": 0,
            "task_completed": int(completed),
            "pair_score": pair_score,
            "acc_score": acc_score,
            "acc_skipped": int(acc_skipped),
            "raw_combined": raw,
            "efficiency_factor": efficiency_factor,
            "num_wasted_steps": num_wasted,
            "num_action_steps": n,
            "judge_masked": 0,
            "judge_skipped": 0,
        }
        if self.use_step_score:
            info["step_score"] = step_score
        for k, v in (pair_res.info or {}).items():
            info[f"pair/{k}"] = v
        for k, v in acc_res_info.items():
            info[f"acc/{k}"] = v
        return RewardResult(value=total, info=info)