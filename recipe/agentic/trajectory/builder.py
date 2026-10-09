"""TrajectoryBuilder: List[TraceRecord] → (TrajectorySummary, prompt_ids, response_ids, response_mask).

算法（ToRL 风格字符级 offset_mapping）：

1. **重建 chat 序列**：
   - prompt 文本 = 第 0 步 messages 用 tokenizer.apply_chat_template 渲染（add_generation_prompt=True）
   - 对每一步 i：
       a. 把 step[i].response_text 当作 assistant 消息追加（mask=1 段）
       b. 若 i+1 存在：把 step[i+1].messages 比 step[i].messages 多出来的尾部消息（通常是
          user/tool 的 observation）用 chat_template 渲染为「observation 文本」（mask=0 段）
       c. 若 i+1 不存在（最后一步）：不追加 observation

2. **字符级 mask → token 级 mask**：
   - response = (a)+(b) 顺序拼接出的完整字符串
   - 用 tokenizer(response, return_offsets_mapping=True) 一次性 tokenize
   - 默认 mask 全 1；对每个 observation 段 (start, end)，按 ToRL 公式：
       mask &= ~(token_starts < end & token_ends > start & token_starts >= start)
   - assistant 段保持 1，observation 段被打 0

3. **EOS / padding**：本 builder 不补 EOS，也不 pad；交给上层 AgentLoop / pad utils。

输出：
    summary: TrajectorySummary（含 prompt_text / response_text）
    prompt_ids:   List[int]
    response_ids: List[int]
    response_mask: List[int]  ← 与 response_ids 同长，1=训练，0=mask
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..envs.base import ActionParser, EnvRunResult
from .summary import StepSummary, TrajectorySummary

# 工具执行观测中的错误标记
# - "错误：" 前缀：react_reactor._execute_tool 返回的各类语义错误
#   （未知工具、缺少参数、参数值不存在等）
# - fallback 文案：react_reactor parse 失败后 yield 的兜底字符串
_OBS_ERROR_MARKERS: tuple = (
    "错误：",
    "当前步骤未能解析模型规划结果",
)

# 工具服务端/执行层自身失败的标记（非模型问题，工具本身 bug 或下游服务异常）
# - "出错：" ：give_link_card 等工具的 except 兜底
# - "请求失败 HTTP" / "请求异常:" ：search_general_kb 等 HTTP/网络异常
# - "检索出错：" ：search_cases 异常
# - '"error":' ：_run_actions 并行执行时 except 包装成 JSON {"error":"..."}
_OBS_TOOL_FAILURE_MARKERS: tuple = (
    "出错：",
    "请求失败 HTTP",
    "请求异常:",
    "检索出错：",
    '"error":',
)


def _obs_has_error(obs_messages: List[Dict[str, str]]) -> bool:
    """检查观测消息列表中是否含有模型可控的工具调用错误标记。"""
    for msg in obs_messages:
        content = msg.get("content", "") or ""
        if any(marker in content for marker in _OBS_ERROR_MARKERS):
            return True
    return False


def _obs_has_tool_failure(obs_messages: List[Dict[str, str]]) -> bool:
    """检查观测消息列表中是否含有工具服务端/执行层自身失败的标记。"""
    for msg in obs_messages:
        content = msg.get("content", "") or ""
        if any(marker in content for marker in _OBS_TOOL_FAILURE_MARKERS):
            return True
    return False


# ---------------------------------------------------------------------------
# 无效步检测（wasted_step）
# ---------------------------------------------------------------------------

# resolve_products 真空跑：
#   resolved_products: []
#   unresolved_names: [非空]
# 说明：实测该训练所用 react_reactor._execute_tool 输出中，"resolved_products: []" 与
# "unresolved_names: [...]" 之间还会插入一行 "pending_groups_count: N"（历史遗留字段，
# 曾被误判为不存在，导致旧正则要求二者紧邻、从未命中过任何真实 observation，见
# grpo_pairwise_20260707_222720 run 的实测：339 条 resolve_products 观测中 203 条
# （59.9%）为真正的解析失败，但旧正则命中数为 0）。现改为允许中间存在
# "pending_groups_count: N" 这一行，兼容新旧两种输出格式。
# 若 resolved_products 为空但紧跟的是 autopicked_products（模糊名已自动选定候选，有用），
# 则不算空跑，故用 (?!autopicked_products) 排除该情况。
_RESOLVE_EMPTY_RE = re.compile(
    r"resolved_products:\s*\[\]\s*\n"
    r"(?:pending_groups_count:\s*\d+\s*\n)?"
    r"(?!autopicked_products)unresolved_names:\s*\[[^\]]+\]",
)

# 各类"检索/子 Agent"工具执行成功但产出零有效信息的标记（覆盖所有 search 类工具，
# 不止 resolve_products）：
# - "（无检索结果）"：search_product_clause / search_product_qa 未检索到条款/QA chunk
# - "（空输出）"：run_general_qa_rag / run_contextual_recommend / run_product_recommend /
#                run_premium_calc / run_policy_query 等子 Agent 生成内容为空时的统一兜底文案
# - "（未找到相关历史对话记录）"：session_search 未命中历史会话
# - "未找到相关历史咨询案例。"：case_search 未命中历史案例
_SEARCH_EMPTY_MARKERS: tuple = (
    "（无检索结果）",
    "（空输出）",
    "（未找到相关历史对话记录）",
    "未找到相关历史咨询案例。",
)


def _obs_is_wasted_step(obs_messages: List[Dict[str, str]]) -> bool:
    """检查观测是否为"无效步"（工具执行成功但产出零有效信息，模型可控）。

    覆盖三类情况：
      1. resolve_products 真空跑：resolved_products:[] + unresolved_names 非空
         （注意排除紧跟 autopicked_products 的情况，那是模糊名已自动选定候选，有用）
      2. 所有 search/检索类工具无结果：search_product_clause、search_product_qa、
         run_general_qa_rag、run_contextual_recommend、run_product_recommend、
         run_premium_calc、run_policy_query、session_search、case_search 等，
         只要命中 _SEARCH_EMPTY_MARKERS 中任一空结果兜底文案即视为无效步
      3. obs_error 类：模型可控的参数/前置校验错误（已被 _obs_has_error 覆盖，
         此处统一归入无效步以便 reward 折扣使用）

    不含 tool_failure（服务端失败，非模型问题，不应惩罚）。
    """
    full_text = "\n".join(msg.get("content", "") or "" for msg in obs_messages)

    # 1. resolve_products 真空跑
    if _RESOLVE_EMPTY_RE.search(full_text):
        return True

    # 2. 所有 search 类工具无结果
    if any(marker in full_text for marker in _SEARCH_EMPTY_MARKERS):
        return True

    # 3. obs_error 类（模型可控错误：参数缺失、前置校验等）
    if any(marker in full_text for marker in _OBS_ERROR_MARKERS):
        return True

    return False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _apply_chat_template(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
    apply_chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """tokenizer.apply_chat_template 的薄封装，统一参数。"""
    kwargs = dict(apply_chat_template_kwargs or {})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )


def _get_forced_gen_prefix(
    tokenizer: Any,
    apply_chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """计算 chat template 在 add_generation_prompt=True 时强制注入但模型不会生成的前缀文本。

    典型场景：Qwen3 + enable_thinking=False，generation prompt 末尾为：
        <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n
    其中 ``<think>\\n\\n</think>\\n\\n`` 是模板强制注入部分，vLLM 将其视为
    prompt token（输入），不会出现在 response_text 里。

    对普通模型（无 forced prefix），返回 ``""``。
    """
    try:
        kwargs = dict(apply_chat_template_kwargs or {})
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}],
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
        marker = "<|im_start|>assistant\n"
        idx = rendered.rfind(marker)
        if idx >= 0:
            return rendered[idx + len(marker):]
    except Exception:
        pass
    return ""


def _extract_obs_segment(
    prefix_cur: str,
    response_text: str,
    prefix_next: str,
    tokenizer: Any,
    cur_messages: List[Dict[str, str]],
    next_messages: List[Dict[str, str]],
    apply_chat_template_kwargs: Optional[Dict[str, Any]] = None,
    forced_gen_prefix: str = "",
) -> str:
    """计算两步之间的 observation 文本段（含 chat-template 过渡 token）。

    传统做法：对 obs_messages 单独 apply_chat_template，会缺失 stop token（如
    ``<|im_end|>``）以及下一轮的 start token（如 ``<|im_start|>assistant\\n``），
    导致 response_text_full 与实际对话 token 序列不一致。

    新做法：利用相邻步骤的完整 prefix 做字符级 diff：
        between = prefix_next[len(prefix_cur):]
        # = response_text + <|im_end|>\\n + obs + <|im_start|>assistant\\n
        obs_segment = between[len(response_text):]
        # = <|im_end|>\\n + obs + <|im_start|>assistant\\n  ← 全部 mask=0

    特殊处理：Qwen3 + enable_thinking=False 时，generation prompt 末尾含
    ``<think>\\n\\n</think>\\n\\n``，vLLM 将其视为 prompt token（不在 response_text
    里）。此时 prefix_cur 末尾含该前缀，而 prefix_next 对应位置直接是 response_text
    内容，导致 startswith 失败。修正：去掉 forced_gen_prefix 后重新 diff，
    obs_segment 仍会包含下一步的 ``<think>\\n\\n</think>\\n\\n``（mask=0，不训练）。

    若 prefix 关系不满足（如业务侧压缩了历史），回退到旧的单独渲染方式并记录警告。
    """
    if prefix_next.startswith(prefix_cur):
        between = prefix_next[len(prefix_cur):]
        if between.startswith(response_text):
            return between[len(response_text):]
        # 极少数情况：chat template 对 response 内容做了轻微格式化（如尾部去空白），
        # 尝试 strip 后匹配
        rs = response_text.rstrip()
        if between.startswith(rs):
            return between[len(rs):]
        logger.warning(
            "[TrajectoryBuilder] response_text 与 between 起点不匹配，"
            "回退到单独渲染。response_text[:60]=%r, between[:60]=%r",
            response_text[:60],
            between[:60],
        )
    elif forced_gen_prefix and prefix_cur.endswith(forced_gen_prefix):
        # Qwen3 enable_thinking=False：去掉 generation prompt 末尾的 forced prefix 后重试
        prefix_cur_base = prefix_cur[: -len(forced_gen_prefix)]
        if prefix_next.startswith(prefix_cur_base):
            between = prefix_next[len(prefix_cur_base):]
            if between.startswith(response_text):
                return between[len(response_text):]
            rs = response_text.rstrip()
            if between.startswith(rs):
                return between[len(rs):]
            # 业务层（如 react_reactor）可能在记录 TraceRecord 之后对 raw response 做了
            # 后处理（如 re.sub 去除 <redacted_thinking> 块）再存入 messages，导致
            # between（来自 messages）与 response_text（来自 TraceRecord 原始响应）不一致。
            # 兜底：找 between 中第一个 <|im_end|> 位置作为 obs_segment 起点——
            # assistant 轮结束标记总是在 response 内容之后，response 本身不含该 token。
            im_end_pos = between.find("<|im_end|>")
            if im_end_pos >= 0:
                logger.info(
                    "[TrajectoryBuilder] response_text 已被业务层后处理（如去除 <redacted_thinking>），"
                    "通过 <|im_end|> 定位 obs_segment 起点。response_text[:60]=%r, between[:60]=%r",
                    response_text[:60],
                    between[:60],
                )
                return between[im_end_pos:]
            logger.warning(
                "[TrajectoryBuilder] forced_gen_prefix 修正后 response_text 与 between 起点仍不匹配，"
                "回退到单独渲染。response_text[:60]=%r, between[:60]=%r",
                response_text[:60],
                between[:60],
            )
        else:
            logger.warning(
                "[TrajectoryBuilder] prefix_next 不以 prefix_cur_base 为前缀，"
                "回退到单独渲染。prefix_cur_base[-30:]=%r, prefix_next[:30]=%r",
                prefix_cur_base[-30:],
                prefix_next[:30],
            )
    else:
        logger.warning(
            "[TrajectoryBuilder] prefix_next 不以 prefix_cur 为前缀（历史被压缩？），"
            "回退到单独渲染。prefix_cur[-30:]=%r, prefix_next[:30]=%r",
            prefix_cur[-30:],
            prefix_next[:30],
        )

    # 回退：旧方式——单独渲染 obs 消息（缺少 stop/start token 但总比崩溃好）
    obs_msgs = _diff_observation_messages(cur_messages, next_messages)
    if obs_msgs:
        return _apply_chat_template(
            tokenizer,
            obs_msgs,
            add_generation_prompt=False,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
        )
    return ""


def _diff_observation_messages(
    cur_messages: List[Dict[str, str]],
    next_messages: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """next_messages 通常 = cur_messages + assistant(response) + [observation...]。

    我们要从 next_messages 切出 ``observation...`` 部分（去掉 cur_messages 长度 + 1
    个 assistant）。若长度异常，返回 next_messages[len(cur_messages):] 兜底，再过滤掉
    role=assistant 的消息。
    """
    n = len(cur_messages)
    tail = next_messages[n:]
    # 去掉紧跟着的 assistant 消息（它就是当前 step 的 response，已经单独处理）
    while tail and tail[0].get("role") == "assistant":
        tail = tail[1:]
    return tail


def _record_to_messages(record: Any) -> List[Dict[str, str]]:
    """TraceRecord.messages 适配（dataclass / dict 都兼容）。"""
    msgs = getattr(record, "messages", None)
    if msgs is None and isinstance(record, dict):
        msgs = record.get("messages", [])
    return [dict(m) for m in (msgs or [])]


def _record_to_response(record: Any) -> str:
    resp = getattr(record, "response", None)
    if resp is None and isinstance(record, dict):
        resp = record.get("response", "")
    return str(resp or "")


def _record_meta(record: Any) -> Tuple[Optional[str], float, float]:
    model = getattr(record, "model", None)
    ts_start = getattr(record, "ts_start", 0.0)
    ts_end = getattr(record, "ts_end", 0.0)
    if isinstance(record, dict):
        model = model or record.get("model")
        ts_start = ts_start or record.get("ts_start", 0.0)
        ts_end = ts_end or record.get("ts_end", 0.0)
    return model, float(ts_start or 0.0), float(ts_end or 0.0)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BuilderOutput:
    summary: TrajectorySummary
    prompt_ids: List[int]
    response_ids: List[int]
    response_mask: List[int]


class TrajectoryBuilder:
    """业务无关的 trajectory builder。

    Args:
        tokenizer: HuggingFace AutoTokenizer（必须支持 apply_chat_template + offset_mapping）
        action_parser: 业务自带 ActionParser
        observation_marker_role: 当某些业务不提供 next_messages 时的兜底（暂未使用）
    """

    def __init__(
        self,
        tokenizer: Any,
        action_parser: ActionParser,
        apply_chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.action_parser = action_parser
        # Qwen3/Qwen3.6 默认 thinking；训练侧需与 vLLM rollout 对齐
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {
            "enable_thinking": False,
        }
        # Qwen3 enable_thinking=False 时 generation prompt 末尾的强制注入文本
        # （如 "<think>\n\n</think>\n\n"），vLLM 视其为 prompt token，不在 response 里。
        # 普通模型此值为 ""。
        self._forced_gen_prefix: str = _get_forced_gen_prefix(
            tokenizer, self.apply_chat_template_kwargs
        )

    # ------------------------------------------------------------------

    def build(
        self,
        env_result: EnvRunResult,
        query: str = "",
        teacher_trace: Optional[str] = None,
    ) -> BuilderOutput:
        records: Sequence[Any] = env_result.records or []

        # ---------------- 1. 解析每一步 ----------------
        steps: List[StepSummary] = []
        observation_segments: List[Tuple[str, str]] = []  # (assistant_text, obs_segment)

        # prefix_cur：当前步骤的完整对话前缀（以 add_generation_prompt=True 渲染），
        # 初始化为 None，第一次迭代时计算并缓存，后续迭代复用上一步的 prefix_next。
        # 这样每步只需额外调用一次 apply_chat_template（计算 prefix_next），
        # 避免重复渲染相同的历史前缀。
        prefix_cur: Optional[str] = None

        for i, rec in enumerate(records):
            messages = _record_to_messages(rec)
            response_text = _record_to_response(rec)
            model, ts_s, ts_e = _record_meta(rec)

            parsed = self.action_parser.parse(response_text)

            steps.append(
                StepSummary(
                    step_idx=i,
                    messages=messages,
                    response_text=response_text,
                    parsed_action=parsed.parsed_action,
                    parse_ok=parsed.parse_ok,
                    is_final=parsed.is_final,
                    action_name=(parsed.parsed_action or {}).get("name", "") if parsed.parsed_action else "",
                    model=model,
                    ts_start=ts_s,
                    ts_end=ts_e,
                )
            )

            # 懒计算 prefix_cur（第一步才需要，之后复用 prefix_next）
            if prefix_cur is None:
                prefix_cur = _apply_chat_template(
                    self.tokenizer,
                    messages,
                    add_generation_prompt=True,
                    apply_chat_template_kwargs=self.apply_chat_template_kwargs,
                )

            # observation 段：用相邻步骤的完整 prefix diff 计算，
            # 可正确包含 stop token（<|im_end|>）和下一轮 start token（<|im_start|>assistant\n），
            # 这些 token 在训练时均应被 mask（mask=0）。
            obs_segment = ""
            if i + 1 < len(records):
                next_messages = _record_to_messages(records[i + 1])
                prefix_next = _apply_chat_template(
                    self.tokenizer,
                    next_messages,
                    add_generation_prompt=True,
                    apply_chat_template_kwargs=self.apply_chat_template_kwargs,
                )
                obs_segment = _extract_obs_segment(
                    prefix_cur,
                    response_text,
                    prefix_next,
                    self.tokenizer,
                    messages,
                    next_messages,
                    apply_chat_template_kwargs=self.apply_chat_template_kwargs,
                    forced_gen_prefix=self._forced_gen_prefix,
                )
                prefix_cur = prefix_next  # 缓存，下一步直接复用

                # 检测工具执行观测是否含错误（利用已 diff 出的 obs_messages）
                obs_msgs = _diff_observation_messages(messages, next_messages)
                if _obs_has_error(obs_msgs):
                    steps[-1].obs_error = True
                if _obs_has_tool_failure(obs_msgs):
                    steps[-1].tool_failure = True
                if _obs_is_wasted_step(obs_msgs):
                    steps[-1].wasted_step = True

            observation_segments.append((response_text, obs_segment))

        # ---------------- 2. 构造 prompt 与 response 文本 ----------------
        if records:
            # 第 0 步的 messages 作为「prompt」（不含 assistant），渲染时 add_generation_prompt=True
            # prefix_cur 在 i=0 时已经算过，直接复用（但只有 1 步时 prefix_cur 仍是 step0 的值）
            prompt_text = _apply_chat_template(
                self.tokenizer,
                steps[0].messages,
                add_generation_prompt=True,
                apply_chat_template_kwargs=self.apply_chat_template_kwargs,
            )
        else:
            # 没有任何主脑 LLM call → 用空 prompt 兜底
            prompt_text = ""

        # response = 按 step 顺序拼接 (assistant_text + observation_text)
        # 同时记录每个 observation 段的字符级 [start, end)
        response_parts: List[str] = []
        observation_char_ranges: List[Tuple[int, int]] = []
        cursor = 0
        for assistant_text, obs_text in observation_segments:
            response_parts.append(assistant_text)
            cursor += len(assistant_text)
            if obs_text:
                obs_start = cursor
                response_parts.append(obs_text)
                cursor += len(obs_text)
                observation_char_ranges.append((obs_start, cursor))
        response_text_full = "".join(response_parts)

        # ---------------- 3. tokenize 并打 mask ----------------
        prompt_ids = self._encode(prompt_text)
        response_ids, response_mask = self._encode_with_mask(
            response_text_full, observation_char_ranges
        )

        # ---------------- 4. 构造 summary ----------------
        summary = TrajectorySummary(
            query=query or env_result.env_info.get("query", ""),
            steps=steps,
            final_answer=env_result.final_answer,
            env_info=dict(env_result.env_info),
            prompt_text=prompt_text,
            response_text=response_text_full,
            teacher_trace=teacher_trace,
        )

        return BuilderOutput(
            summary=summary,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
        )

    # ------------------------------------------------------------------
    # tokenize helpers
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> List[int]:
        if not text:
            return []
        enc = self.tokenizer(text, add_special_tokens=False)
        return list(enc["input_ids"])

    def _encode_with_mask(
        self,
        text: str,
        observation_char_ranges: Sequence[Tuple[int, int]],
    ) -> Tuple[List[int], List[int]]:
        """ToRL 风格字符级 offset_mapping → token 级 mask。

        默认全 1；落在某个 observation 段内的 token 打 0。
        """
        if not text:
            return [], []

        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        token_ids = list(enc["input_ids"])
        offsets = list(enc["offset_mapping"])
        mask = [1] * len(token_ids)

        if not observation_char_ranges:
            return token_ids, mask

        # 对每个 obs (start, end)，遵循 ToRL 算法：
        #   mask=0 当且仅当 (token_start < end) AND (token_end > start) AND (token_start >= start)
        for ob_start, ob_end in observation_char_ranges:
            for idx, (t_s, t_e) in enumerate(offsets):
                if (t_s < ob_end) and (t_e > ob_start) and (t_s >= ob_start):
                    mask[idx] = 0

        return token_ids, mask