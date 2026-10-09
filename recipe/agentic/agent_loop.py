"""AgenticAgentLoop: 业务无关的 verl AgentLoop 实现。

把 BusinessEnv + TrajectoryBuilder + WeightedReward 包成 verl 标准 AgentLoop。

发布副本对所有任务保留上游奖励，链均值为 total/K。
此协议不是原 read-only 实验的等价复现。

执行流（async）：
    1. env.setup_once(vllm_base_url, vllm_model_name)（首次幂等）
    2. env_result = await asyncio.to_thread(env.run, query=..., ext_params=...)
       —— 业务同步生成器在线程池跑，event loop 不阻塞；ContextVar 跨线程拷贝
    3. summary, prompt_ids, resp_ids, mask = builder.build(env_result)
    4. reward_value, sub_info = reward.compute(RewardContext(summary=summary, ...))
    5. 返回 AgentLoopOutput（含 prompt_ids/response_ids/response_mask/reward_score/extra_fields）

业务 LLM 调用不走 self.server_manager；而是通过 agentic_train_offline.llm_bridge 在主脑请求里
直接 HTTP 打 vllm_base_url（verl 内置 vllm/sglang 都是 OpenAI 兼容服务器）。

dataset 行字段约定（必须，由 data/prepare_dataset.py 产出）：
    - "raw_prompt": List[{"role","content"}]  ← 至少含一条 user 消息，content 就是 query
                                                 也兼容 messages[-1]["content"] 当 query
    - "ext_params": dict (可选)
    - "extra_info": dict (可选；含 teacher_trace / ground_truth 等)
    - "ground_truth": Any (可选)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.utils.profiler import simple_timer

from .envs.base import BusinessEnv
from .reward.base import RewardCombiner, RewardContext
from .reward.weighted import WeightedReward, build_weighted_reward_from_config
from .trajectory.builder import TrajectoryBuilder

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("agentic")
class AgenticAgentLoop(AgentLoopBase):
    """Agentic RL 业务无关的 AgentLoop。

    config 入口（在 rollout.agent.custom_async_server 同级或 trainer config 顶层）::

        agentic:
          env:
            _target_: recipe.agentic.envs.AiToCReactEnv
            ai_to_c_repo_path: /workspace/ai2c
            session_dir_override: /tmp/rl_sessions
          reward:
            components:
              - _target_: recipe.agentic.reward.components.TaskCompletedReward
                weight: 1.0
              - _target_: recipe.agentic.reward.components.ToolCallSuccessReward
                weight: 0.3
            normalize: true
          vllm_base_url: "http://127.0.0.1:8000/v1"     # 训练 actor vLLM
          vllm_model_name: "qwen-32b-rl"
          per_rollout_timeout_sec: 600
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        agentic_cfg = self._resolve_agentic_config()

        # env 实例化（hydra-style _target_ + kwargs）
        self.env: BusinessEnv = _instantiate_target(agentic_cfg["env"])
        # reward 装配
        _reward_cfg = _to_plain(agentic_cfg.get("reward") or {})
        if "components" in _reward_cfg:
            self.reward: RewardCombiner = build_weighted_reward_from_config(_reward_cfg)
        elif "_target_" in _reward_cfg:
            self.reward = _instantiate_target(_reward_cfg)
        else:
            # 只有 normalize/clip 等全局设置，没有 components → 空 combiner（reward 恒为 0）
            self.reward = WeightedReward(
                components=[],
                weights=[],
                normalize=bool(_reward_cfg.get("normalize", True)),
                clip_min=_reward_cfg.get("clip_min"),
                clip_max=_reward_cfg.get("clip_max"),
            )

        self.vllm_model_name: Optional[str] = agentic_cfg.get("vllm_model_name")
        self.per_rollout_timeout_sec: float = float(
            agentic_cfg.get("per_rollout_timeout_sec", 600)
        )
        # Qwen3/Qwen3.6 默认 thinking；训练侧 tokenizer 渲染与 vLLM rollout 需一致
        self.apply_chat_template_kwargs: Dict[str, Any] = dict(
            agentic_cfg.get("apply_chat_template_kwargs") or {"enable_thinking": False}
        )

        # verl 内部 vLLM 服务器使用 get_free_port() 分配随机端口；config 中填写的
        # 固定端口（如 8000）往往不正确。
        # 注意：__init__ 在 async Ray actor 的事件循环线程中被同步调用，此时调用
        # ray.get() 会阻塞事件循环，可能导致结果异常（空列表 / 超时）。
        # 因此这里只存储 cfg_url，真正的探测延迟到 run()（async context）中进行。
        self.vllm_base_url: Optional[str] = agentic_cfg.get("vllm_base_url")

        # 同步 vLLM chat_template_kwargs 到 env（主脑 HTTP 推理走 llm_bridge）
        if hasattr(self.env, "_vllm_chat_template_kwargs"):
            self.env._vllm_chat_template_kwargs = self.apply_chat_template_kwargs

        # Builder 用本 worker 的 tokenizer（与 verl 训练侧一致）
        self.builder = TrajectoryBuilder(
            tokenizer=self.tokenizer,
            action_parser=self.env.action_parser,
            apply_chat_template_kwargs=self.apply_chat_template_kwargs,
        )

        # verl rollout 长度限制（用于截断 response_ids）
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        # 进程级 setup（幂等）；URL 可能尚未探测到实际端口，run() 开头会再次修正。
        try:
            self.env.setup_once(self.vllm_base_url, self.vllm_model_name)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[AgenticAgentLoop] env.setup_once failed: %s", exc)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    async def _async_detect_vllm_url(self) -> Optional[str]:
        """在 async 上下文里通过 GlobalRequestLoadBalancer 探测实际 vLLM 地址。

        verl 的 vLLM 服务器使用 get_free_port() 分配随机端口；此方法 await Ray
        远程调用，不阻塞事件循环，避免 __init__ 中同步 ray.get() 的问题。

        返回探测到的 base_url（含 /v1），失败时返回 self.vllm_base_url（配置值）。
        """
        try:
            lb = getattr(getattr(self, "server_manager", None), "_load_balancer", None)
            if lb is None:
                logger.warning(
                    "[AgenticAgentLoop] server_manager 无 _load_balancer，"
                    "使用 cfg vllm_url=%s",
                    self.vllm_base_url,
                )
                return self.vllm_base_url
            server_ids: list = await lb.get_all_servers.remote()
            if not server_ids:
                logger.warning(
                    "[AgenticAgentLoop] get_all_servers() 返回空列表，"
                    "使用 cfg vllm_url=%s",
                    self.vllm_base_url,
                )
                return self.vllm_base_url
            addr = server_ids[0]  # e.g. "127.0.0.1:52345" or "[::1]:52345"
            if not addr.startswith("http"):
                addr = f"http://{addr}"
            url = f"{addr}/v1"
            if url != self.vllm_base_url:
                logger.warning(
                    "[AgenticAgentLoop] 实际 vLLM URL=%s（cfg=%s）",
                    url,
                    self.vllm_base_url,
                )
            return url
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[AgenticAgentLoop] async vLLM URL 探测失败（%s），使用 cfg=%s",
                exc,
                self.vllm_base_url,
            )
            return self.vllm_base_url

    async def _async_detect_vllm_model_name(self, base_url: str) -> Optional[str]:
        """查询 /v1/models 拿到 vLLM 实际注册的 model name。

        verl 以 `vllm serve <local_path>` 启动，vLLM 默认将 local_path 作为
        model id（而非 config 中填写的别名如 "gemma-4-31b-rl"）。
        此方法在 async 线程里发出同步 HTTP 请求，避免阻塞事件循环。
        """
        import asyncio as _asyncio

        def _fetch():
            try:
                import requests as _req
                resp = _req.get(f"{base_url}/models", timeout=5)
                resp.raise_for_status()
                data = resp.json()
                models = data.get("data") or []
                if models:
                    return models[0].get("id")
            except Exception as exc:  # noqa: BLE001
                logger.warning("[AgenticAgentLoop] /v1/models 查询失败（%s）", exc)
            return None

        model_name = await _asyncio.to_thread(_fetch)
        if model_name and model_name != self.vllm_model_name:
            logger.warning(
                "[AgenticAgentLoop] 实际 vLLM model_name=%r（cfg=%r）",
                model_name,
                self.vllm_model_name,
            )
        return model_name

    # ------------------------------------------------------------------
    # config 解析
    # ------------------------------------------------------------------

    def _resolve_agentic_config(self) -> Dict[str, Any]:
        """从 trainer config 取 agentic 子配置。

        约定挂载路径（按优先级）：
            1. self.config.agentic（推荐）
            2. self.config.actor_rollout_ref.rollout.agent.agentic
            3. self.config.recipe.agentic
        """
        cfg = self.config
        for path in (
            ("agentic",),
            ("actor_rollout_ref", "rollout", "agent", "agentic"),
            ("recipe", "agentic"),
        ):
            cur = cfg
            ok = True
            for p in path:
                cur = cur.get(p) if hasattr(cur, "get") else getattr(cur, p, None)
                if cur is None:
                    ok = False
                    break
            if ok and cur is not None:
                return _to_plain(cur)
        raise ValueError(
            "AgenticAgentLoop: config 中找不到 `agentic` 节（试过 agentic / "
            "actor_rollout_ref.rollout.agent.agentic / recipe.agentic）"
        )

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    async def run(self, sampling_params: Dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        # 0) 在 async 上下文里探测实际 vLLM URL + model name，并同步到 env。
        #    使用 await 而非 __init__ 中的同步 ray.get()，避免阻塞 Ray 事件循环。
        #    verl 以 `vllm serve <model_path>` 启动，实际 model name = model_path，
        #    而非 config 中填写的别名；因此需查 /v1/models 自动修正。
        actual_url = await self._async_detect_vllm_url()
        if actual_url and actual_url != self.env._vllm_base_url:
            self.env._vllm_base_url = actual_url
        actual_model = await self._async_detect_vllm_model_name(actual_url or self.vllm_base_url)
        # 只有当 vLLM 返回的是非路径别名时才覆盖；若返回本地路径（以 "/" 开头），
        # 说明 vLLM 未设置 --served-model-name，保留 config 中填写的别名，
        # 避免 LLM bridge 用错误的 model id 发起请求。
        if actual_model and not actual_model.startswith("/"):
            self.env._vllm_model_name = actual_model
        elif self.vllm_model_name and self.vllm_model_name != self.env._vllm_model_name:
            self.env._vllm_model_name = self.vllm_model_name

        # 1) 取 query / ext_params / extra_info
        import json as _json
        raw_prompt = kwargs.get("raw_prompt") or kwargs.get("messages") or []
        query = _extract_query(raw_prompt) or str(kwargs.get("query", ""))

        # ext_params 在 parquet 中以 JSON 字符串存储，需反序列化为 dict
        _ext_raw = kwargs.get("ext_params") or {}
        if isinstance(_ext_raw, str):
            try:
                ext_params: Dict[str, Any] = _json.loads(_ext_raw) if _ext_raw.strip() else {}
            except (ValueError, TypeError):
                ext_params = {}
        else:
            ext_params = dict(_to_plain(_ext_raw) or {})

        extra_info = _to_plain(kwargs.get("extra_info") or {})
        ground_truth = kwargs.get("ground_truth", extra_info.get("ground_truth"))
        teacher_trace = extra_info.get("teacher_trace") if isinstance(extra_info, dict) else None
        # 从 extra_info 提取历史会话，与 trace_collect 保持一致
        history: List[Dict[str, Any]] = list(extra_info.get("history") or []) if isinstance(extra_info, dict) else []

        metrics: Dict[str, float] = {}

        # 2) env.run（同步业务 chat_stream → 线程池）
        with simple_timer("agentic_env_run", metrics):
            try:
                env_result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.env.run,
                        query=query,
                        ext_params=ext_params,
                        history=history,
                    ),
                    timeout=self.per_rollout_timeout_sec,
                )
            except asyncio.TimeoutError:
                logger.warning("[AgenticAgentLoop] env.run timeout (>%.0fs)", self.per_rollout_timeout_sec)
                from .envs.base import EnvRunResult
                env_result = EnvRunResult(
                    records=[], final_answer="",
                    env_info={"query": query, "status": "timeout", "error": "wall_clock_timeout"},
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("[AgenticAgentLoop] env.run failed: %s", exc)
                from .envs.base import EnvRunResult
                env_result = EnvRunResult(
                    records=[], final_answer="",
                    env_info={"query": query, "status": "error", "error": f"{type(exc).__name__}: {exc}"},
                )

        # 3) TrajectoryBuilder（CPU 计算，丢线程池）
        with simple_timer("agentic_builder", metrics):
            builder_out = await asyncio.to_thread(
                self.builder.build, env_result, query, teacher_trace
            )

        # 4) Reward（同步组件可能含 HTTP judge → to_thread）
        with simple_timer("agentic_reward", metrics):
            reward_res = await asyncio.to_thread(
                self.reward.compute,
                RewardContext(summary=builder_out.summary, ground_truth=ground_truth, extra_info=extra_info),
            )

        # 5) 截断（verl 长度上限）
        # prompt 保留开头（system prompt 必须在前），不能取尾部 [-n:]
        prompt_ids = builder_out.prompt_ids[: self.prompt_length] if self.prompt_length else builder_out.prompt_ids
        response_ids = builder_out.response_ids[: self.response_length] if self.response_length else builder_out.response_ids
        response_mask = builder_out.response_mask[: self.response_length] if self.response_length else builder_out.response_mask

        # env.run 失败时 prompt_ids/response_ids 为空列表，tokenizer.pad 对空列表
        # 会返回 Python list 而非 Tensor，导致后续 .dim() 崩溃。
        # 用一个 pad token 占位（reward=0 的样本不参与有效梯度更新）。
        _pad_id: int = getattr(self.tokenizer, "pad_token_id", None) or 0
        if not prompt_ids:
            prompt_ids = [_pad_id]
        if not response_ids:
            response_ids = [_pad_id]
            response_mask = [0]

        # 6) 组装 AgentLoopOutput
        agentic_metrics_payload: Dict[str, Any] = {
            "agentic/env_status": env_result.env_info.get("status", "unknown"),
            "agentic/env_elapsed_sec": env_result.env_info.get("elapsed_sec", 0.0),
            "agentic/num_total_calls": env_result.env_info.get("num_total_calls", 0),
            "agentic/num_mainbrain_calls": env_result.env_info.get("num_mainbrain_calls", 0),
            "agentic/num_steps": builder_out.summary.num_steps,
            "agentic/num_action_steps": builder_out.summary.num_action_steps,
            "agentic/num_parse_ok": builder_out.summary.num_parse_ok,
            "agentic/num_obs_errors": _summary_metric(builder_out.summary, "num_obs_errors"),
            "agentic/has_final_step": int(builder_out.summary.has_final_step),
            "agentic/has_error": int(builder_out.summary.has_error),
            "agentic/tool_call_success_rate": builder_out.summary.tool_call_success_rate(),
            "agentic/obs_error_rate": _summary_metric(builder_out.summary, "obs_error_rate"),
            "agentic/num_tool_failures": _summary_metric(builder_out.summary, "num_tool_failures"),
            "agentic/tool_failure_rate": _summary_metric(builder_out.summary, "tool_failure_rate"),
            "agentic/num_wasted_steps": _summary_metric(builder_out.summary, "num_wasted_steps"),
            "agentic/wasted_step_rate": _summary_metric(builder_out.summary, "wasted_step_rate"),
            "agentic/final_answer_len": len(env_result.final_answer or ""),
            "agentic/response_token_len": len(response_ids),
            "agentic/prompt_token_len": len(prompt_ids),
        }
        # 把 reward 子项也带出去
        agentic_metrics_payload.update(reward_res.info or {})

        loop_metrics = AgentLoopMetrics(
            generate_sequences=metrics.get("agentic_env_run", 0.0),
            tool_calls=float(builder_out.summary.num_action_steps),
            compute_score=metrics.get("agentic_reward", 0.0),
        )

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            reward_score=float(reward_res.value),
            num_turns=builder_out.summary.num_steps,
            metrics=loop_metrics,
            extra_fields={
                "agentic_metrics": agentic_metrics_payload,
                "agentic_env_info": dict(env_result.env_info),
                "agentic_final_answer": env_result.final_answer,
                "agentic_summary": builder_out.summary.to_dict(),
                "reward_extra_info": _build_agentic_reward_extra_info(
                    reward_res.info or {},
                    builder_out,
                    env_result,
                    response_ids,
                ),
            },
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _reward_info_float(reward_info: dict, *keys: str, default: float = 0.0) -> float:
    """从 WeightedReward info 按优先级取第一个有效 float（跳过 NaN）。"""
    import math

    for k in keys:
        if k not in reward_info:
            continue
        v = reward_info[k]
        if v is None:
            continue
        try:
            fv = float(v)
            if math.isnan(fv):
                continue
            return fv
        except (TypeError, ValueError):
            continue
    return default


def _build_agentic_reward_extra_info(
    reward_info: Dict[str, Any],
    builder_out: Any,
    env_result: Any,
    response_ids: List[int],
) -> Dict[str, float]:
    """对齐生产 pairwise 脚本的 SwanLab train/* 字段。"""
    return {
        # ── 质量门控信号 ──────────────────────────────────────────────
        "task_completed": _reward_info_float(
            reward_info,
            "reward/ai_to_c_combined/task_completed",
            "reward/ai_to_c_gated_pairwise/task_completed",
            "reward/ai_to_c_gated_judge/task_completed",
            "reward/ai_to_c_efficient/task_completed",
            "reward/task_completed",
            "reward/appworld_efficient/appworld_success",
            "reward/appworld_verifier/appworld_success",
        ),
        "format_violation": _reward_info_float(
            reward_info,
            "reward/ai_to_c_combined/format_violation",
            "reward/ai_to_c_gated_pairwise/format_violation",
            "reward/ai_to_c_gated_judge/format_violation",
        ),
        # ── pairwise 分 ────────────────────────────────────────────────
        "pair_score": _reward_info_float(
            reward_info,
            "reward/ai_to_c_combined/pair_score",
            "reward/ai_to_c_gated_pairwise/pair_score",
            "reward/ai_to_c_gated_judge/gsb_score",
        ),
        "gsb_score": _reward_info_float(
            reward_info,
            "reward/ai_to_c_combined/pair_score",
            "reward/ai_to_c_gated_pairwise/pair_score",
            "reward/ai_to_c_gated_judge/gsb_score",
        ),
        "model_advantage": _reward_info_float(
            reward_info,
            "reward/ai_to_c_combined/pair/model_advantage",
            "reward/ai_to_c_gated_pairwise/judge_model_advantage",
        ),
        "score_fwd": _reward_info_float(
            reward_info,
            "reward/ai_to_c_combined/pair/score_fwd",
            "reward/ai_to_c_gated_pairwise/judge_score_fwd",
        ),
        "score_rev_raw": _reward_info_float(
            reward_info,
            "reward/ai_to_c_combined/pair/score_rev_raw",
            "reward/ai_to_c_gated_pairwise/judge_score_rev_raw",
        ),
        "dim_intent": _reward_info_float(
            reward_info,
            "reward/ai_to_c_combined/pair/dim_intent",
            "reward/ai_to_c_gated_pairwise/judge_dim_intent",
        ),
        "dim_accuracy": _reward_info_float(
            reward_info,
            "reward/ai_to_c_combined/pair/dim_accuracy",
            "reward/ai_to_c_gated_pairwise/judge_dim_accuracy",
        ),
        "dim_completeness": _reward_info_float(
            reward_info,
            "reward/ai_to_c_combined/pair/dim_completeness",
            "reward/ai_to_c_gated_pairwise/judge_dim_completeness",
        ),
        "dim_practicality": _reward_info_float(
            reward_info,
            "reward/ai_to_c_combined/pair/dim_practicality",
            "reward/ai_to_c_gated_pairwise/judge_dim_practicality",
        ),
        "dim_expression": _reward_info_float(
            reward_info,
            "reward/ai_to_c_combined/pair/dim_expression",
            "reward/ai_to_c_gated_pairwise/judge_dim_expression",
        ),
        # ── accuracy 分 ────────────────────────────────────────────────
        "acc_score": _reward_info_float(
            reward_info,
            "reward/ai_to_c_combined/acc_score",
        ),
        "acc_skipped": _reward_info_float(
            reward_info,
            "reward/ai_to_c_combined/acc_skipped",
        ),
        # ── 组合得分 ────────────────────────────────────────────────────
        "raw_combined": _reward_info_float(
            reward_info,
            "reward/ai_to_c_combined/raw_combined",
        ),
        "judge_called": _reward_info_float(
            reward_info,
            "reward/ai_to_c_combined/pair/judge_called",
            "reward/ai_to_c_gated_pairwise/judge_called",
            "reward/ai_to_c_gated_judge/judge_called",
        ),
        "judge_retries_exhausted": _reward_info_float(
            reward_info,
            "reward/ai_to_c_combined/pair/judge_failed",
            "reward/ai_to_c_gated_pairwise/judge_judge_failed",
            "reward/ai_to_c_gated_pairwise/judge_masked",
            "reward/ai_to_c_gated_judge/retries_exhausted",
        ),
        "judge_retry_attempts": _reward_info_float(
            reward_info,
            "reward/ai_to_c_combined/pair/fwd_attempts",
            "reward/ai_to_c_gated_pairwise/judge_fwd_attempts",
            "reward/ai_to_c_gated_judge/retry_attempts",
        ),
        "step_score": _reward_info_float(
            reward_info,
            "reward/ai_to_c_combined/step_score",
            "reward/ai_to_c_gated_pairwise/step_score",
            "reward/ai_to_c_gated_judge/step_score",
            "reward/ai_to_c_efficient/step_score",
            "reward/appworld_efficient/step_score",
            default=1.0,
        ),
        "efficiency_factor": _reward_info_float(
            reward_info,
            "reward/ai_to_c_combined/efficiency_factor",
            "reward/ai_to_c_gated_pairwise/efficiency_factor",
            "reward/appworld_efficient/step_score",
        ),
        # ── 轨迹长度 ────────────────────────────────────────────────────
        "num_action_steps": float(builder_out.summary.num_action_steps),
        "response_token_len": float(len(response_ids)),
        "final_answer_char_len": float(len((env_result.final_answer or "").strip())),
        "total_model_output_char_len": float(
            sum(len(s.response_text) for s in builder_out.summary.steps)
        ),
        # ── 工具调用质量 ────────────────────────────────────────────────
        "num_obs_errors": _summary_metric(builder_out.summary, "num_obs_errors"),
        "obs_error_rate": _summary_metric(builder_out.summary, "obs_error_rate"),
        "num_tool_failures": _summary_metric(builder_out.summary, "num_tool_failures"),
        "tool_failure_rate": _summary_metric(builder_out.summary, "tool_failure_rate"),
        # ── 无效步（模型可控）──────────────────────────────────────────
        "num_wasted_steps": _summary_metric(builder_out.summary, "num_wasted_steps"),
        "wasted_step_rate": _summary_metric(builder_out.summary, "wasted_step_rate"),
        # ── 失败 / 截断类型 ─────────────────────────────────────────────
        "is_timeout": float(env_result.env_info.get("status") == "timeout"),
        "is_env_error": float(env_result.env_info.get("status") == "error"),
        "is_no_final_step": float(not builder_out.summary.has_final_step),
    }


def _extract_task_reward_extra(task: Dict[str, Any]) -> Dict[str, Any]:
    """从 task_chain 单任务提取 pairwise reward 需要的 extra_info（含 baseline_answer）。"""
    extra: Dict[str, Any] = {}
    nested = task.get("extra_info") if isinstance(task.get("extra_info"), dict) else {}
    ba = task.get("baseline_answer")
    if ba is None:
        ba = nested.get("baseline_answer")
    if ba is not None and str(ba).strip():
        extra["baseline_answer"] = str(ba).strip()
    for k in ("ground_truth", "teacher_trace"):
        v = task.get(k, nested.get(k) if nested else None)
        if v is not None:
            extra[k] = v
    return extra


def _nanmean(values: List[float]) -> float:
    import math

    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _summary_metric(summary: Any, name: str, default: float = 0.0) -> float:
    """兼容旧版 TrajectorySummary（无 obs_error / wasted_step 字段时回退 default）。"""
    attr = getattr(summary, name, None)
    if attr is None:
        return float(default)
    try:
        return float(attr() if callable(attr) else attr)
    except Exception:  # noqa: BLE001
        return float(default)


def _build_stharc_reward_extra_info(
    reward_infos: List[Dict[str, Any]],
    builder_outs: List[Any],
    env_results: List[Any],
    response_ids: List[int],
    *,
    total_reward: float,
    prospective_bonus: float,
    chain_reward: float,
    chain_length: int,
    num_mutations: int,
    avg_task_reward: float,
    avg_frozen_reward: float,
    avg_delta_reward: float,
    truncation: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """链级聚合：对齐生产 pairwise 字段 + STHARC 专有标量，供 train/* SwanLab。"""
    # 用首个非空 reward_info 做 key 探测；均值聚合各任务
    def _mean_key(*keys: str, default: float = 0.0) -> float:
        vals = [_reward_info_float(info, *keys, default=float("nan")) for info in reward_infos]
        # 若全部 fallback 到 nan（key 不存在），再走 default
        import math

        finite = [v for v in vals if not math.isnan(v)]
        if not finite:
            # 区分「缺 key」与「真 NaN」：若任意 info 含这些 key 且值为 nan，返回 nan
            for info in reward_infos:
                for k in keys:
                    if k in info:
                        try:
                            if math.isnan(float(info[k])):
                                return float("nan")
                        except (TypeError, ValueError):
                            pass
            return default
        return sum(finite) / len(finite)

    extra = {
        "task_completed": _mean_key(
            "reward/ai_to_c_combined/task_completed",
            "reward/ai_to_c_gated_pairwise/task_completed",
            "reward/ai_to_c_efficient/task_completed",
            "reward/task_completed",
            "reward/appworld_efficient/appworld_success",
            "reward/appworld_verifier/appworld_success",
        ),
        "format_violation": _mean_key(
            "reward/ai_to_c_combined/format_violation",
            "reward/ai_to_c_gated_pairwise/format_violation",
        ),
        "pair_score": _mean_key(
            "reward/ai_to_c_combined/pair_score",
            "reward/ai_to_c_gated_pairwise/pair_score",
        ),
        "gsb_score": _mean_key(
            "reward/ai_to_c_combined/pair_score",
            "reward/ai_to_c_gated_pairwise/pair_score",
        ),
        "model_advantage": _mean_key(
            "reward/ai_to_c_combined/pair/model_advantage",
            "reward/ai_to_c_gated_pairwise/judge_model_advantage",
        ),
        "score_fwd": _mean_key(
            "reward/ai_to_c_combined/pair/score_fwd",
            "reward/ai_to_c_gated_pairwise/judge_score_fwd",
        ),
        "score_rev_raw": _mean_key(
            "reward/ai_to_c_combined/pair/score_rev_raw",
            "reward/ai_to_c_gated_pairwise/judge_score_rev_raw",
        ),
        "dim_intent": _mean_key(
            "reward/ai_to_c_combined/pair/dim_intent",
            "reward/ai_to_c_gated_pairwise/judge_dim_intent",
        ),
        "dim_accuracy": _mean_key(
            "reward/ai_to_c_combined/pair/dim_accuracy",
            "reward/ai_to_c_gated_pairwise/judge_dim_accuracy",
        ),
        "dim_completeness": _mean_key(
            "reward/ai_to_c_combined/pair/dim_completeness",
            "reward/ai_to_c_gated_pairwise/judge_dim_completeness",
        ),
        "dim_practicality": _mean_key(
            "reward/ai_to_c_combined/pair/dim_practicality",
            "reward/ai_to_c_gated_pairwise/judge_dim_practicality",
        ),
        "dim_expression": _mean_key(
            "reward/ai_to_c_combined/pair/dim_expression",
            "reward/ai_to_c_gated_pairwise/judge_dim_expression",
        ),
        "acc_score": _mean_key("reward/ai_to_c_combined/acc_score"),
        "acc_skipped": _mean_key("reward/ai_to_c_combined/acc_skipped"),
        "raw_combined": _mean_key("reward/ai_to_c_combined/raw_combined"),
        # ── judge 调用质量（链上均值；便于排查 API / 重试）────────────────
        "judge_called": _mean_key(
            "reward/ai_to_c_combined/pair/judge_called",
            "reward/ai_to_c_gated_pairwise/judge_called",
            "reward/ai_to_c_gated_judge/judge_called",
        ),
        "judge_retries_exhausted": _mean_key(
            "reward/ai_to_c_combined/pair/judge_failed",
            "reward/ai_to_c_gated_pairwise/judge_judge_failed",
            "reward/ai_to_c_gated_pairwise/judge_masked",
            "reward/ai_to_c_gated_judge/retries_exhausted",
        ),
        "judge_retry_attempts": _mean_key(
            "reward/ai_to_c_combined/pair/fwd_attempts",
            "reward/ai_to_c_gated_pairwise/judge_fwd_attempts",
            "reward/ai_to_c_gated_judge/retry_attempts",
        ),
        "step_score": _mean_key(
            "reward/ai_to_c_combined/step_score",
            "reward/ai_to_c_efficient/step_score",
            "reward/appworld_efficient/step_score",
            default=1.0,
        ),
        "efficiency_factor": _mean_key(
            "reward/ai_to_c_combined/efficiency_factor",
            "reward/ai_to_c_gated_pairwise/efficiency_factor",
            "reward/appworld_efficient/step_score",
            default=1.0,
        ),
        "num_action_steps": float(sum(b.summary.num_action_steps for b in builder_outs)),
        "response_token_len": float(len(response_ids)),
        "final_answer_char_len": float(
            sum(len((e.final_answer or "").strip()) for e in env_results) / max(len(env_results), 1)
        ),
        "total_model_output_char_len": float(
            sum(len(s.response_text) for b in builder_outs for s in b.summary.steps)
        ),
        "num_obs_errors": float(
            sum(_summary_metric(b.summary, "num_obs_errors") for b in builder_outs)
        ),
        "obs_error_rate": _nanmean(
            [_summary_metric(b.summary, "obs_error_rate") for b in builder_outs]
        ),
        "num_tool_failures": float(
            sum(_summary_metric(b.summary, "num_tool_failures") for b in builder_outs)
        ),
        "tool_failure_rate": _nanmean(
            [_summary_metric(b.summary, "tool_failure_rate") for b in builder_outs]
        ),
        "num_wasted_steps": float(
            sum(_summary_metric(b.summary, "num_wasted_steps") for b in builder_outs)
        ),
        "wasted_step_rate": _nanmean(
            [_summary_metric(b.summary, "wasted_step_rate") for b in builder_outs]
        ),
        "is_timeout": float(any(e.env_info.get("status") == "timeout" for e in env_results)),
        "is_env_error": float(any(e.env_info.get("status") == "error" for e in env_results)),
        "is_no_final_step": float(any(not b.summary.has_final_step for b in builder_outs)),
        # ── STHARC 专有 ────────────────────────────────────────────────
        "total_reward": float(total_reward),
        "prospective_bonus": float(prospective_bonus),
        "chain_reward": float(chain_reward),
        "chain_length": float(chain_length),
        "num_mutations": float(num_mutations),
        "avg_task_reward": float(avg_task_reward),
        "avg_frozen_reward": float(avg_frozen_reward),
        "avg_delta_reward": float(avg_delta_reward),
    }
    if truncation:
        for k, v in truncation.items():
            try:
                extra[f"trunc/{k}"] = float(v)
            except (TypeError, ValueError):
                continue
    return extra


def _normalize_stharc_credit(raw: Any) -> str:
    """credit=pu（默认 STHARC PU）| sage | skillrise。"""
    val = str(raw or "pu").strip().lower()
    if val in ("", "pu", "stharc", "prospective", "lambda"):
        return "pu"
    if val in ("sage", "sage_style", "skill_integrated"):
        return "sage"
    if val in ("skillrise", "skill_rise"):
        return "skillrise"
    raise ValueError(f"stharc.credit 必须是 pu | sage | skillrise（got {raw!r}）")


def _to_plain(obj: Any) -> Any:
    """OmegaConf DictConfig / ListConfig → 原生 dict/list，便于 isinstance 判断与 ** 解包。"""
    try:
        from omegaconf import OmegaConf
        if hasattr(obj, "_content"):  # DictConfig / ListConfig
            return OmegaConf.to_container(obj, resolve=True)
    except Exception:  # noqa: BLE001
        pass
    return obj


def _instantiate_target(cfg: Any) -> Any:
    """轻量的 `_target_` 实例化（不依赖 hydra.utils.instantiate，避免递归 resolve 副作用）。"""
    import importlib

    cfg = _to_plain(cfg)
    if not isinstance(cfg, dict) or "_target_" not in cfg:
        raise ValueError(f"_instantiate_target: cfg 必须含 _target_，got: {cfg!r}")
    cfg = dict(cfg)  # shallow copy
    target = cfg.pop("_target_")
    module_path, _, cls_name = target.rpartition(".")
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    return cls(**cfg)


def _extract_query(raw_prompt: Any) -> str:
    """从 raw_prompt（list of messages 或 str）里取 query。"""
    if isinstance(raw_prompt, str):
        return raw_prompt
    if not raw_prompt:
        return ""
    try:
        # 取最后一条 user 消息
        for m in reversed(list(raw_prompt)):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content") or "")
        # 退化：取最后一条 content
        last = list(raw_prompt)[-1]
        if isinstance(last, dict):
            return str(last.get("content") or "")
    except Exception:  # noqa: BLE001
        pass
    return str(raw_prompt)


# ═══════════════════════════════════════════════════════════════════════════════
# STHARCAgentLoop
# ═══════════════════════════════════════════════════════════════════════════════

@register("stharc")
class STHARCAgentLoop(AgentLoopBase):
    """STHARC-GRPO agent loop：单条 task chain 的完整执行（chain g 层 + task k 层）。

    外层 group 循环由 verl trainer 通过 rollout.n=G 并行多个 STHARCAgentLoop 实例实现。

    输入（来自 parquet 的单条样本）：
        task_chain  List[dict]   ← K 个任务，每个含 raw_prompt 和 ext_params（JSON 字符串）
        chain_g     int          ← 本条 chain 的编号（0 ~ G-1，由 uid 中的重复行推断）
        group_id    str          ← GRPO group 标识（通常就是 chain_id）

    输出：
        AgentLoopOutput（chain_reward 作为 reward_score，所有 K 任务 token 拼接）

    YAML 配置示例::

        stharc:
          env:
            _target_: recipe.agentic.envs.AiToCReactEnv
            ai_to_c_repo_path: /workspace/ai2c
          reward:
            components:
              - _target_: recipe.agentic.reward.components.TaskCompletedReward
                weight: 1.0
          vllm_base_url: "http://127.0.0.1:8000/v1"
          vllm_model_name: "qwen-32b-rl"
          per_rollout_timeout_sec: 600
          # STHARC 专有参数
          gamma:            0.9
          lam:              0.25
          credit:           "pu"            # pu（STHARC）| sage | skillrise
          attribution:      "latest_only"   # gamma_decay | latest_only | equal_split | random_shuffle
          shuffle_seed:     0               # 仅 random_shuffle 用
              run_frozen_chain: false           # optional diagnostic only; never subtracted from credit
          base_user_id:     "default"
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        agentic_cfg = self._resolve_stharc_config()

        self.env: Any = _instantiate_target(agentic_cfg["env"])
        _reward_cfg = _to_plain(agentic_cfg.get("reward") or {})
        if "components" in _reward_cfg:
            from .reward.weighted import build_weighted_reward_from_config
            self.reward = build_weighted_reward_from_config(_reward_cfg)
        elif "_target_" in _reward_cfg:
            self.reward = _instantiate_target(_reward_cfg)
        else:
            from .reward.weighted import WeightedReward
            self.reward = WeightedReward(
                components=[],
                weights=[],
                normalize=bool(_reward_cfg.get("normalize", True)),
                clip_min=_reward_cfg.get("clip_min"),
                clip_max=_reward_cfg.get("clip_max"),
            )

        self.vllm_model_name: Optional[str] = agentic_cfg.get("vllm_model_name")
        self.vllm_base_url: Optional[str] = agentic_cfg.get("vllm_base_url")
        self.per_rollout_timeout_sec: float = float(
            agentic_cfg.get("per_rollout_timeout_sec", 600)
        )
        self.apply_chat_template_kwargs: Dict[str, Any] = dict(
            agentic_cfg.get("apply_chat_template_kwargs") or {"enable_thinking": False}
        )

        # STHARC 专有参数
        self.gamma: float = float(agentic_cfg.get("gamma", 0.9))
        self.lam: float = float(agentic_cfg.get("lam", 0.25))
        self.credit: str = _normalize_stharc_credit(agentic_cfg.get("credit", "pu"))
        self.attribution: str = str(agentic_cfg.get("attribution", "latest_only"))
        # Random-PU 负对照用；仅 attribution=random_shuffle 时生效
        self.shuffle_seed: int = int(agentic_cfg.get("shuffle_seed", 0))
        self.run_frozen: bool = bool(agentic_cfg.get("run_frozen_chain", False))
        self.base_user_id: str = str(agentic_cfg.get("base_user_id", "default"))
        # 单任务截断上限（拼链预算）；截断会发生，但会记入 trunc/* 供检测
        _mtr = agentic_cfg.get("max_task_response_length")
        self.max_task_response_length: Optional[int] = (
            int(_mtr) if _mtr is not None else None
        )
        _mtc = agentic_cfg.get("max_task_context_length")
        self.max_task_context_length: Optional[int] = (
            int(_mtc) if _mtc is not None else None
        )
        # 仅检测/打点，默认不因截断中断训练
        self.abort_on_packing_truncation: bool = bool(
            agentic_cfg.get("abort_on_packing_truncation", False)
        )
        self.abort_on_exec_truncation: bool = bool(
            agentic_cfg.get("abort_on_exec_truncation", False)
        )
        # All task rewards are preserved from the upstream reward pipeline.

        # 脏链（截断 / 无 final / env 失败）不上 PU：chain_reward=total_reward，避免奖「又长又脏」
        _prc = agentic_cfg.get("pu_require_clean", False)
        if isinstance(_prc, str):
            self.pu_require_clean = _prc.strip().lower() in ("1", "true", "yes", "on")
        else:
            self.pu_require_clean = bool(_prc)

        # trace 写入目录（从 trainer.rollout_data_dir 派生，写到 traces/ 子目录）
        _trainer_cfg = _to_plain(getattr(self.config, "trainer", None) or {})
        _rollout_dir = (
            _trainer_cfg.get("rollout_data_dir") if isinstance(_trainer_cfg, dict) else None
        )
        self.trace_dir: Optional[str] = (
            str(_rollout_dir) + "/traces" if _rollout_dir else None
        )

        # 同步 vLLM chat_template_kwargs 到 env
        if hasattr(self.env, "_vllm_chat_template_kwargs"):
            self.env._vllm_chat_template_kwargs = self.apply_chat_template_kwargs

        self.builder = TrajectoryBuilder(
            tokenizer=self.tokenizer,
            action_parser=self.env.action_parser,
            apply_chat_template_kwargs=self.apply_chat_template_kwargs,
        )

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        # 未显式配置时：插入的 pk 预算对齐 prompt_length
        if self.max_task_context_length is None and self.prompt_length:
            self.max_task_context_length = int(self.prompt_length)

        try:
            self.env.setup_once(self.vllm_base_url, self.vllm_model_name)
        except Exception as exc:
            logger.exception("[STHARCAgentLoop] env.setup_once failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # config 解析
    # ------------------------------------------------------------------

    def _resolve_stharc_config(self) -> Dict[str, Any]:
        """从 trainer config 取 stharc 子配置（或 agentic 作为兜底）。"""
        cfg = self.config
        for path in (
            ("stharc",),
            ("agentic",),
            ("actor_rollout_ref", "rollout", "agent", "stharc"),
            ("actor_rollout_ref", "rollout", "agent", "agentic"),
            ("recipe", "stharc"),
            ("recipe", "agentic"),
        ):
            cur = cfg
            ok = True
            for p in path:
                cur = cur.get(p) if hasattr(cur, "get") else getattr(cur, p, None)
                if cur is None:
                    ok = False
                    break
            if ok and cur is not None:
                return _to_plain(cur)
        raise ValueError(
            "STHARCAgentLoop: config 中找不到 `stharc` 或 `agentic` 节"
        )

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    async def run(self, sampling_params: Dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        import json as _json
        import importlib

        from .stharc.core import (
            rollout_with_tracking,
            execute_harness_action,
            mark_provenance,
        )
        from .stharc.paper_credit import compute_task_credit, task_outcome, pack_task_credit
        # HarnessState 实现模块可配置（stharc.harness_state_module），默认沿用
        # ai2c 耦合的文件系统实现；GenericBenchmarkEnv 等解耦 env 可通过配置换成
        # ``recipe.agentic.stharc.generic_harness_state``（进程内 dict 沙箱，无
        # ai2c/文件系统依赖），两者暴露完全一致的 HarnessState API
        # （create_h_init / clone_for_chain / to_ext_params_patch /
        #  provenance_registry / snapshot_for_log / cleanup）。
        # 绝对导入 + 半初始化兜底（循环导入时模块可能已在 sys.modules 但缺属性）
        _hs_module_path = str(
            self._resolve_stharc_config().get(
                "harness_state_module", "recipe.agentic.stharc.harness_state"
            )
        )
        _hs_mod = importlib.import_module(_hs_module_path)
        HarnessState = getattr(_hs_mod, "HarnessState", None)
        if HarnessState is None:
            _hs_mod = importlib.reload(_hs_mod)
            HarnessState = getattr(_hs_mod, "HarnessState", None)
        if HarnessState is None:
            raise ImportError(
                "cannot import HarnessState from "
                f"{getattr(_hs_mod, '__file__', _hs_mod)!r}; "
                f"exports={ [n for n in dir(_hs_mod) if not n.startswith('_')] }"
            )
        from .reward.base import RewardContext

        # 0) 探测 vLLM URL
        # 默认会通过 GlobalRequestLoadBalancer 探测 verl 自己在 GPU 上起的内部
        # vLLM rollout 引擎地址，并覆盖 stharc.vllm_base_url/vllm_model_name
        # （ai2c 生产场景下两者是同一个引擎，这样做是对的）。但如果用户是自己
        # 手动起了一个独立的外部 vLLM OpenAI server（stharc.disable_vllm_autodetect:
        # true），则完全跳过探测，强制使用配置里写的 vllm_base_url/vllm_model_name，
        # 避免被自动探测到的内部引擎地址（可能不支持 /v1/chat/completions，
        # 或压根不是想要连的那个服务）悄悄覆盖掉。
        if bool(self._resolve_stharc_config().get("disable_vllm_autodetect", False)):
            self.env._vllm_base_url = self.vllm_base_url
            self.env._vllm_model_name = self.vllm_model_name
        else:
            actual_url = await self._async_detect_vllm_url()
            if actual_url and actual_url != getattr(self.env, "_vllm_base_url", None):
                self.env._vllm_base_url = actual_url
            actual_model = await self._async_detect_vllm_model_name(actual_url or self.vllm_base_url)
            # 注意：不能用 `not actual_model.startswith("/")` 过滤掉探测结果——
            # verl 自己起的内部 vLLM 引擎默认没配 prometheus.served_model_name
            # 别名时，`/v1/models` 合法返回的 model id 就是本地绝对路径（例如
            # actor_rollout_ref.model.path），这恰恰是唯一能用的真实 model 名；
            # 之前这里会把它当"不像名字"丢弃，转而用配置里的 vllm_model_name
            # （给外部手动起的 server 用的别名，内部引擎根本没注册这个名字），
            # 导致请求打过去报 "The model `xxx` does not exist."。
            if actual_model:
                self.env._vllm_model_name = actual_model
            elif self.vllm_model_name and self.vllm_model_name != getattr(self.env, "_vllm_model_name", None):
                self.env._vllm_model_name = self.vllm_model_name

        # 1) 解析 task_chain 和 chain 元信息
        _raw_task_chain = kwargs.get("task_chain") or "[]"
        if isinstance(_raw_task_chain, str):
            try:
                task_chain: List[Dict[str, Any]] = _json.loads(_raw_task_chain)
            except (ValueError, TypeError):
                task_chain = []
        else:
            task_chain = list(_raw_task_chain)

        group_id = str(kwargs.get("group_id") or kwargs.get("chain_id") or kwargs.get("uid") or "group_0")
        chain_g = int(kwargs.get("chain_g") or 0)

        if not task_chain:
            raise ValueError("STHARC requires a non-empty task_chain")

        # 2) 创建 H_init 和本 chain 的独立沙箱 H_g
        # 用 uuid 后缀确保同一 chain 的 G 个并行 worker 各自独立（rollout.n=G 时
        # 多个 worker 共享同一 chain_id/chain_g，若不加后缀会发生沙箱路径冲突）
        H_init = HarnessState.create_h_init(
            group_id=group_id,
            base_user_id=self.base_user_id,
        )
        _run_uid = uuid4().hex[:8]
        H_g = H_init.clone_for_chain(f"chain_{chain_g}_{_run_uid}")
        _h_init_cleaned = False  # 追踪 H_init 是否已在 try 块内提前清理

        total_reward = 0.0
        harness_mutations: List = []
        task_read_sets: List = []
        task_rewards_list: List[float] = []
        all_builder_outs: List = []
        all_env_results: List = []                    # 每个任务的原始 env_result（含 records）
        all_harness_actions_per_task: List[List] = [] # 每个任务的 harness 写操作列表
        all_reward_infos: List[Dict[str, Any]] = []   # 每任务 WeightedReward.info（SwanLab）
        # 预先初始化，防止 try 块中途抛异常导致 finally 后的引用出现 NameError
        frozen_rewards: List[float] = [0.0] * len(task_chain)
        prospective_bonus: float = 0.0
        chain_reward: float = 0.0
        sage_gen_bonus: float = 0.0
        sage_use_bonus: float = 0.0
        skillrise_curate_return: float = 0.0
        skillrise_docs: List[str] = []
        concat_builder_outs: List = []
        sandbox_snapshot: Optional[Dict[str, Any]] = None

        try:
            # 3) 链内 K 个任务依次执行
            _logged_first_env_error = False
            _skill_doc = ""
            for k, task in enumerate(task_chain):
                try:
                    env_result, harness_actions_k, read_set_k = await asyncio.wait_for(
                        asyncio.to_thread(
                            rollout_with_tracking, self.env, task, H_g, False
                        ),
                        timeout=self.per_rollout_timeout_sec,
                    )
                except asyncio.TimeoutError:
                    logger.warning("[STHARCAgentLoop] task %d timeout", k)
                    from .envs.base import EnvRunResult
                    env_result = EnvRunResult(
                        records=[], final_answer="",
                        env_info={"status": "timeout", "error": "wall_clock_timeout"},
                    )
                    harness_actions_k = []
                    read_set_k = set()
                except Exception as exc:
                    logger.exception("[STHARCAgentLoop] task %d failed: %s", k, exc)
                    from .envs.base import EnvRunResult
                    env_result = EnvRunResult(
                        records=[], final_answer="",
                        env_info={"status": "error", "error": f"{type(exc).__name__}: {exc}"},
                    )
                    harness_actions_k = []
                    read_set_k = set()

                # 构建 trajectory
                query = _extract_query(task.get("raw_prompt") or task.get("messages") or task.get("query", ""))
                builder_out = await asyncio.to_thread(
                    self.builder.build, env_result, query, None
                )

                # 计算本任务 reward（pairwise 需要 task 上的 baseline_answer）
                task_extra = _extract_task_reward_extra(task if isinstance(task, dict) else {})
                reward_res = await asyncio.to_thread(
                    self.reward.compute,
                    RewardContext(
                        summary=builder_out.summary,
                        ground_truth=task_extra.get("ground_truth"),
                        extra_info=task_extra,
                    ),
                )
                r_k = task_outcome(task, reward_res.value)

                total_reward += r_k
                task_read_sets.append(read_set_k)
                task_rewards_list.append(r_k)
                all_builder_outs.append(builder_out)
                concat_builder_outs.append(builder_out)
                all_env_results.append(env_result)
                all_harness_actions_per_task.append(harness_actions_k)
                all_reward_infos.append(dict(reward_res.info or {}))

                # 首个 env error 打 ERROR，避免整链静默全挂后只看到 is_env_error=1.0
                _estatus = (getattr(env_result, "env_info", None) or {}).get("status")
                if not _logged_first_env_error and _estatus == "error":
                    _logged_first_env_error = True
                    _einfo = dict(getattr(env_result, "env_info", None) or {})
                    _tep = task.get("ext_params") if isinstance(task, dict) else {}
                    if isinstance(_tep, str):
                        try:
                            _tep = _json.loads(_tep) if _tep.strip() else {}
                        except (ValueError, TypeError):
                            _tep = {}
                    logger.error(
                        "[STHARCAgentLoop] first env error in chain group_id=%s k=%d "
                        "appworld_task_id=%r appworld_root=%r vllm_url=%r vllm_model=%r "
                        "error=%r",
                        group_id,
                        k,
                        (_tep or {}).get("appworld_task_id") if isinstance(_tep, dict) else None,
                        getattr(self.env, "appworld_root", None),
                        getattr(self.env, "_vllm_base_url", None),
                        getattr(self.env, "_vllm_model_name", None),
                        _einfo.get("error"),
                    )

                # 记录 harness mutation provenance
                for action in harness_actions_k:
                    output = execute_harness_action(action, H_g)
                    harness_mutations.append((k, action, output))
                    mark_provenance(H_g, output, source=(k, action))

            # 4) frozen chain baseline（阶段一：b_j=0，阶段二：run_frozen=True）
            # 同一 group 的 G 条 chain 共享一份 frozen_rewards：
            #   第一个拿到文件锁的 worker 计算并写盘，其余 worker 等锁后直接读缓存。
            #   模式与 HarnessState.create_h_init 的文件锁幂等初始化完全一致。
            # SAGE / SkillRise 不算 PU、不跑 frozen。
            if self.credit in ("sage", "skillrise"):
                frozen_rewards = [0.0] * len(task_chain)
            elif self.run_frozen:
                # global_steps：verl AgentLoopWorker 注入，用于 frozen 缓存按 step 失效
                _global_step = kwargs.get("global_steps")
                try:
                    global_step = int(_global_step) if _global_step is not None else -1
                except (TypeError, ValueError):
                    global_step = -1
                frozen_rewards = await asyncio.to_thread(
                    _get_or_compute_frozen_rewards,
                    self.env,
                    task_chain,
                    H_init,
                    group_id,
                    self.trace_dir,
                    self.reward,
                    self.builder,
                    global_step,

                )
            else:
                frozen_rewards = [0.0] * len(task_chain)

            # H_init 在 frozen rewards 计算完成后即可释放。
            # 不放到 finally 块：避免其他 worker 仍在 run_frozen_chain → clone_for_chain
            # 中执行 copytree 时 H_init 目录被并发删除（竞态）。
            H_init.cleanup()
            _h_init_cleaned = True

            # Paper credit is task-local; frozen outcomes are diagnostics only.
            task_returns, task_bonuses = compute_task_credit(
                harness_mutations, task_read_sets, task_rewards_list,
                credit=self.credit, gamma=self.gamma, lam=self.lam,
                attribution=self.attribution, shuffle_seed=self.shuffle_seed,
            )
            prospective_bonus = sum(task_bonuses) if self.credit == "pu" else 0.0
            sage_gen_bonus = sum(task_bonuses) if self.credit == "sage" else 0.0
            skillrise_curate_return = sum(task_bonuses) if self.credit == "skillrise" else 0.0
            chain_reward = sum(task_returns)

        finally:
            # cleanup 前落盘沙箱快照（skills 清单 + 变更 skill 全文 + 最终 memory）
            try:
                mutated_skill_ids = [
                    str(a.get("target_id") or "")
                    for actions in all_harness_actions_per_task
                    for a in actions
                    if a.get("op_type") == "skill_manage"
                ]
                sandbox_snapshot = H_g.snapshot_for_log(
                    mutated_skill_ids=mutated_skill_ids,
                )
            except Exception as exc:
                logger.warning(
                    "[STHARCAgentLoop] sandbox snapshot failed group=%s chain_g=%s: %s",
                    group_id, chain_g, exc,
                )
                sandbox_snapshot = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "group_id": group_id,
                    "chain_tag": getattr(H_g, "chain_tag", None),
                    "skills_root": str(getattr(H_g, "skills_root", "")),
                    "memory_user_id": getattr(H_g, "memory_user_id", None),
                }
            H_g.cleanup()
            if not _h_init_cleaned:
                H_init.cleanup()

        # 6) 拼接 K 个任务 + 截断检测（只打点，默认不中断）
        K = len(task_chain)
        _max_task_resp = self.max_task_response_length
        _max_task_ctx = self.max_task_context_length
        if _max_task_resp is None and self.response_length and K > 0:
            _ctx_budget = (K - 1) * int(_max_task_ctx or 0) if K > 1 else 0
            _max_task_resp = max(1, (int(self.response_length) - _ctx_budget) // K)
        chain_trajectory = _concat_builder_outputs(
            concat_builder_outs or all_builder_outs,
            max_task_response_length=_max_task_resp,
            max_task_context_length=_max_task_ctx,
        )
        concat_trunc = dict(getattr(chain_trajectory, "truncation_stats", None) or {})
        per_task_trunc = list(getattr(chain_trajectory, "per_task_truncation", None) or [])

        _pad_id = getattr(self.tokenizer, "pad_token_id", None) or 0
        _response_limit = self.response_length if self.response_length else None
        _prompt_full = _to_int_list(chain_trajectory.prompt_ids)
        _resp_full = _to_int_list(chain_trajectory.response_ids)
        _mask_full = _to_int_list(chain_trajectory.response_mask)
        _task_ids_full = list(chain_trajectory.response_task_ids)
        if not len(_resp_full) == len(_mask_full) == len(_task_ids_full):
            raise ValueError("STHARC response, mask and task IDs must have equal lengths")
        pack_trunc = _measure_chain_pack_truncation(
            _prompt_full,
            _resp_full,
            _mask_full,
            prompt_limit=self.prompt_length,
            response_limit=_response_limit,
        )
        prompt_ids = (
            _prompt_full[: int(self.prompt_length)] if self.prompt_length else _prompt_full
        )
        response_ids = (
            _resp_full[: int(_response_limit)] if _response_limit else _resp_full
        )
        response_mask = (
            _mask_full[: int(_response_limit)] if _response_limit else _mask_full
        )

        response_task_ids = (
            _task_ids_full[: int(_response_limit)] if _response_limit else _task_ids_full
        )
        if not prompt_ids:
            prompt_ids = [_pad_id]
        if not response_ids:
            response_ids = [_pad_id]
            response_mask = [0]
            response_task_ids = [-1]

        env_trunc = _aggregate_env_truncation_stats(all_env_results)
        trunc_report = _finalize_truncation_report(concat_trunc, pack_trunc, env_trunc)
        # 任务级明细进 trace（SwanLab 只吃标量）
        truncated_idxs = [t["task_idx"] for t in per_task_trunc if t.get("any_trunc")]
        for i, er in enumerate(all_env_results):
            info = dict(getattr(er, "env_info", None) or {})
            t = info.get("truncation") if isinstance(info.get("truncation"), dict) else {}
            hit = bool(
                t.get("obs_char_trunc_count")
                or t.get("context_overflow")
                or t.get("max_steps_exceeded")
                or info.get("status") in ("context_overflow", "max_steps_exceeded")
            )
            if hit and i not in truncated_idxs:
                truncated_idxs.append(i)
        trunc_report_full = {
            **trunc_report,
            "per_task": per_task_trunc,
            "truncated_task_idxs": truncated_idxs,
        }
        if trunc_report.get("any_trunc_hit", 0) > 0:
            logger.warning(
                "[STHARCAgentLoop] truncation DETECTED group=%s g=%s "
                "truncated_task_idxs=%s packing=%s exec=%s",
                group_id,
                chain_g,
                trunc_report_full.get("truncated_task_idxs"),
                trunc_report.get("packing_trunc_hit"),
                trunc_report.get("exec_trunc_hit"),
            )

        # 5b) 干净链才吃 PU（须在 trunc 统计之后；主报 avg_task 不变）
        import math as _math_pu
        pu_clean_gated = 0.0
        prospective_bonus_raw = float(prospective_bonus)
        if (
            self.credit == "pu"
            and self.pu_require_clean
            and not _math_pu.isnan(float(chain_reward))
            and not _math_pu.isnan(float(total_reward))
        ):
            _dirty = float(trunc_report.get("any_trunc_hit", 0) or 0) > 0
            if not _dirty:
                _dirty = any(
                    not bool(getattr(getattr(bo, "summary", None), "has_final_step", True))
                    for bo in all_builder_outs
                )
            if not _dirty:
                for _er in all_env_results:
                    _st = (getattr(_er, "env_info", None) or {}).get("status")
                    if _st in ("timeout", "error", "max_steps_exceeded", "context_overflow"):
                        _dirty = True
                        break
            if _dirty:
                prospective_bonus = 0.0
                task_bonuses = [0.0] * len(task_rewards_list)
                task_returns = list(task_rewards_list)
                chain_reward = sum(task_returns)
                pu_clean_gated = 1.0
                sage_gen_bonus = 0.0
                sage_use_bonus = 0.0
                skillrise_curate_return = 0.0

        # 7) 落盘 trace（含 chain/task 截断检测）
        if self.trace_dir:
            await asyncio.to_thread(
                _write_chain_trace,
                trace_dir=self.trace_dir,
                group_id=group_id,
                chain_g=chain_g,
                task_chain=task_chain,
                all_env_results=all_env_results,
                all_harness_actions_per_task=all_harness_actions_per_task,
                task_rewards_list=task_rewards_list,
                frozen_rewards=frozen_rewards,
                chain_reward=chain_reward,
                prospective_bonus=prospective_bonus,
                sandbox_snapshot=sandbox_snapshot,
                task_read_sets=task_read_sets,
                truncation=trunc_report_full,
                credit=self.credit,
                sage_gen_bonus=sage_gen_bonus,
                sage_use_bonus=sage_use_bonus,
                skillrise_curate_return=skillrise_curate_return,
            )

        if self.abort_on_packing_truncation and trunc_report.get("packing_trunc_hit", 0) > 0:
            raise RuntimeError(
                f"[STHARCAgentLoop] abort_on_packing_truncation group={group_id} g={chain_g} "
                f"detail={trunc_report}"
            )
        if self.abort_on_exec_truncation and trunc_report.get("exec_trunc_hit", 0) > 0:
            raise RuntimeError(
                f"[STHARCAgentLoop] abort_on_exec_truncation group={group_id} g={chain_g} "
                f"detail={trunc_report}"
            )

        # 汇聚 scalar 指标（供 SwanLab / trainer 直接 log，无需自行解析列表）
        import math as _math
        avg_task_reward = total_reward / K if K > 0 else 0.0
        avg_frozen_reward = sum(frozen_rewards) / K if K > 0 else 0.0
        # delta = 平均增量收益（衡量 harness 写操作对任务的实际帮助）
        if _math.isnan(avg_task_reward) or _math.isnan(avg_frozen_reward):
            avg_delta_reward = float("nan")
        else:
            avg_delta_reward = avg_task_reward - avg_frozen_reward

        reward_extra_info = _build_stharc_reward_extra_info(
            all_reward_infos,
            all_builder_outs,
            all_env_results,
            response_ids,
            total_reward=total_reward,
            prospective_bonus=prospective_bonus,
            chain_reward=chain_reward,
            chain_length=K,
            num_mutations=len(harness_mutations),
            avg_task_reward=avg_task_reward,
            avg_frozen_reward=avg_frozen_reward,
            avg_delta_reward=avg_delta_reward,
            truncation=trunc_report,
        )
        reward_extra_info["pu_clean_gated"] = float(pu_clean_gated)
        reward_extra_info["prospective_bonus_raw"] = float(prospective_bonus_raw)
        reward_extra_info["sage_gen_bonus"] = float(sage_gen_bonus)
        reward_extra_info["sage_use_bonus"] = float(sage_use_bonus)
        reward_extra_info["skillrise_curate_return"] = float(skillrise_curate_return)
        reward_extra_info["skillrise_n_curate"] = float(len(skillrise_docs))

        # AgentLoopOutput 是 pydantic BaseModel，其 as_dict()（main_ppo_sync.py
        # AgentLoopWorkerTQ._agent_loop_postprocess 里用来构造写入 TransferQueue
        # 的记录）调用的是 model_dump(exclude_unset=True)——num_turns 字段哪怕有
        # 默认值 0，只要构造时没有显式传参就不会出现在结果字典里，进而不会被写入
        # TransferQueue。main_ppo_sync.py._validate()/fit() 里都会硬性
        # data.pop("num_turns")，缺了这个 key 会直接 KeyError 炸掉整个 validate/
        # train step。这里用整条 chain 里所有任务的 step 数之和作为 num_turns
        # （与 AgenticAgentLoop.run() 用 builder_out.summary.num_steps 的口径一致）。
        total_num_turns = sum(int(bo.summary.num_steps) for bo in all_builder_outs) if all_builder_outs else 0

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            reward_score=float(chain_reward),
            num_turns=total_num_turns,
            metrics=AgentLoopMetrics(),
            extra_fields={
                "stharc_task_credit": pack_task_credit(
                    task_returns, response_task_ids, response_mask
                ),
                "stharc_metrics": {
                    # ── 链级汇总 ─────────────────────────────────────────
                    "total_reward":       total_reward,
                    "prospective_bonus":  prospective_bonus,
                    "prospective_bonus_raw": prospective_bonus_raw,
                    "pu_clean_gated":     float(pu_clean_gated),
                    "chain_reward":       chain_reward,
                    "credit":             self.credit,
                    "sage_gen_bonus":     sage_gen_bonus,
                    "sage_use_bonus":     sage_use_bonus,
                    "skillrise_curate_return": skillrise_curate_return,
                    "skillrise_n_curate":     float(len(skillrise_docs)),
                    "chain_length":       K,
                    "num_mutations":      len(harness_mutations),
                    # ── 任务级列表（供 trace / debug）────────────────────
                    "task_rewards":       task_rewards_list,
                    "frozen_rewards":     frozen_rewards,
                    # ── SwanLab scalar 指标 ──────────────────────────────
                    "avg_task_reward":    avg_task_reward,
                    "avg_frozen_reward":  avg_frozen_reward,
                    "avg_delta_reward":   avg_delta_reward,
                    # ── 截断统计（理想 any_trunc_hit=0）─────────────────
                    **{f"trunc/{k}": v for k, v in trunc_report.items()},
                    # ── 标识字段 ─────────────────────────────────────────
                    "group_id":           group_id,
                    "chain_g":            chain_g,
                },
                "reward_extra_info": reward_extra_info,
            },
        )

    # ------------------------------------------------------------------
    # 复用 AgenticAgentLoop 的 vLLM 探测方法
    # ------------------------------------------------------------------

    async def _async_detect_vllm_url(self) -> Optional[str]:
        try:
            lb = getattr(getattr(self, "server_manager", None), "_load_balancer", None)
            if lb is None:
                return self.vllm_base_url
            server_ids: list = await lb.get_all_servers.remote()
            if not server_ids:
                return self.vllm_base_url
            addr = server_ids[0]
            if not addr.startswith("http"):
                addr = f"http://{addr}"
            return f"{addr}/v1"
        except Exception:
            return self.vllm_base_url

    async def _async_detect_vllm_model_name(self, base_url: str) -> Optional[str]:
        def _fetch():
            try:
                import requests as _req
                resp = _req.get(f"{base_url}/models", timeout=5)
                resp.raise_for_status()
                data = resp.json()
                models = data.get("data") or []
                if models:
                    return models[0].get("id")
            except Exception:
                pass
            return None

        return await asyncio.to_thread(_fetch)


def _serialize_record(record: Any) -> Dict[str, Any]:
    """把单个 TraceRecord 序列化为纯 dict（保留原始 messages 和 response）。

    兼容 dataclass 和 dict 两种格式（TraceRecord 在不同版本可能结构略有差异）。
    """
    if isinstance(record, dict):
        msgs = record.get("messages") or []
        # 优先取 "response"（TraceRecord 标准字段），再 fallback 到 "response_text"
        resp = record["response"] if "response" in record else record.get("response_text", "")
        model = record.get("model")
        ts_start = record.get("ts_start", 0.0)
        ts_end = record.get("ts_end", 0.0)
    else:
        msgs = list(getattr(record, "messages", None) or [])
        _resp = getattr(record, "response", _MISSING := object())
        resp = _resp if _resp is not _MISSING else getattr(record, "response_text", "")
        model = getattr(record, "model", None)
        ts_start = float(getattr(record, "ts_start", 0.0) or 0.0)
        ts_end = float(getattr(record, "ts_end", 0.0) or 0.0)

    # messages 里每条也确保是普通 dict
    serialized_msgs = []
    for m in msgs:
        if isinstance(m, dict):
            serialized_msgs.append(dict(m))
        else:
            serialized_msgs.append({
                "role": getattr(m, "role", "unknown"),
                "content": getattr(m, "content", str(m)),
            })

    return {
        "messages": serialized_msgs,
        "response": str(resp or ""),
        "model": model,
        "ts_start": ts_start,
        "ts_end": ts_end,
    }


def _write_chain_trace(
    trace_dir: str,
    group_id: str,
    chain_g: int,
    task_chain: List[Dict[str, Any]],
    all_env_results: List[Any],
    all_harness_actions_per_task: List[List[Dict[str, Any]]],
    task_rewards_list: List[float],
    frozen_rewards: List[float],
    chain_reward: float,
    prospective_bonus: float,
    sandbox_snapshot: Optional[Dict[str, Any]] = None,
    task_read_sets: Optional[List[Any]] = None,
    truncation: Optional[Dict[str, float]] = None,
    credit: str = "pu",
    sage_gen_bonus: float = 0.0,
    sage_use_bonus: float = 0.0,
    skillrise_curate_return: float = 0.0,
) -> None:
    """将一条完整 chain 的对话轨迹写入 JSONL 文件。

    文件路径：{trace_dir}/{group_id}_{chain_g}.jsonl
      - 每次追加写入一行 JSON（一次训练过程中同一 group+chain 会执行多步，每步 append 一条）
      - 每行包含该 chain 的完整信息：每个任务的原始 messages/response + harness 变更
        + cleanup 前的沙箱快照（skill 清单 / 变更 skill 全文 / 最终 memory）

    日志结构::

        {
          "group_id": "chain_00001",
          "chain_id": "credit_diagnostic_memory_00000",
          "chain_g": 0,
          "data_source": "credit_diagnostic_memory",
          "chain_reward": 0.85,
          "prospective_bonus": 0.12,
          "provenance_answer_key": {"mem::secret_fact_0": {...}},  # CreditAlign 用
          "sandbox": { ... },
          "tasks": [
            {
              "task_idx": 0,
              "task_id": "write_0",
              "harness_op": {"type": "memory_write", "writes_oid": "mem::secret_fact_0", ...},
              "query": "...",
              "reward": 0.0,
              "frozen_reward": 0.0,
              "delta_reward": 0.0,
              "env_status": "ok",
              "num_tool_calls": 2,
              "final_answer": "...",
              "turns": [ ... ],
              "harness_mutations": [ ... ],
              "read_set": ["skill::skill_xxx", "mem::key_a"]
            },
            ...
          ]
        }
    """
    import json as _json
    import os as _os

    _os.makedirs(trace_dir, exist_ok=True)
    filepath = _os.path.join(trace_dir, f"{group_id}_{chain_g}.jsonl")

    tasks_data = []
    for k, (task, env_result, harness_actions_k) in enumerate(
        zip(task_chain, all_env_results, all_harness_actions_per_task)
    ):
        query = _extract_query(task.get("raw_prompt") or task.get("messages") or task.get("query", ""))
        env_info = dict(getattr(env_result, "env_info", {}) or {})

        turns = [_serialize_record(r) for r in (getattr(env_result, "records", None) or [])]

        b_k = frozen_rewards[k] if k < len(frozen_rewards) else 0.0
        read_set_k = task_read_sets[k] if task_read_sets is not None and k < len(task_read_sets) else None
        # 从 task.extra_info._provenance_debug 抽出诊断元数据（不参与训练，只供 CreditAlign）
        prov_dbg = {}
        if isinstance(task, dict):
            extra = task.get("extra_info") if isinstance(task.get("extra_info"), dict) else {}
            raw_dbg = extra.get("_provenance_debug")
            if isinstance(raw_dbg, dict):
                prov_dbg = raw_dbg
        # packing 侧 per-task 截断（来自 chain truncation.per_task）+ env 执行截断
        pack_t = None
        if isinstance(truncation, dict):
            for pt in truncation.get("per_task") or []:
                if isinstance(pt, dict) and int(pt.get("task_idx", -1)) == k:
                    pack_t = pt
                    break
        env_t = env_info.get("truncation") if isinstance(env_info.get("truncation"), dict) else {}
        task_trunc = {
            "packing": pack_t,
            "exec": env_t,
            "any_trunc": bool(
                (pack_t or {}).get("any_trunc")
                or env_t.get("obs_char_trunc_count")
                or env_t.get("context_overflow")
                or env_t.get("max_steps_exceeded")
                or env_info.get("status") in ("context_overflow", "max_steps_exceeded")
            ),
        }
        tasks_data.append({
            "task_idx":         k,
            "task_id":          prov_dbg.get("task_id"),
            "harness_op":       prov_dbg.get("harness_op"),
            "query":            query,
            "reward":           task_rewards_list[k] if k < len(task_rewards_list) else 0.0,
            "frozen_reward":    b_k,
            "delta_reward":     (task_rewards_list[k] if k < len(task_rewards_list) else 0.0) - b_k,
            "env_status":       env_info.get("status", "unknown"),
            "env_error":        env_info.get("error"),
            "final_answer":     str(getattr(env_result, "final_answer", "") or ""),
            "elapsed_sec":      env_info.get("elapsed_sec", 0.0),
            "num_mainbrain_calls": env_info.get("num_mainbrain_calls", len(turns)),
            "num_tool_calls":   env_info.get("num_tool_calls", 0),
            "appworld_success": env_info.get("appworld_success"),
            "appworld_eval":    env_info.get("appworld_eval"),
            "truncation":       task_trunc,
            "turns":            turns,
            "harness_mutations": harness_actions_k,
            "read_set":         sorted(read_set_k) if read_set_k is not None else [],
        })

    # chain 级 provenance：取首个非空 answer_key（同 chain 内各 task 共享）
    chain_data_source = ""
    chain_id_from_prov = group_id
    provenance_answer_key: Dict[str, Any] = {}
    for task in task_chain:
        if not isinstance(task, dict):
            continue
        extra = task.get("extra_info") if isinstance(task.get("extra_info"), dict) else {}
        dbg = extra.get("_provenance_debug") if isinstance(extra.get("_provenance_debug"), dict) else {}
        if not chain_data_source and dbg.get("data_source"):
            chain_data_source = str(dbg.get("data_source"))
        if dbg.get("chain_id"):
            chain_id_from_prov = str(dbg.get("chain_id"))
        if not provenance_answer_key and isinstance(dbg.get("provenance_answer_key"), dict):
            provenance_answer_key = dict(dbg.get("provenance_answer_key") or {})

    entry = {
        "group_id":          group_id,
        "chain_id":          chain_id_from_prov,
        "chain_g":           chain_g,
        "data_source":       chain_data_source,
        "chain_reward":      chain_reward,
        "prospective_bonus": prospective_bonus,
        "credit":            credit,
        "sage_gen_bonus":    sage_gen_bonus,
        "sage_use_bonus":    sage_use_bonus,
        "skillrise_curate_return": skillrise_curate_return,
        "frozen_rewards":    frozen_rewards,
        "provenance_answer_key": provenance_answer_key,
        "sandbox":           sandbox_snapshot,
        "truncation":        truncation or {},
        "tasks":             tasks_data,
    }

    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        logger.warning("[STHARCAgentLoop] 写 trace 失败 path=%s: %s", filepath, exc)


def _empty_concat_trunc_stats() -> Dict[str, float]:
    return {
        "n_task_resp_trunc": 0.0,
        "n_task_ctx_trunc": 0.0,
        "task_resp_tokens_dropped": 0.0,
        "task_ctx_tokens_dropped": 0.0,
        "task_resp_loss_tokens_dropped": 0.0,
        "raw_prompt0_len": 0.0,
        "raw_chain_response_len": 0.0,
    }


def _to_int_list(ids: Any) -> List[int]:
    if ids is None:
        return []
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return list(ids)


def _concat_builder_outputs(
    all_builder_outs: List[Any],
    *,
    max_task_response_length: Optional[int] = None,
    max_task_context_length: Optional[int] = None,
) -> Any:
    """拼接 K 个任务的 TrajectoryBuilder 输出（方向 B）。

    结构：
      prompt_ids  = task_0 的 prompt（system + tools + task_0 用户问题）
      response_ids = task_0 rollout
                   + [task_1 prompt tokens, mask=0]   ← task_1 用户问题作为环境上下文
                   + task_1 rollout
                   + [task_2 prompt tokens, mask=0]   ← task_2 用户问题作为环境上下文
                   + task_2 rollout
                   + ...

    task_k（k≥1）的 prompt_ids 以 mask=0 插入，使模型在 backward 时可以
    attend 到后续任务的问题描述，同时不对这些 context token 计算 loss。

    task_k 的 rollout 内部 mask 由 TrajectoryBuilder 独立生成：
      - assistant 生成（Thought/Action/Final）：mask=1，计算 loss
      - 环境观察（工具返回）：mask=0，不计算 loss

    max_task_response_length / max_task_context_length：
      拼 chain 前按任务强制截断，避免前几个 task 过长挤掉后续 task 的 loss token。
      与启动脚本公式 CHAIN_RESP_LEN = K*MAX_RESP + (K-1)*MAX_PROMPT 对齐。

    返回对象额外带：
      truncation_stats: 标量计数（供 SwanLab）
      per_task_truncation: 每 task 是否被截、丢了多少 token（供 trace 排查）
    """
    trunc = _empty_concat_trunc_stats()
    per_task: List[Dict[str, Any]] = []

    if not all_builder_outs:
        class _Empty:
            prompt_ids = []
            response_ids = []
            response_mask = []
            response_task_ids = []
            summary = None
            truncation_stats = trunc
            per_task_truncation = per_task
        return _Empty()

    first = all_builder_outs[0]
    if len(all_builder_outs) == 1:
        class _One:
            pass
        out = _One()
        out.prompt_ids = _to_int_list(first.prompt_ids)
        rids = _to_int_list(first.response_ids)
        rmask = _to_int_list(first.response_mask)
        if len(rmask) != len(rids) or any(m not in (0, 1) for m in rmask):
            raise ValueError("STHARC builder response mask must be binary and match response length")
        trunc["raw_prompt0_len"] = float(len(out.prompt_ids))
        trunc["raw_chain_response_len"] = float(len(rids))
        t0 = {
            "task_idx": 0,
            "resp_trunc": False,
            "ctx_trunc": False,
            "resp_tokens_dropped": 0,
            "ctx_tokens_dropped": 0,
            "resp_loss_tokens_dropped": 0,
            "raw_resp_len": len(rids),
            "raw_ctx_len": 0,
            "any_trunc": False,
        }
        if max_task_response_length and len(rids) > int(max_task_response_length):
            dropped = len(rids) - int(max_task_response_length)
            loss_dropped = sum(1 for m in rmask[int(max_task_response_length):] if int(m) == 1)
            trunc["n_task_resp_trunc"] = 1.0
            trunc["task_resp_tokens_dropped"] = float(dropped)
            trunc["task_resp_loss_tokens_dropped"] = float(loss_dropped)
            t0.update(
                resp_trunc=True,
                resp_tokens_dropped=dropped,
                resp_loss_tokens_dropped=loss_dropped,
                any_trunc=True,
            )
            rids = rids[: int(max_task_response_length)]
            rmask = rmask[: int(max_task_response_length)]
        per_task.append(t0)
        out.response_ids = rids
        out.response_mask = rmask
        out.response_task_ids = [0 if m == 1 else -1 for m in rmask]
        out.summary = first.summary
        out.truncation_stats = trunc
        out.per_task_truncation = per_task
        return out

    all_response_ids: List[int] = []
    all_response_mask: List[int] = []
    all_response_task_ids: List[int] = []

    for k, bo in enumerate(all_builder_outs):
        tinfo: Dict[str, Any] = {
            "task_idx": k,
            "resp_trunc": False,
            "ctx_trunc": False,
            "resp_tokens_dropped": 0,
            "ctx_tokens_dropped": 0,
            "resp_loss_tokens_dropped": 0,
            "raw_resp_len": 0,
            "raw_ctx_len": 0,
            "any_trunc": False,
        }
        # k≥1：先把 task_k 的 prompt_ids 以 mask=0 插入，提供问题上下文
        if k > 0:
            ctx_ids = _to_int_list(bo.prompt_ids)
            tinfo["raw_ctx_len"] = len(ctx_ids)
            if max_task_context_length and len(ctx_ids) > int(max_task_context_length):
                dropped = len(ctx_ids) - int(max_task_context_length)
                trunc["n_task_ctx_trunc"] += 1.0
                trunc["task_ctx_tokens_dropped"] += float(dropped)
                tinfo["ctx_trunc"] = True
                tinfo["ctx_tokens_dropped"] = dropped
                tinfo["any_trunc"] = True
                ctx_ids = ctx_ids[: int(max_task_context_length)]
            all_response_ids.extend(ctx_ids)
            all_response_mask.extend([0] * len(ctx_ids))
            all_response_task_ids.extend([-1] * len(ctx_ids))

        rids = _to_int_list(bo.response_ids)
        rmask = _to_int_list(bo.response_mask)
        if len(rmask) != len(rids) or any(m not in (0, 1) for m in rmask):
            raise ValueError("STHARC builder response mask must be binary and match response length")
        tinfo["raw_resp_len"] = len(rids)
        if max_task_response_length and len(rids) > int(max_task_response_length):
            dropped = len(rids) - int(max_task_response_length)
            loss_dropped = sum(1 for m in rmask[int(max_task_response_length):] if int(m) == 1)
            trunc["n_task_resp_trunc"] += 1.0
            trunc["task_resp_tokens_dropped"] += float(dropped)
            trunc["task_resp_loss_tokens_dropped"] += float(loss_dropped)
            tinfo["resp_trunc"] = True
            tinfo["resp_tokens_dropped"] = dropped
            tinfo["resp_loss_tokens_dropped"] = loss_dropped
            tinfo["any_trunc"] = True
            rids = rids[: int(max_task_response_length)]
            rmask = rmask[: int(max_task_response_length)]
        all_response_ids.extend(rids)
        all_response_mask.extend(rmask)
        all_response_task_ids.extend(k if m == 1 else -1 for m in rmask)
        per_task.append(tinfo)

    prompt_ids = _to_int_list(first.prompt_ids)
    trunc["raw_prompt0_len"] = float(len(prompt_ids))
    trunc["raw_chain_response_len"] = float(len(all_response_ids))

    class _Concat:
        pass

    out = _Concat()
    out.prompt_ids = prompt_ids
    out.response_ids = all_response_ids
    out.response_mask = all_response_mask
    out.response_task_ids = all_response_task_ids
    out.summary = all_builder_outs[-1].summary
    out.truncation_stats = trunc
    out.per_task_truncation = per_task
    return out


def _measure_chain_pack_truncation(
    prompt_ids: List[int],
    response_ids: List[int],
    response_mask: List[int],
    *,
    prompt_limit: Optional[int],
    response_limit: Optional[int],
) -> Dict[str, float]:
    """链级 packing 截断：prompt0 / 整链 response 是否被 response_length 硬截。"""
    stats = {
        "prompt0_trunc": 0.0,
        "prompt0_tokens_dropped": 0.0,
        "chain_resp_trunc": 0.0,
        "chain_resp_tokens_dropped": 0.0,
        "chain_resp_loss_tokens_dropped": 0.0,
        "packed_prompt_len": float(len(prompt_ids)),
        "packed_response_len": float(len(response_ids)),
    }
    if prompt_limit and len(prompt_ids) > int(prompt_limit):
        stats["prompt0_trunc"] = 1.0
        stats["prompt0_tokens_dropped"] = float(len(prompt_ids) - int(prompt_limit))
    if response_limit and len(response_ids) > int(response_limit):
        lim = int(response_limit)
        stats["chain_resp_trunc"] = 1.0
        stats["chain_resp_tokens_dropped"] = float(len(response_ids) - lim)
        if response_mask:
            stats["chain_resp_loss_tokens_dropped"] = float(
                sum(1 for m in response_mask[lim:] if int(m) == 1)
            )
    return stats


def _aggregate_env_truncation_stats(env_results: List[Any]) -> Dict[str, float]:
    """从各 task env_info 汇总执行期截断（观测字符 / context overflow / max_steps）。"""
    stats = {
        "n_obs_char_trunc": 0.0,
        "obs_chars_dropped": 0.0,
        "n_context_overflow": 0.0,
        "n_max_steps_exceeded": 0.0,
        "n_env_timeout": 0.0,
        "n_env_error": 0.0,
    }
    for er in env_results:
        info = dict(getattr(er, "env_info", None) or {})
        t = info.get("truncation") if isinstance(info.get("truncation"), dict) else {}
        stats["n_obs_char_trunc"] += float(t.get("obs_char_trunc_count", 0) or 0)
        stats["obs_chars_dropped"] += float(t.get("obs_chars_dropped", 0) or 0)
        status = str(info.get("status") or "")
        if status == "context_overflow" or bool(t.get("context_overflow")):
            stats["n_context_overflow"] += 1.0
        if status == "max_steps_exceeded" or bool(t.get("max_steps_exceeded")):
            stats["n_max_steps_exceeded"] += 1.0
        if status == "timeout":
            stats["n_env_timeout"] += 1.0
        if status == "error":
            stats["n_env_error"] += 1.0
    return stats


def _finalize_truncation_report(
    concat_stats: Dict[str, float],
    pack_stats: Dict[str, float],
    env_stats: Dict[str, float],
) -> Dict[str, float]:
    """统一截断报表；any_trunc=1 表示本 chain 存在内容丢失（含观测截断）。"""
    report = {}
    report.update(concat_stats or {})
    report.update(pack_stats or {})
    report.update(env_stats or {})
    packing_hit = (
        report.get("n_task_resp_trunc", 0)
        + report.get("n_task_ctx_trunc", 0)
        + report.get("prompt0_trunc", 0)
        + report.get("chain_resp_trunc", 0)
    )
    exec_hit = (
        report.get("n_obs_char_trunc", 0)
        + report.get("n_context_overflow", 0)
        + report.get("n_max_steps_exceeded", 0)
    )
    report["packing_trunc_hit"] = 1.0 if packing_hit > 0 else 0.0
    report["exec_trunc_hit"] = 1.0 if exec_hit > 0 else 0.0
    report["any_trunc_hit"] = 1.0 if (packing_hit + exec_hit) > 0 else 0.0
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# frozen chain 共享缓存（文件锁幂等，与 HarnessState.create_h_init 模式一致）
# ═══════════════════════════════════════════════════════════════════════════════

def _get_or_compute_frozen_rewards(
    env: Any,
    task_chain: List[Dict[str, Any]],
    H_init: Any,
    group_id: str,
    trace_dir: Optional[str] = None,
    reward: Any = None,
    builder: Any = None,
    global_step: int = -1,

) -> List[float]:
    """同一 group、同一 training step 的 G 条 chain 共享一份 frozen_rewards。

    第一个拿到文件锁的 worker 调用 run_frozen_chain 计算并将结果写入
    ``/tmp/rl_stharc/group_{group_id}/frozen_rewards.json``；
    其余 worker 等锁释放后直接读取缓存，不再重复计算。

    跨 step 失效策略（双保险）：
      1. 缓存 payload 含 ``global_step``，step 不一致则忽略
      2. ``HarnessState.create_h_init`` 重建 H_init 时删除该文件

    Args:
        env:         AiToCReactEnv 实例
        task_chain:  K 个任务列表
        H_init:      本 chain 的 harness 初始状态
        group_id:    GRPO group 标识（同一 group 的 G 条 chain 传相同值）
        trace_dir:   trace 写入目录（非 None 时第一个 worker 写 frozen chain trace）
        reward:      与 live 相同的 reward（builder.build + reward.compute）
        builder:     与 live 相同的 TrajectoryBuilder
        global_step: 当前 trainer global_steps（同 step 内共享，跨 step 失效）

    Returns:
        frozen_rewards: list[float]，长度为 len(task_chain)
    """
    import fcntl
    import json
    import tempfile
    from pathlib import Path

    from .stharc.core import run_frozen_chain
    from .stharc.frozen_cache import (
        build_frozen_rewards_payload,
        load_frozen_rewards_cache,
    )
    from .stharc.harness_state import RL_SANDBOX_BASE

    frozen_path = RL_SANDBOX_BASE / f"group_{group_id}" / "frozen_rewards.json"
    lock_path = RL_SANDBOX_BASE / f"group_{group_id}_frozen.lock"

    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    expected_len = len(task_chain)

    def _compute() -> Tuple[List[float], List[Any]]:
        return run_frozen_chain(
            env,
            task_chain,
            H_init,
            reward=reward,
            builder=builder,

        )

    def _persist(rewards: List[float]) -> None:
        payload = build_frozen_rewards_payload(rewards, global_step)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=frozen_path.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            Path(tmp_path).rename(frozen_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    try:
        with open(lock_path, "w") as _lf:
            fcntl.flock(_lf.fileno(), fcntl.LOCK_EX)
            try:
                if frozen_path.exists():
                    try:
                        cached = load_frozen_rewards_cache(
                            json.loads(frozen_path.read_text(encoding="utf-8")),
                            global_step=global_step,
                            expected_len=expected_len,
                        )
                    except (ValueError, TypeError, OSError):
                        cached = None
                    if cached is not None:
                        return cached
                    logger.warning(
                        "[STHARCAgentLoop] 忽略过期/无效 frozen 缓存"
                        "（group=%s step=%s），重新计算",
                        group_id,
                        global_step,
                    )

                rewards, frozen_env_results = _compute()

                if trace_dir:
                    try:
                        _write_frozen_chain_trace(
                            trace_dir=trace_dir,
                            group_id=group_id,
                            task_chain=task_chain,
                            frozen_env_results=frozen_env_results,
                            frozen_rewards=rewards,
                        )
                    except Exception as _te:
                        logger.warning(
                            "[STHARCAgentLoop] 写 frozen trace 失败（group=%s）: %s",
                            group_id, _te,
                        )

                _persist(rewards)
                return rewards
            finally:
                fcntl.flock(_lf.fileno(), fcntl.LOCK_UN)

    except Exception:
        # 任何异常（锁失败、IO 错误等）均回退到实时计算，不中断训练
        logger.warning(
            "[STHARCAgentLoop] frozen_rewards 共享缓存失败（group=%s），回退到实时计算",
            group_id,
        )
        rewards, frozen_env_results = _compute()
        if trace_dir:
            try:
                _write_frozen_chain_trace(
                    trace_dir=trace_dir,
                    group_id=group_id,
                    task_chain=task_chain,
                    frozen_env_results=frozen_env_results,
                    frozen_rewards=rewards,
                )
            except Exception as _te:
                logger.warning(
                    "[STHARCAgentLoop] 写 frozen trace 失败（fallback，group=%s）: %s",
                    group_id, _te,
                )
        return rewards


def _write_frozen_chain_trace(
    trace_dir: str,
    group_id: str,
    task_chain: List[Dict[str, Any]],
    frozen_env_results: List[Any],
    frozen_rewards: List[float],
) -> None:
    """将 frozen chain 的对话轨迹写入 JSONL 文件。

    文件路径：{trace_dir}/{group_id}_frozen.jsonl
    每个 group 追加一行，结构与标准 chain trace 一致，但：
      - chain_g 固定为 "frozen"
      - 无 harness_mutations（写操作被 mock，不产生真实变更）
      - 每个任务含完整 turns（LLM 对话记录）和 frozen_reward（即 b_j）

    日志结构::

        {
          "group_id": "chain_00001",
          "chain_g": "frozen",
          "frozen_rewards": [0.3, 0.0, 0.5],
          "tasks": [
            {
              "task_idx": 0,
              "query": "...",
              "frozen_reward": 0.3,
              "env_status": "ok",
              "final_answer": "...",
              "turns": [...]
            },
            ...
          ]
        }
    """
    import json as _json
    import os as _os

    _os.makedirs(trace_dir, exist_ok=True)
    filepath = _os.path.join(trace_dir, f"{group_id}_frozen.jsonl")

    tasks_data = []
    for k, (task, env_result) in enumerate(zip(task_chain, frozen_env_results)):
        query = _extract_query(task.get("raw_prompt") or task.get("messages") or task.get("query", ""))
        env_info = dict(getattr(env_result, "env_info", {}) or {})
        turns = [_serialize_record(r) for r in (getattr(env_result, "records", None) or [])]

        tasks_data.append({
            "task_idx":            k,
            "query":               query,
            "frozen_reward":       frozen_rewards[k] if k < len(frozen_rewards) else 0.0,
            "env_status":          env_info.get("status", "unknown"),
            "final_answer":        str(getattr(env_result, "final_answer", "") or ""),
            "elapsed_sec":         env_info.get("elapsed_sec", 0.0),
            "num_mainbrain_calls": env_info.get("num_mainbrain_calls", len(turns)),
            "turns":               turns,
        })

    entry = {
        "group_id":      group_id,
        "chain_g":       "frozen",
        "frozen_rewards": frozen_rewards,
        "tasks":         tasks_data,
    }

    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        logger.warning("[STHARCAgentLoop] 写 frozen trace 失败 path=%s: %s", filepath, exc)