"""STHARC-GRPO: Skill & Task Harness Agentic RL with Chains.

子模块：
  agent_loop / entry — Hydra 入口（re-export STHARCAgentLoop）
  harness_state   — HarnessState（每条 chain 独占的 harness 沙箱封装）
  harness_tracker — HarnessTracker（ContextVar；research 侧）
  ai2c_harness_patch — 不改 ai2c 源码，monkeypatch 沙箱 / tracker / frozen
  core            — PU / SAGE-style / SkillRise-style credit + run_frozen_chain
  task_chain      — task chain 构建工具（三种策略）
"""
