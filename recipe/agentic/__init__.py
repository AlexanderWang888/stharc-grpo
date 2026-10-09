"""AgenticRL recipe for verl.

业务无关的 agentic 训练框架（GRPO），三层解耦：
  业务（可换）  ←→  TrajectorySummary（标准结构）  ←→  Reward（独立配置）

参考方案：rl_plan_final.md
"""