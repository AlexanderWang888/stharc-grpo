"""GSB 维度绝对打分 prompt（单轨迹，非 A/B 比较）。

评分维度（5 个，各自独立打分，代码侧等权平均为最终分）：
  1. 意图/需求识别度（intent）
  2. 内容准确性（accuracy）
  3. 内容完整性（completeness）
  4. 实用性（practicality）
  5. 表达质量（expression）

每个维度采用二值打分：符合标准为 1，不符合为 0，不允许中间值。
"""

from __future__ import annotations

import re

# 中国居民身份证号：15位纯数字 或 17位数字+末位数字/X
_RE_ID_CARD = re.compile(r"\b\d{17}[\dXx]\b|\b\d{15}\b")
# 手机号：1开头的11位数字（前后非数字）
_RE_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")


def _mask_sensitive(text: str) -> str:
    """将身份证号和手机号替换为占位符，避免触发下游敏感词过滤。"""
    text = _RE_ID_CARD.sub("[IDCARD]", text)
    text = _RE_PHONE.sub("[PHONE]", text)
    return text


JUDGE_SYSTEM_PROMPT = """\
你是一位经验丰富的保险 AI 顾问系统质量评测专家，具备保险产品知识储备和用户体验敏感度。
你的任务是对单个 AI 系统回复在 5 个维度上分别进行二值打分（0 或 1），而非与其他系统比较。
每个维度只能打 0 或 1：符合标准为 1，不符合为 0，不允许任何中间值。
"""

_JUDGE_USER_PROMPT_TEMPLATE = """\
## 评测任务说明

请对以下 AI 系统（ReAct orchestrator）的单条回复，在以下 5 个维度上**分别**进行二值打分。
**每个维度只能填 0 或 1，不允许其他值：符合标准填 1，不符合填 0。**

---

## 评分维度

### 1. 意图/需求识别度（intent）
- 用户核心诉求是否被准确识别并针对性回应？
- 是否完整处理了用户问题中的多个子诉求？
- 是否存在「答非所问」或遗漏关键问题维度的情况？
- **打 1 的标准**：核心诉求被准确识别且针对性回应，无明显答非所问；**否则打 0**。

### 2. 内容准确性（accuracy）
- 产品名称、保额、保障责任、免责条款等核心信息是否真实、无混淆？
- 是否存在捏造产品特性、错误引用条款或数据的情况？
- **打 1 的标准**：核心信息真实无误，无捏造或明显错误；**否则打 0**。

### 3. 内容完整性（completeness）
- 是否覆盖了用户询问场景下的关键保障维度？
- 是否给出了「选择哪个更合适」的判断依据（如适用）？
- 是否存在回复严重偏短、关键信息缺失的情况？
- **打 1 的标准**：关键信息完整，无严重缺失；**否则打 0**。

### 4. 实用性（practicality）
- 是否给出了具体、可操作的建议（而非泛泛而谈）？
- 是否结合用户的具体场景提供个性化分析？
- **打 1 的标准**：给出了具体可操作的建议，有个性化分析；**否则打 0**。

### 5. 表达质量（expression）
- 结构是否清晰易读？表格/列表等格式是否恰当使用？
- 是否存在明显冗余、套话或逻辑跳跃？
- **打 1 的标准**：结构清晰，无明显冗余或逻辑混乱；**否则打 0**。

---

## 特殊情况

| 情况 | 处理方式 |
|------|----------|
| 无有效输出 / 空回复 | 所有维度 score = 0 |
| 转人工/直返类意图且回复合理简短 | 按实际质量正常打分 |

---

## 输入信息

**用户问题：**
{user_query}

**系统回复：**
{model_answer}

---

## 输出要求

请严格按以下 JSON 格式输出，不得添加任何额外文字、代码块标记或注释：

{{"dimensions": {{"intent": 0, "accuracy": 0, "completeness": 0, "practicality": 0, "expression": 0}}, "reason": "简洁说明（150字以内）"}}

其中每个维度只能填整数 0 或 1，最终综合分由调用方对 5 个维度等权平均计算，无需在此输出。
"""


def build_judge_user_prompt(
    user_query: str,
    model_answer: str,
    *,
    num_action_steps: int = 0,
) -> str:
    """填充模板，返回 judge user prompt。

    在拼接前对 user_query 和 model_answer 做脱敏处理，
    将手机号替换为 [PHONE]、身份证号替换为 [IDCARD]，
    避免下游 judge 服务因敏感词过滤返回 400 错误。
    """
    ans = str(model_answer).strip() if model_answer and str(model_answer) != "nan" else "（无有效输出）"
    return _JUDGE_USER_PROMPT_TEMPLATE.format(
        user_query=_mask_sensitive(str(user_query).strip()),
        model_answer=_mask_sensitive(ans),
    )
