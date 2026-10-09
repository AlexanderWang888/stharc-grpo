"""GSB 对比打分 prompt（react vs react，正向 model=系统B，反向 model=系统A）。

与 compare_eval/gsb/prompts/react_vs_react.py 对齐，确保 reward 和线下 eval 指标语义一致。

评分方式：对 5 个维度各输出 B 相对 A 的差值（{-2,-1,0,1,2}），取平均作为 raw_score ∈ [-2, 2]。
  正向：A=baseline, B=model  → raw_score = model - baseline 视角
  反向：A=model,    B=baseline → raw_score = baseline - model 视角（调用方取反）

脱敏：对 user_query / model_answer / baseline_answer 做手机号、身份证号、银行卡号替换，
      避免触发下游 judge API 的敏感词过滤。
"""

from __future__ import annotations

import re

# 身份证号：18位（末位可为X）或 15位纯数字
# 使用 (?<!\d) / (?!\d) 代替 \b，避免多字节字符边界判断不一致
_RE_ID_CARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)|(?<!\d)\d{15}(?!\d)")
# 手机号：1 开头的 11 位数字
_RE_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
# 银行卡号：
#   分组格式：首组4位 + 2~3个4位组（空格/短横分隔）+ 可选尾组1~4位
#     16位: 4-4-4-4          → 首 + {2组4位} + {1组4位，无尾}
#     19位: 4-4-4-4-3        → 首 + {3组4位，贪婪} + {尾3位}
#     例：6212 2622 0105 8559 859
#   连续格式：62开头的16-19位纯数字（银联卡主前缀）
#     例：6212262201058559819
# 注意：仅匹配边界处的数字串，避免误替换日期/时间等其他数字
_RE_BANK_CARD = re.compile(
    r"(?<!\d)\d{4}(?:[ \-]\d{4}){2,3}(?:[ \-]\d{1,4})?(?!\d)"
    r"|(?<!\d)62\d{14,17}(?!\d)"
)


def _mask_sensitive(text: str) -> str:
    text = _RE_ID_CARD.sub("[IDCARD]", text)
    text = _RE_PHONE.sub("[PHONE]", text)
    text = _RE_BANK_CARD.sub("[BANKCARD]", text)
    return text


JUDGE_SYSTEM_PROMPT = """\
你是一位经验丰富的保险 AI 顾问系统质量评测专家，具备保险产品知识储备和用户体验敏感度。
你的任务是比较两个 AI 系统（系统A 和 系统B）对同一用户问题的回答质量，\
对每个维度给出系统B相对于系统A的差值评分。
"""

_JUDGE_USER_PROMPT_TEMPLATE = """\
## 评测任务说明

你需要比较两个 AI 系统（系统A 和 系统B）对同一用户问题的回答质量。
对以下 **4 个维度**，分别给出**系统B相对于系统A**的差值评分。

评分标准（每个维度独立打分）：
| 分值 | 含义 |
|------|------|
| **+2** | 系统B显著更好 |
| **+1** | 系统B略好 |
|  **0** | 两者相当 |
| **-1** | 系统B略差 |
| **-2** | 系统B显著更差 |

---

## 评分维度（优先级由高到低）

### 1. 意图/需求识别度（intent）
- 用户核心诉求（产品推荐/条款解读/对比选购/通用问答/投诉转人工等）是否被准确识别并针对性回应？
- 是否完整处理了用户问题中的多个子诉求？
- 是否存在「答非所问」或遗漏关键问题维度的情况？

### 2. 内容完整性（completeness）
- 是否覆盖了用户询问场景下的关键保障维度？
- 是否给出了「选择哪个更合适」的判断依据（如适用）？
- 是否存在回复严重偏短、关键信息缺失的情况？

### 3. 实用性（practicality）
- 是否给出了具体、可操作的建议（而非泛泛而谈）？
- 是否结合用户的具体场景（年龄、目的地、家庭情况等）提供个性化分析？
- 对于追问类问题，是否有效推进了决策流程？

### 4. 表达质量（expression）
- 是否存在明显冗余套话（如"感谢您的提问"、"希望以上内容对您有帮助"等开头/结尾话术）？套话越多越差。
- 是否使用了富文本格式（Markdown 加粗、标题 #、列表 -/*、表格 | 等）？此场景要求纯文字输出，系统B使用了富文本而系统A未使用，直接打 -2；反之系统B未使用而系统A使用，直接打 +2；若两者一致则按格式清晰度正常判断。
- 语言是否简洁自然、逻辑清晰，没有明显逻辑跳跃？

---

## 特殊情况处理规则

| 情况 | 处理方式 |
|------|----------|
| 两者均为空/无有效输出 | 所有维度打 0 |
| 系统B为空，系统A有实质内容 | 所有维度打 -2 |
| 系统A为空，系统B有实质内容 | 所有维度打 +2 |
| 转人工/直返类意图（两者均直返） | 按实际质量正常打分 |

---

## 输入信息

**用户问题：**
{user_query}

---

**系统A的回复：**
{answer_a}

---

**系统B的回复：**
{answer_b}

---

## 输出要求

请严格按以下 JSON 格式输出，不得添加任何额外文字、代码块标记或注释：

{{"scores": {{"intent": 0, "completeness": 0, "practicality": 0, "expression": 0}}, "reason": "简洁说明判断依据（150字以内）：指出系统B在哪个维度上优/劣，并举出具体内容差异"}}

其中每个维度只能填整数 -2、-1、0、1 或 2，最终综合分由调用方对 4 个维度等权平均计算，无需在此输出。
"""


def build_pairwise_judge_prompt(
    user_query: str,
    baseline_answer: str,
    model_answer: str,
) -> str:
    """正向：A=baseline, B=model。judge 输出 B-A = model-baseline 视角。"""
    a = str(baseline_answer).strip() if baseline_answer and str(baseline_answer) != "nan" else "（无有效输出）"
    b = str(model_answer).strip() if model_answer and str(model_answer) != "nan" else "（无有效输出）"
    return _JUDGE_USER_PROMPT_TEMPLATE.format(
        user_query=_mask_sensitive(str(user_query).strip()),
        answer_a=_mask_sensitive(a),
        answer_b=_mask_sensitive(b),
    )


def build_pairwise_judge_prompt_reversed(
    user_query: str,
    baseline_answer: str,
    model_answer: str,
) -> str:
    """反向：A=model, B=baseline。judge 输出 B-A = baseline-model 视角，调用方需取反。"""
    a = str(baseline_answer).strip() if baseline_answer and str(baseline_answer) != "nan" else "（无有效输出）"
    b = str(model_answer).strip() if model_answer and str(model_answer) != "nan" else "（无有效输出）"
    return _JUDGE_USER_PROMPT_TEMPLATE.format(
        user_query=_mask_sensitive(str(user_query).strip()),
        answer_a=_mask_sensitive(b),   # A 位放 model
        answer_b=_mask_sensitive(a),   # B 位放 baseline
    )
