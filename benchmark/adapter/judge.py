"""
Shared LLM-as-a-Judge evaluation utilities.

All benchmarks (GAIA, WebwalkerQA, XBench, DeepSearchQA) use LLM-based
judging for semantic equivalence.  This module provides common judge
functions so each adapter can call the appropriate judge.

Judge models are accessed via the OpenAI-compatible API (same API base
as the main model — defaults to OpenRouter).
"""

import json
import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def _get_judge_client(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
):
    """Lazy-create an OpenAI client for judging."""
    from openai import OpenAI

    key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base = api_base or os.getenv("OPENAI_API_BASE") or "https://openrouter.ai/api/v1"
    return OpenAI(api_key=key, base_url=base)


def _judge_max_tokens() -> int:
    # Without an explicit max_tokens the provider default output cap applies,
    # which can truncate the judge's rationale+JSON on long set answers; the
    # json_repair fallback then salvages a partial object with missing counts.
    return int(os.getenv("JUDGE_MAX_TOKENS", "8192"))


# ── Generic English judge (used by GAIA & WebwalkerQA) ──────────────

def judge_equivalence(
    question: str,
    golden_answer: str,
    pred_answer: Any,
    *,
    model: str = "qwen/qwen3.5-122b-a10b",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Dict[str, str]:
    """LLM-based semantic equivalence judge (English, GAIA-style).

    Returns dict with keys: question, judgement, golden_answer, pred_answer.
    judgement is one of: "correct", "incorrect", "error".
    """
    # Normalise pred_answer
    if isinstance(pred_answer, dict):
        pred_answer = pred_answer.get("answer", str(pred_answer))
    pred_answer = str(pred_answer).strip() if pred_answer else ""

    if not pred_answer:
        return {
            "question": question,
            "judgement": "incorrect",
            "golden_answer": golden_answer,
            "pred_answer": pred_answer,
        }

    prompt = f"""Please determine if the predicted answer is equivalent to the labeled answer.
Question:  {question}
Labeled Answer:  {golden_answer}
Predicted Answer: {pred_answer}
Are these answers equivalent?
The output should in the following json format:
{{
  "rationale": "your rationale for the judgement, as a text",
  "judgement": "your judgement result, can only be 'correct' or 'incorrect'"
}}"""

    try:
        client = _get_judge_client(api_key, api_base)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a fair judge evaluating if two answers "
                        "to a question are equivalent."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=_judge_max_tokens(),
        )
        text = response.choices[0].message.content.strip()
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            import json_repair
            result = json_repair.loads(text)

        judgement = result.get("judgement", "").strip().lower()
        if judgement not in ("correct", "incorrect"):
            judgement = "incorrect"

        return {
            "question": question,
            "judgement": judgement,
            "golden_answer": golden_answer,
            "pred_answer": pred_answer,
        }
    except Exception as e:
        logger.error(f"Judge equivalence failed: {e}")
        return {
            "question": question,
            "judgement": "error",
            "golden_answer": golden_answer,
            "pred_answer": pred_answer,
        }


# ── WebwalkerQA judge ────────────────────────────────────────────────

def judge_webwalkerqa(
    question: str,
    golden_answer: str,
    pred_answer: Any,
    *,
    model: str = "qwen/qwen3.5-122b-a10b",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Dict[str, str]:
    """LLM judge for WebwalkerQA — focuses on core answer correctness."""
    if isinstance(pred_answer, dict):
        pred_answer = pred_answer.get("answer", str(pred_answer))
    pred_answer = str(pred_answer).strip() if pred_answer else ""

    if not pred_answer:
        return {
            "question": question,
            "judgement": "incorrect",
            "golden_answer": golden_answer,
            "pred_answer": pred_answer,
        }

    prompt = f"""You are a general AI assistant. Based on the [Correct Answer] provided below, determine whether the [Response] to the [Original Question] is correct.

[Original Question]: {question}

[Correct Answer]: {golden_answer}

[Response]: {pred_answer}

Your judgment must follow this standard:
- Focus only on whether there are substantial differences between the [Response] and the [Correct Answer]
- Do not comment on the background of the question
- Do not attempt to resolve the problem again
- Only focus on judging whether the answers are consistent
- If the [Response] is consistent with the [Correct Answer], or within an acceptable small margin of error for numerical questions, judge as "correct"
- Otherwise (i.e., in cases of any inconsistency, ambiguity, non-equivalence, or incorrectly extracted answer), judge as "incorrect"

Output JSON format:
{{
  "judgement": "correct" or "incorrect"
}}"""

    try:
        client = _get_judge_client(api_key, api_base)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a fair judge for web navigation tasks. "
                        "Focus on core answer correctness, not formatting."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            import json_repair
            result = json_repair.loads(text)

        judgement = result.get("judgement", "").strip().lower()
        if judgement not in ("correct", "incorrect"):
            judgement = "incorrect"

        return {
            "question": question,
            "judgement": judgement,
            "golden_answer": golden_answer,
            "pred_answer": pred_answer,
        }
    except Exception as e:
        logger.error(f"WebwalkerQA judge failed: {e}")
        return {
            "question": question,
            "judgement": "error",
            "golden_answer": golden_answer,
            "pred_answer": pred_answer,
        }


# ── XBench judge (Chinese) ──────────────────────────────────────────

XBENCH_JUDGE_PROMPT = """\
你是一个通用人工智能助手。根据下面给出的[正确答案], 判断以下对[原问题]的[回答]的回答是否正确。

[原问题]: {question}

[正确答案]: {correct_answer}

[回答]:{response}

你的判断必须按照以下格式和标准进行:

最终答案: 从[回答]中提取出的最终准确答案。如果[回答]中没有明确的最终答案, 则填写'无'。

解释: 根据[正确]解释为什么[最终答案]是正确的或错误的。只关注[最终答案]与[正确答案]之间是否存在实质性差异, 不要评论题目的背景, 不要尝试重新解题, 不要为任何不同于[正确答案]的答案辩护, 只专注于判断答案是否一致。

结论: 如果[最终答案]与上方给出的[正确答案]一致, 或者在数值题目中处于可接受的微小误差范围内, 则填写'正确'; 否则（即存在任何不一致、歧义、不等价或提取出的答案错误的情况）填写'错误'。"""


def judge_xbench(
    question: str,
    golden_answer: str,
    pred_answer: Any,
    *,
    model: str = "qwen/qwen3.5-122b-a10b",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Tuple[int, str, str]:
    """XBench-style Chinese judge.  Returns (score, extracted_answer, explanation).

    Phase 1: direct regex match → skip LLM call.
    Phase 2: LLM judge with Chinese prompt.
    """
    pred_str = str(pred_answer).strip() if pred_answer else ""
    if not pred_str:
        return 0, "", ""

    # Phase 1: direct match
    simple = re.search(r"最终答案[:：]\s*(.*)", pred_str)
    if simple:
        extracted = simple.group(1).strip()
        if extracted == golden_answer:
            return 1, extracted, "答案完全正确, 无需调用LLM Judge"

    # Phase 2: LLM judge
    prompt = XBENCH_JUDGE_PROMPT.format(
        question=question,
        correct_answer=golden_answer,
        response=pred_str,
    )
    try:
        client = _get_judge_client(api_key, api_base)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        judge_text = response.choices[0].message.content
        if not isinstance(judge_text, str):
            return 0, "", "Judge Response error"

        # Extract structured fields via regex
        ans_m = re.search(r"最终答案[:：]\s*(.*)", judge_text)
        extracted = ans_m.group(1).strip() if ans_m else ""

        conc_m = re.search(r"结论[:：]\s*(正确|错误)", judge_text)
        conclusion = conc_m.group(1).strip() if conc_m else ""

        expl_m = re.search(r"解释[:：]\s*(.*)", judge_text)
        explanation = expl_m.group(1).strip() if expl_m else ""

        score = 1 if conclusion == "正确" else 0
        return score, extracted, explanation
    except Exception as e:
        logger.error(f"XBench judge failed: {e}")
        return 0, "", f"Judge error: {e}"


# ── DeepSearchQA judge (set-based F1) ────────────────────────────────

def judge_deepsearchqa_item(
    problem: str,
    golden_answer: str,
    pred_answer: Any,
    answer_type: str = "Single Answer",
    *,
    model: str = "qwen/qwen3.5-122b-a10b",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Dict[str, Any]:
    """DeepSearchQA judge using LLM to match predicted items to gold items.

    For Single Answer: binary match (precision=recall=F1 = 0 or 1).
    For Set Answer: compute precision, recall, F1 over item sets.

    Returns dict with: precision, recall, f1, judgement, details.
    """
    pred_str = str(pred_answer).strip() if pred_answer else ""
    gold_str = str(golden_answer).strip() if golden_answer else ""

    if not pred_str:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "judgement": "incorrect",
            "details": "Empty prediction",
        }

    if answer_type == "Single Answer":
        # For single answer, ask LLM for equivalence
        eq_result = judge_equivalence(
            problem, gold_str, pred_str,
            model=model, api_key=api_key, api_base=api_base,
        )
        is_correct = eq_result.get("judgement") == "correct"
        score = 1.0 if is_correct else 0.0
        return {
            "precision": score,
            "recall": score,
            "f1": score,
            "judgement": eq_result.get("judgement", "incorrect"),
            "details": eq_result,
        }

    # Set Answer: parse both sides into items and match via LLM
    prompt = f"""You are evaluating a set-based answer for a research question.

## Question
{problem}

## Gold Answer (ground truth)
{gold_str}

## Predicted Answer
{pred_str}

## Task
1. Parse both the Gold Answer and Predicted Answer into individual items/elements.
2. For each predicted item, determine if it semantically matches any gold item.
3. For each gold item, determine if it was found in the prediction.

## Output
Return a JSON object with:
{{
  "gold_items": ["item1", "item2", ...],
  "pred_items": ["item1", "item2", ...],
  "matched_count": <number of predicted items that match a gold item>,
  "gold_count": <total gold items>,
  "pred_count": <total predicted items>
}}

Be generous with matching — items are equivalent if they refer to the same entity/value, even if worded differently. But do NOT match items that are genuinely different."""

    try:
        client = _get_judge_client(api_key, api_base)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise evaluator for set-based answers. "
                        "Parse and match items carefully."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=_judge_max_tokens(),
        )
        text = response.choices[0].message.content.strip()
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            import json_repair
            result = json_repair.loads(text)

        matched = result.get("matched_count", 0)
        gold_count = result.get("gold_count", 1)
        pred_count = result.get("pred_count", 1)

        gold_count = max(gold_count, 1)
        pred_count = max(pred_count, 1)

        precision = matched / pred_count
        recall = matched / gold_count
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        judgement = "correct" if f1 >= 1.0 else ("partially_correct" if f1 > 0 else "incorrect")

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "judgement": judgement,
            "details": result,
        }
    except Exception as e:
        logger.error(f"DeepSearchQA judge failed: {e}")
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "judgement": "error",
            "details": str(e),
        }


# ── MoNaCo judge (prompt + scoring aligned with monaco-main) ───────

MONACO_SINGLE_ANSWER_JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.
[question]: {question}
[response]: '{response}'

Your judgment must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as ’None’ if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer ’yes’ if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems, a margin of 1 to 3.5 percentage points is acceptable. Answer ’no’ otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

precision: Answer ’1’ if extracted_final_answer matches the [correct_answer] given above. Answer ’0’ otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect. In the case where [correct_answer] is a number or percentage, then answer with the following formula to compute the normalized similarity score: [1 - (abs([correct_answer] - extracted_final_answer) / max(abs([correct_answer]), abs(extracted_final_answer)))]

final precision: Extract the precision score from above, just the final score (number).
"""


MONACO_MULTI_ANSWER_JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.
[question]: {question}
[response]: '{response}'

Your judgment must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as ’None’ if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

final answer length: Provide the overall number of unique answers that appear in [response], not just the correct ones. Be sure to provide a number, not an estimate!

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer ’yes’ if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems, a margin of 1 to 5.5 percentage points is acceptable. Answer ’no’ otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

overlapping answers: List all of the answers in [response] that also appear in [correct_answer]. You can consider an answer from [response] to match with an answer in [correct_answer] if it is equivalent or is within a small margin of error for numerical problems, a margin of 1 to 5.5 percentage points is acceptable. List all of the [response] answer appearing in [correct_answer] with each answer delimited by '###'. If the number of overlapping answers is zero, output 'NULL'.
"""


def _parse_monaco_gold_answers(golden_answer: Any) -> list:
    """Parse MoNaCo gold answers from list/string formats."""
    if isinstance(golden_answer, list):
        return golden_answer

    if isinstance(golden_answer, str):
        text = golden_answer.strip()
        if not text:
            return []

        # Try JSON first
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except Exception:
            pass

        # Fall back to Python-literal list repr
        try:
            import ast

            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except Exception:
            return [text]

    return [golden_answer]


def _compute_monaco_llm_judge_score_v2(
    llm_judgement: str,
    gold_answers_length: int,
) -> Optional[Dict[str, Any]]:
    """Port of monaco-main/prompts/evaluate_final_answers.py::compute_llm_judge_score_V2."""

    def extract_single_answer_score() -> Dict[str, Any]:
        try:
            prec = float(
                llm_judgement.split("final precision:")[1]
                .split("\n")[0]
                .replace("...", "")
            )
        except Exception:
            prec = 0.0
        return {"judge_score": prec, "precision": prec}

    def extract_multi_answer_scores() -> Optional[Dict[str, Any]]:
        length_keyword = "\nfinal answer length:"
        if length_keyword not in llm_judgement:
            return None

        len_substr = llm_judgement.replace(
            "final answer length: None ",
            "final answer length: 0 ",
        )
        len_substr = len_substr.replace("The response lists over", "")
        len_substr = len_substr.split(length_keyword)[1].strip().split("\n")[0].strip()
        len_substr = len_substr.split(" ")[0].strip() if " " in len_substr else len_substr
        predicted_length = int(len_substr)

        correct_predicted_answers_keyword = "\noverlapping answers:"
        if correct_predicted_answers_keyword not in llm_judgement:
            return None

        answer_delimiter = "###"
        empty_answer_keyword = "NULL"
        answers_chunk = llm_judgement.split(correct_predicted_answers_keyword)[1].replace(
            answer_delimiter + empty_answer_keyword,
            answer_delimiter,
        ).strip()
        answers_chunk = answers_chunk[:-3] if answers_chunk.endswith(answer_delimiter) else answers_chunk
        answers = answers_chunk.split(answer_delimiter)
        num_correct = len(answers) if answers != [empty_answer_keyword] else 0

        if predicted_length == 0.0:
            recall = 0.0
            precision = 0.0
            f1 = 0.0
        else:
            recall = float(min(num_correct, gold_answers_length)) / gold_answers_length if num_correct != 0 else 0.0
            predicted_length = max(predicted_length, num_correct)
            precision = float(num_correct) / predicted_length if num_correct != 0 else 0.0
            f1 = (2 * (precision * recall)) / (precision + recall) if num_correct != 0 else 0.0

        return {
            "judge_score": f1,
            "precision": precision,
            "recall": recall,
            "gold answers length": gold_answers_length,
            "predicted answers num": predicted_length,
            "correct predictions": answers,
            "num correct": num_correct,
        }

    llm_judgement = llm_judgement.replace("final_answer_length", "final answer length")
    llm_judgement = llm_judgement.replace("overlapping_answers", "overlapping answers")
    llm_judgement = llm_judgement.replace("final_precision", "final precision")

    if gold_answers_length == 1:
        return extract_single_answer_score()
    if gold_answers_length > 1:
        return extract_multi_answer_scores()
    return None


def judge_monaco_item(
    question: str,
    golden_answer: Any,
    pred_answer: Any,
    *,
    model: str = "qwen/qwen3.5-122b-a10b",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Dict[str, Any]:
    """MoNaCo LLM-as-a-judge aligned with monaco-main prompt + parsing flow."""
    pred_str = str(pred_answer).strip() if pred_answer else ""
    if not pred_str:
        return {
            "judge_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "judgement": "incorrect",
            "llm_judgement": "",
        }

    gold_answers = _parse_monaco_gold_answers(golden_answer)
    gold_len = len(gold_answers)
    if gold_len <= 0:
        gold_answers = [golden_answer]
        gold_len = 1

    prompt_template = (
        MONACO_SINGLE_ANSWER_JUDGE_PROMPT
        if gold_len == 1
        else MONACO_MULTI_ANSWER_JUDGE_PROMPT
    )
    prompt = prompt_template.format(
        question=question,
        response=pred_str,
        correct_answer=gold_answers,
    )

    try:
        client = _get_judge_client(api_key, api_base)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        llm_judgement = response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"MoNaCo judge failed: {e}")
        return {
            "judge_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "judgement": "error",
            "llm_judgement": f"Judge error: {e}",
        }

    parsed = _compute_monaco_llm_judge_score_v2(llm_judgement, gold_len) or {}

    precision = float(parsed.get("precision", 0.0))
    # monaco-main single-answer flow does not expose recall/f1; keep aligned defaults
    if gold_len == 1:
        recall = precision
        f1 = precision
    else:
        recall = float(parsed.get("recall", 0.0))
        f1 = float(parsed.get("judge_score", 0.0))

    judge_score = float(parsed.get("judge_score", f1))
    judgement = "correct" if judge_score > 0 else "incorrect"

    return {
        "judge_score": judge_score,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "judgement": judgement,
        "llm_judgement": llm_judgement,
        "details": parsed,
    }
