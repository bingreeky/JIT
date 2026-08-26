"""Best-of-N selection: let the model pick its own favourite harness.

The meta model generates N harnesses per case at temperature 1; exactly one of
them gets executed. Choosing it is a pure model judgement -- no benchmark score
is available at selection time, and using one would be leakage.

Two strategies, both over an OpenAI-compatible endpoint:

``logprob``  Teacher-force each candidate through the meta model and rank by the
             summed log-probability of its completion tokens. It needs the
             ``prompt_logprobs`` extension
             (vLLM >= 0.9 / SGLang) on ``/v1/completions`` plus the model's
             tokenizer, so it only works against a self-hosted meta model.

``judge``    Show the task and the N candidate designs to a model and ask it to
             pick the best, as strict JSON. Plain ``/v1/chat/completions``, so it
             works against any provider -- including hosted APIs where the
             logprob route is unavailable.

``auto`` (default) probes for ``prompt_logprobs`` support and falls back to
``judge``, printing which route it took.

Both strategies first apply the same free static gate: prefer candidates that
actually emitted all five harness blocks (``n_sub == 5``).
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

N_EXPECTED_BLOCKS = 5


# --------------------------------------------------------------------------- #
# Shared: turn per-candidate scores into a per-case pick
# --------------------------------------------------------------------------- #
def _pick(
    rows: List[Dict[str, Any]],
    score_key: str,
    rule_name: str,
    n_expected: int = N_EXPECTED_BLOCKS,
) -> Dict[str, Any]:
    """Per case: prefer complete candidates, then argmax `score_key`.

    A case with no scorable candidate falls back to rollout 0 rather than being
    dropped -- an unselected case would silently shrink the eval set.
    """
    by_case: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    all_cases = set()
    for row in rows:
        all_cases.add(row["case"])
        if score_key in row:
            by_case[row["case"]][row["rollout"]] = row

    selected: Dict[str, int] = {}
    detail: Dict[str, Any] = {}
    n_filtered = n_fallback = 0

    for case in sorted(all_cases):
        candidates = by_case.get(case, {})
        if not candidates:
            selected[case] = 0
            detail[case] = {"rule": "fallback_rollout0"}
            n_fallback += 1
            continue
        complete = {k: v for k, v in candidates.items() if v.get("n_sub") == n_expected}
        pool = complete or candidates
        if complete and len(complete) < len(candidates):
            n_filtered += 1
        pick = max(pool, key=lambda k: pool[k][score_key])
        selected[case] = pick
        detail[case] = {
            "rule": rule_name + ("_complete_only" if complete else "_any"),
            "pick": pick,
            score_key: {str(k): round(float(v[score_key]), 3) for k, v in candidates.items()},
            "n_sub": {str(k): v.get("n_sub") for k, v in candidates.items()},
        }

    distribution = defaultdict(int)
    for rollout in selected.values():
        distribution[rollout] += 1
    print(
        f"[select] {len(selected)} cases; pick distribution "
        f"{dict(sorted(distribution.items()))}; incomplete-filtered on {n_filtered} "
        f"cases; {n_fallback} fallbacks",
        flush=True,
    )
    return {"selected": selected, "detail": detail}


# --------------------------------------------------------------------------- #
# Strategy 1: policy log-probability
# --------------------------------------------------------------------------- #
def supports_prompt_logprobs(api_base: str, model: str, api_key: str = "EMPTY") -> bool:
    """Probe whether the endpoint accepts the ``prompt_logprobs`` extension."""
    try:
        response = requests.post(
            f"{api_base.rstrip('/')}/completions",
            headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
            json={
                "model": model,
                "prompt": "probe",
                "max_tokens": 1,
                "temperature": 0.0,
                "prompt_logprobs": 0,
            },
            timeout=60,
        )
        if not response.ok:
            return False
        return response.json()["choices"][0].get("prompt_logprobs") is not None
    except Exception:  # noqa: BLE001 - any failure means "use the other route"
        return False


def _tokenize_candidates(
    rows: List[Dict[str, Any]], tokenizer_dir: str, max_model_len: int
) -> List[Tuple[int, int, List[int]]]:
    """Render prompt + completion the way training did, and tokenize.

    Both texts are produced with the chat template, split at the CHARACTER
    level, then tokenized separately and concatenated -- this reproduces TRL's
    DPO tokenization, including its non-canonical newline handling at the
    boundary, so the log-probabilities are comparable to training-time ones.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

    jobs: List[Tuple[int, int, List[int]]] = []
    for i, row in enumerate(rows):
        if "completion" not in row:
            continue
        messages = [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["user"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        full_text = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": row["completion"]}], tokenize=False
        )
        if not full_text.startswith(prompt_text):
            # Template quirk: fall back to the longest common prefix.
            k = 0
            while k < min(len(prompt_text), len(full_text)) and prompt_text[k] == full_text[k]:
                k += 1
            prompt_text = prompt_text[:k]
            row["template_prefix_mismatch"] = True

        prefix = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        completion = tokenizer(full_text[len(prompt_text) :], add_special_tokens=False)["input_ids"]
        full = list(prefix) + list(completion)
        if len(full) > max_model_len - 8:
            row["skipped_too_long"] = len(full)
            continue
        row["n_prefix_tokens"] = len(prefix)
        row["n_completion_tokens"] = len(completion)
        jobs.append((i, len(prefix), full))
    return jobs


def _wait_ready(api_base: str, api_key: str, timeout_s: int = 1800) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if requests.get(
                f"{api_base.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
                timeout=10,
            ).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(15)
    raise RuntimeError(f"endpoint {api_base} not ready within {timeout_s}s")


def _score_one(
    api_base: str, model: str, api_key: str, prefix_len: int, full_ids: List[int], retries: int = 6
) -> float:
    """Sum of the completion tokens' log-probabilities, via prompt_logprobs."""
    body = {
        "model": model,
        "prompt": full_ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": 0,
        "stream": False,
    }
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            response = requests.post(
                f"{api_base.rstrip('/')}/completions",
                headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
                json=body,
                timeout=1800,
            )
            if response.status_code >= 500:
                raise requests.RequestException(f"HTTP {response.status_code}: {response.text[:300]}")
            response.raise_for_status()
            logprobs = response.json()["choices"][0]["prompt_logprobs"]
            if logprobs is None or len(logprobs) != len(full_ids):
                raise ValueError(
                    f"prompt_logprobs length {len(logprobs) if logprobs else None} "
                    f"!= {len(full_ids)}"
                )
            return sum(
                logprobs[pos][str(full_ids[pos])]["logprob"]
                for pos in range(prefix_len, len(full_ids))
            )
        except (requests.RequestException, KeyError, ValueError) as exc:  # noqa: PERF203
            last_error = exc
            if attempt < retries - 1:
                # A self-hosted server may be reloading behind a keepalive.
                _wait_ready(api_base, api_key)
                time.sleep(10)
    raise RuntimeError(f"scoring failed after {retries} attempts: {last_error}")


def select_by_logprob(
    rows: List[Dict[str, Any]],
    api_base: str,
    model: str,
    api_key: str,
    tokenizer_dir: str,
    max_model_len: int = 163840,
    concurrency: int = 3,
) -> Dict[str, Any]:
    """Rank candidates by the meta model's own sequence log-probability."""
    jobs = _tokenize_candidates(rows, tokenizer_dir, max_model_len)
    print(
        f"[select] logprob: scoring {len(jobs)}/{len(rows)} candidates via {api_base} "
        f"(model={model}, concurrency={concurrency})",
        flush=True,
    )
    _wait_ready(api_base, api_key)

    started = time.time()
    with cf.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {
            pool.submit(_score_one, api_base, model, api_key, prefix_len, full_ids): i
            for (i, prefix_len, full_ids) in jobs
        }
        for done, future in enumerate(cf.as_completed(futures), start=1):
            i = futures[future]
            try:
                rows[i]["sum_logprob"] = future.result()
                print(
                    f"[select] {done}/{len(jobs)} case={rows[i]['case']} "
                    f"r{rows[i]['rollout']} lp={rows[i]['sum_logprob']:.1f} "
                    f"({time.time() - started:.0f}s)",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                rows[i]["score_error"] = str(exc)[:300]
                print(
                    f"[select] {done}/{len(jobs)} case={rows[i]['case']} "
                    f"r{rows[i]['rollout']} FAILED: {rows[i]['score_error']}",
                    flush=True,
                )

    return _pick(rows, "sum_logprob", "argmax_logprob")


# --------------------------------------------------------------------------- #
# Strategy 2: model-as-judge
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM_PROMPT = """\
You are reviewing candidate agent harnesses. A harness is a small agent \
framework made of five files -- memory.py, planning.py, action.py, \
tool_policy.py and prompt.yaml -- that will be executed as-is to solve the task \
below.

Exactly one candidate will be run. Pick the one most likely to actually solve \
the task, judging on:
  1. correctness and robustness of the control flow (does it terminate, does it
     recover from a failed tool call, can it reach a final answer);
  2. fit to the specific task and to the tools that are available;
  3. soundness of the memory/planning design for this task's horizon;
  4. absence of obvious bugs, truncation or missing pieces.

Prefer a simple design that will run over an ambitious one that will not.

Reply with STRICT JSON and nothing else:
{"best": <candidate index>, "reason": "<one sentence>"}
"""


def _render_candidate(row: Dict[str, Any], index: int, char_budget: int) -> str:
    """Compact view of one candidate for the judge prompt."""
    completion = str(row.get("completion", ""))
    if len(completion) > char_budget:
        head = completion[: int(char_budget * 0.7)]
        tail = completion[-int(char_budget * 0.3) :]
        completion = f"{head}\n\n... [{len(completion) - char_budget} chars elided] ...\n\n{tail}"
    flags = []
    if row.get("n_sub") != N_EXPECTED_BLOCKS:
        flags.append(f"INCOMPLETE: only {row.get('n_sub')}/{N_EXPECTED_BLOCKS} files emitted")
    header = f"### Candidate {index}" + (f"  [{'; '.join(flags)}]" if flags else "")
    return f"{header}\n{completion}"


def _parse_choice(text: str, n_candidates: int) -> Optional[int]:
    """Pull ``best`` out of the judge reply, tolerating fenced or chatty JSON."""
    match = re.search(r"\{[^{}]*\"best\"[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            value = int(json.loads(match.group(0))["best"])
            if 0 <= value < n_candidates:
                return value
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    fallback = re.search(r"\b(?:best|candidate)\D{0,10}(\d+)", text, re.IGNORECASE)
    if fallback:
        value = int(fallback.group(1))
        if 0 <= value < n_candidates:
            return value
    return None


def _judge_case(
    case: str,
    candidates: List[Dict[str, Any]],
    model,
    char_budget: int,
) -> Dict[str, Any]:
    """Ask the judge model which candidate to execute for one case."""
    from scripts.models.base import MessageRole

    # `user` is the generation prompt, which already contains the task spec and
    # the tool catalogue -- reuse it rather than re-deriving the task.
    task_prompt = str(candidates[0].get("user", ""))[:20000]
    body = "\n\n".join(
        _render_candidate(row, i, char_budget) for i, row in enumerate(candidates)
    )
    messages = [
        {"role": MessageRole.SYSTEM, "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": MessageRole.USER,
            "content": (
                f"## Task the harness must solve\n{task_prompt}\n\n"
                f"## Candidates ({len(candidates)})\n{body}\n\n"
                f'Reply with {{"best": <0..{len(candidates) - 1}>, "reason": "..."}}'
            ),
        },
    ]
    reply = model(messages)
    text = str(getattr(reply, "content", reply) or "")
    choice = _parse_choice(text, len(candidates))
    return {
        "case": case,
        "choice": choice,
        "raw": text[:500],
    }


def select_by_judge(
    rows: List[Dict[str, Any]],
    model_id: str,
    api_base: str,
    api_key: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    char_budget: int = 24000,
    concurrency: int = 4,
) -> Dict[str, Any]:
    """Rank candidates by asking a model which one it would run."""
    from scripts.models.openai_server import OpenAIServerModel

    model = OpenAIServerModel(
        model_id=model_id,
        api_base=api_base,
        api_key=api_key or "EMPTY",
        temperature=temperature,
        max_tokens=max_tokens,
    )

    by_case: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    all_cases = set()
    for row in rows:
        all_cases.add(row["case"])
        if "completion" in row:
            by_case[row["case"]].append(row)
    for candidates in by_case.values():
        candidates.sort(key=lambda r: r["rollout"])

    print(
        f"[select] judge: {len(by_case)}/{len(all_cases)} cases have candidates "
        f"(model={model_id}, concurrency={concurrency})",
        flush=True,
    )

    verdicts: Dict[str, Dict[str, Any]] = {}
    with cf.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {
            pool.submit(_judge_case, case, candidates, model, char_budget): case
            for case, candidates in by_case.items()
        }
        for done, future in enumerate(cf.as_completed(futures), start=1):
            case = futures[future]
            try:
                verdicts[case] = future.result()
            except Exception as exc:  # noqa: BLE001
                verdicts[case] = {"case": case, "choice": None, "raw": f"ERROR: {exc}"[:500]}
            print(f"[select] judged {done}/{len(by_case)} case={case}", flush=True)

    # Turn the judge's pick into the same score shape `_pick` consumes, so the
    # complete-candidates gate and the fallback rule stay identical.
    #
    # A case whose verdict could not be parsed is left UNSCORED on purpose:
    # `_pick` then reports it as `fallback_rollout0` instead of dressing an
    # arbitrary pick up as a judgement. The fallback count in the log is how
    # you notice a judge model that cannot follow the output format.
    unparsed = 0
    for case, candidates in by_case.items():
        choice = verdicts.get(case, {}).get("choice")
        if choice is None:
            unparsed += 1
            continue
        for position, row in enumerate(candidates):
            row["judge_score"] = 1.0 if position == choice else 0.0
            row["judge_reason"] = verdicts.get(case, {}).get("raw", "")
    if unparsed:
        print(
            f"[select] WARNING: {unparsed}/{len(by_case)} judge verdicts were "
            f"unparseable; those cases fall back to rollout 0",
            flush=True,
        )

    result = _pick(rows, "judge_score", "judge_pick")
    result["judge_verdicts"] = verdicts
    return result


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def select(
    rows: List[Dict[str, Any]],
    *,
    strategy: str,
    api_base: str,
    model: str,
    api_key: str,
    tokenizer_dir: str = "",
    judge_model: str = "",
    judge_base: str = "",
    judge_key: str = "",
    max_model_len: int = 163840,
    concurrency: int = 3,
) -> Dict[str, Any]:
    """Run the requested selection strategy, resolving ``auto``."""
    chosen = strategy
    if strategy == "auto":
        can_logprob = bool(tokenizer_dir) and supports_prompt_logprobs(api_base, model, api_key)
        chosen = "logprob" if can_logprob else "judge"
        reason = (
            "endpoint supports prompt_logprobs and a tokenizer is available"
            if can_logprob
            else (
                "no tokenizer given" if not tokenizer_dir
                else "endpoint does not support prompt_logprobs"
            )
        )
        print(f"[select] strategy=auto -> {chosen} ({reason})", flush=True)

    if chosen == "logprob":
        if not tokenizer_dir:
            raise SystemExit(
                "--selector logprob needs --tokenizer (a local model dir or a HF repo id) "
                "to reproduce the training-time tokenization. Use --selector judge for a "
                "tokenizer-free, provider-agnostic alternative."
            )
        result = select_by_logprob(
            rows, api_base, model, api_key, tokenizer_dir, max_model_len, concurrency
        )
    elif chosen == "judge":
        result = select_by_judge(
            rows,
            judge_model or model,
            judge_base or api_base,
            judge_key or api_key,
            concurrency=concurrency,
        )
    else:
        raise SystemExit(f"unknown selector strategy '{strategy}'")

    result["strategy"] = chosen
    return result


def load_selection(path: str) -> Dict[str, int]:
    """Read a selection file written by this module (or a bare ``{qid: r}`` map)."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return {str(k): int(v) for k, v in (payload.get("selected", payload)).items()}


def write_candidate_records(rows: Sequence[Dict[str, Any]], path: str) -> None:
    """Persist the per-candidate scoring record (without the bulky completions)."""
    keep = (
        "case",
        "rollout",
        "n_sub",
        "final_stage",
        "nonempty_think",
        "error",
        "n_prefix_tokens",
        "n_completion_tokens",
        "skipped_too_long",
        "template_prefix_mismatch",
        "score_error",
        "sum_logprob",
        "judge_score",
    )
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(
                json.dumps({k: row[k] for k in keep if k in row}, ensure_ascii=False) + "\n"
            )
