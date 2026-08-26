"""
OpenAI-compatible API server model.
Adapted from Flash-Searcher FlashOAgents/models.py
"""

import json
import logging
import os
import threading
import time
from typing import Dict, List, Optional

from scripts.kernel.token_counter import count_tokens_messages, count_tokens_text
from .base import (
    ChatMessage, EmptyContentError, Model,
    parse_json_if_needed, parse_tool_args_if_needed,
)

logger = logging.getLogger(__name__)


# Scope gate for EXEC_NATIVE_TOOLCALL: attach the native tools schema only on turns
# whose prompt is actually asking for the {"think","tools"} action JSON. A harness
# makes two other kinds of ctx.model() call that want plain text -- planning (a
# numbered plan) and force-final (a {"think","answer"} wrap-up). Passing tools plus
# tool_choice=auto on the wrap-up call makes some models (glm-5.2) keep calling
# tools instead: the back-filled {"think","tools"} JSON is then handed to the judge
# verbatim as the final answer. Measured on one deepsearchqa case: 40 steps
# exhausted -> the wrap-up returns tool_calls -> scored 0, where the deepseek
# baseline on the same harness scored 1.0.
# The marker is matched in all three quoting styles. Audited over 100 generated
# prompt.yaml files from one deepsearchqa run: 99 step prompts contain one of them
# (the single exception falls back to the pure prompt protocol), and no final or
# planning prompt matches -- no false positives.
_TOOL_PROTOCOL_MARKERS = ('"tools"', "'tools'", "`tools`")


def _trailing_user_text(messages) -> str:
    """Text of the last user message (str or text-parts list)."""
    for msg in reversed(messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return ""
    return ""


class OpenAIServerModel(Model):
    """Connects to an OpenAI-compatible API server.

    Args:
        model_id: The model identifier (e.g. "gpt-4o").
        api_base: The base URL of the API server.
        api_key: The API key for authentication.
        organization: Optional organization.
        project: Optional project.
        temperature: Optional sampling temperature. If None, rely on the
            model provider's default behavior.
        custom_role_conversions: Custom role conversion mapping.
        **kwargs: Additional keyword arguments passed to the API.
    """
    _failure_lock = threading.Lock()
    _consecutive_api_failures = 0
    _max_consecutive_api_failures = max(
        1,
        int(os.getenv("MODULAR_AGENT_MAX_CONSECUTIVE_API_FAILURES", "5")),
    )

    def __init__(
        self,
        model_id: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        temperature: Optional[float] = None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        import openai

        if temperature is not None:
            kwargs["temperature"] = temperature

        super().__init__(**kwargs)
        self.model_id = model_id
        self.client = openai.OpenAI(
            base_url=api_base,
            api_key=api_key,
            organization=organization,
            project=project,
        )
        self.custom_role_conversions = custom_role_conversions

    @classmethod
    def _record_api_success(cls) -> None:
        with cls._failure_lock:
            if cls._consecutive_api_failures > 0:
                logger.info(
                    "Model API call recovered, resetting consecutive failure counter "
                    "(was %s).",
                    cls._consecutive_api_failures,
                )
            cls._consecutive_api_failures = 0

    @classmethod
    def _record_api_failure_and_maybe_exit(cls, error: Exception) -> None:
        with cls._failure_lock:
            cls._consecutive_api_failures += 1
            current_failures = cls._consecutive_api_failures
            threshold = cls._max_consecutive_api_failures

        logger.error(
            "Model API call failed (%s/%s consecutive failures): %s",
            current_failures,
            threshold,
            error,
        )
        if current_failures >= threshold:
            logger.critical(
                "Consecutive model API failures reached %s, terminating process.",
                threshold,
            )
            raise SystemExit(1)

    @staticmethod
    def truncate_content_based_on_stop_sequences(content: str, stop_sequences: List[str]) -> str:
        if not stop_sequences:
            return content
        for stop_seq in stop_sequences:
            index = content.find(stop_seq)
            if index != -1:
                content = content[:index + len(stop_seq)]
                break
        return content

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from=None,
        **kwargs,
    ) -> ChatMessage:
        from openai import (
            BadRequestError, APIStatusError, APIConnectionError, OpenAIError,
        )

        # EXEC_NATIVE_TOOLCALL=1: use OpenAI native function calling instead of the
        # JSON protocol agreed in the prompt. Why -- a harness only reads the
        # {"think","tools"} text out of `content`, and some models (glm-5.2) cannot
        # hold to that convention over a long multi-turn run: without a reminder,
        # compliance is only 14-34%. Their native tool calls, however, work fine. With
        # this on, the request carries the tools schema and the response's tool_calls
        # are back-filled into the JSON the harness expects (below), so neither
        # harness code nor prompts have to change. Off by default.
        native = (os.environ.get("EXEC_NATIVE_TOOLCALL") == "1"
                  and tools_to_call_from is None
                  and getattr(self, "_native_tool_registry", None))
        if native:
            trailing = _trailing_user_text(messages)
            if not any(m in trailing for m in _TOOL_PROTOCOL_MARKERS):
                native = False
        if native:
            tools_to_call_from = list(self._native_tool_registry.values())

        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )

        # Most harnesses drive tools through a prompt-level JSON protocol (the system
        # prompt asks for {"think":..., "tools":[...]}) rather than native function
        # calling. Some models drift off that convention as the context grows and start
        # answering in prose; the harness then parses no tools, which shows up as
        # "Tool calls: 0" -- a step that does nothing. Measured with glm-5.2 on
        # officebench, JSON compliance fell from 83% at step 1 to 5% from step 6 on
        # (deepseek stays above 85% throughout); re-anchoring with this reminder
        # restored 3/3. Off by default: deepseek does not need it, and the control arm
        # must stay byte-identical.
        if os.environ.get("EXEC_FORMAT_REMINDER") == "1" and not tools_to_call_from:
            msgs = completion_kwargs.get("messages")
            if isinstance(msgs, list) and msgs:
                # The wording must refer to the FIELD NAMES, never to a section
                # heading like "Action Format": on travel the harness prompt is
                # written in Chinese and contains no such heading, so the old
                # heading-based wording had no effect there at all (JSON stayed
                # False). Naming the fields fixed it immediately.
                msgs.append({
                    "role": "user",
                    "content": (
                        "Reply ONLY with the single JSON object required by the system prompt "
                        '(the object containing the "think" and "tools" fields). '
                        "No prose, no markdown, no explanation outside the JSON."
                    ),
                })

        # When base.py attaches tools it also hardcodes tool_choice="required" (every
        # step must call a tool). That is harmful for these ReAct harnesses: a model
        # with nothing left to look up is forced to pick something, execute_code is the
        # most general escape hatch, and the run degenerates into dozens of consecutive
        # execute_code steps that never reach final_answer (measured on one travel
        # case: 36 consecutive execute_code steps out of 80, final_answer never called,
        # while in prompt mode the tool distribution matches deepseek). Use auto, which
        # hands the decision of when to wrap up back to the model.
        if native:
            completion_kwargs["tool_choice"] = "auto"

        # Handle o3/o4 models that don't support stop sequences
        if 'o3' in self.model_id.lower() or 'o4' in self.model_id.lower():
            completion_kwargs.pop('stop', None)

        max_retries = 5
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**completion_kwargs)

                usage = getattr(response, "usage", None)
                input_tokens = None
                output_tokens = None
                estimated_tokens = False

                if usage is not None:
                    input_tokens = (
                        getattr(usage, "prompt_tokens", None)
                        if not isinstance(usage, dict)
                        else usage.get("prompt_tokens")
                    )
                    output_tokens = (
                        getattr(usage, "completion_tokens", None)
                        if not isinstance(usage, dict)
                        else usage.get("completion_tokens")
                    )

                if input_tokens is None:
                    input_tokens = count_tokens_messages(completion_kwargs.get("messages", []) or [])
                    estimated_tokens = True

                raw_message = response.choices[0].message
                if (
                    not raw_message.content
                    and not getattr(raw_message, 'tool_calls', None)
                    and not getattr(raw_message, 'reasoning', None)
                    and not getattr(raw_message, 'reasoning_content', None)
                ):
                    raise EmptyContentError(response)

                if output_tokens is None:
                    out_text = response.choices[0].message.content or ""
                    output_tokens = count_tokens_text(out_text)
                    estimated_tokens = True

                self._record_token_usage(
                    input_token_count=input_tokens,
                    output_token_count=output_tokens,
                    estimated=estimated_tokens,
                )

                message_dump = response.choices[0].message.model_dump(
                    include={"role", "content", "tool_calls", "reasoning_content", "reasoning"}
                )
                # vLLM >= ~0.22 returns the reasoning-parser split under
                # "reasoning" (OpenAI style); older versions used
                # "reasoning_content" (DeepSeek style). Normalize onto the
                # ChatMessage.reasoning_content field so downstream stitching
                # (_full_response_text) sees the pre-</think> half either way.
                if not message_dump.get("reasoning_content"):
                    message_dump["reasoning_content"] = message_dump.pop("reasoning", None)
                else:
                    message_dump.pop("reasoning", None)
                message = ChatMessage.from_dict(message_dump)
                message.raw = response

                if 'o3' in self.model_id.lower() or 'o4' in self.model_id.lower():
                    message.content = self.truncate_content_based_on_stop_sequences(message.content, stop_sequences)

                # Native mode: a harness only parses the {"think","tools"} text out
                # of `content` -- it never looks at message.tool_calls. So back-fill
                # the native tool_calls into that JSON and write it to `content`,
                # taking "think" from the model's own reasoning_content (which is
                # exactly the same notion). Prompts and harness code stay untouched
                # while the request travels the path the model handles best.
                if native and message.tool_calls:
                    payload = {
                        "think": (message.reasoning_content or "").strip(),
                        "tools": [
                            {
                                "name": tc.function.name,
                                "arguments": parse_json_if_needed(tc.function.arguments),
                            }
                            for tc in message.tool_calls
                        ],
                    }
                    message.content = json.dumps(payload, ensure_ascii=False)

                if tools_to_call_from is not None:
                    self._record_api_success()
                    return parse_tool_args_if_needed(message)
                self._record_api_success()
                return message

            except BadRequestError as e:
                logger.error(f"Bad Request Error: {e}")
                self._record_api_failure_and_maybe_exit(e)
                raise
            except APIConnectionError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Network error: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed after {max_retries} retries.")
                    self._record_api_failure_and_maybe_exit(e)
                    raise
            except (APIStatusError, EmptyContentError) as e:
                # Distinguish retriable from non-retriable status codes
                status_code = getattr(e, "status_code", None)
                retriable_codes = {429, 500, 502, 503, 504}

                if status_code is not None and status_code not in retriable_codes:
                    # Non-retriable errors (401 auth, 403 forbidden, etc.)
                    logger.error(
                        f"Non-retriable API error (HTTP {status_code}): {e}"
                    )
                    self._record_api_failure_and_maybe_exit(e)
                    raise

                if attempt < max_retries - 1:
                    # Use shorter delay for rate-limits, longer for server errors
                    delay = 10 if status_code == 429 else 30
                    logger.warning(
                        f"API status error (HTTP {status_code}): {e}. "
                        f"Retrying in {delay}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Failed after {max_retries} retries.")
                    self._record_api_failure_and_maybe_exit(e)
                    raise
            except json.JSONDecodeError as e:
                # The provider returned a body that is not valid JSON, so
                # httpx's response.json() blows up inside the SDK before any
                # OpenAI exception type is constructed -- it is a truncated or
                # garbled response from the router, not a problem with the
                # request. Seen on OpenRouter :floor routing for
                # qwen/qwen3.5-397b-a17b ("Expecting value: line 267 column 1").
                # It used to fall through to the bare `except Exception` below,
                # which re-raises with no retry at all, so a one-off transport
                # glitch killed the whole case (status="error", score 0) and
                # burned a full wrapper attempt to recover it.
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Malformed JSON in API response: {e}. "
                        f"Retrying in {retry_delay}s "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed after {max_retries} retries: {e}")
                    self._record_api_failure_and_maybe_exit(e)
                    raise
            except OpenAIError as e:
                logger.error(f"API error: {e}.")
                self._record_api_failure_and_maybe_exit(e)
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}.")
                raise
