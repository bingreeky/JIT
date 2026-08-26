"""
Model base classes and utilities.
Adapted from Flash-Searcher FlashOAgents/models.py
"""

import json
import logging
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class EmptyContentError(Exception):
    def __init__(self, response):
        self.response = response
        super().__init__(f"Empty content in response: {response}")


def get_dict_from_nested_dataclasses(obj, ignore_key=None):
    def convert(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(obj).items() if k != ignore_key}
        return obj
    return convert(obj)


@dataclass
class ChatMessageToolCallDefinition:
    arguments: Any
    name: str
    description: Optional[str] = None

    @classmethod
    def from_hf_api(cls, tool_call_definition) -> "ChatMessageToolCallDefinition":
        return cls(
            arguments=tool_call_definition.arguments,
            name=tool_call_definition.name,
            description=tool_call_definition.description,
        )


@dataclass
class ChatMessageToolCall:
    function: ChatMessageToolCallDefinition
    id: str
    type: str

    @classmethod
    def from_hf_api(cls, tool_call) -> "ChatMessageToolCall":
        return cls(
            function=ChatMessageToolCallDefinition.from_hf_api(tool_call.function),
            id=tool_call.id,
            type=tool_call.type,
        )


@dataclass
class ChatMessage:
    role: str
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ChatMessageToolCall]] = None
    raw: Optional[Any] = None

    def model_dump_json(self):
        return json.dumps(get_dict_from_nested_dataclasses(self, ignore_key="raw"))

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        if data.get("tool_calls"):
            tool_calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallDefinition(
                        **{k: v for k, v in tc["function"].items() if k != "parameters"}
                    ),
                    id=tc["id"],
                    type=tc["type"]
                )
                for tc in data["tool_calls"]
            ]
            data["tool_calls"] = tool_calls
        return cls(**data)

    def dict(self):
        return json.dumps(get_dict_from_nested_dataclasses(self))


def parse_json_if_needed(arguments: Union[str, dict]) -> Union[str, dict]:
    if isinstance(arguments, dict):
        return arguments
    try:
        return json.loads(arguments)
    except Exception:
        return arguments


def parse_tool_args_if_needed(message: ChatMessage) -> ChatMessage:
    if message.tool_calls is not None:
        for tool_call in message.tool_calls:
            tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
    return message


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool-call"
    TOOL_RESPONSE = "tool-response"

    @classmethod
    def roles(cls):
        return [r.value for r in cls]


tool_role_conversions = {
    MessageRole.TOOL_CALL: MessageRole.ASSISTANT,
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}


def get_tool_json_schema(tool) -> Dict:
    properties = deepcopy(tool.inputs)
    required = []
    for key, value in properties.items():
        if value["type"] == "any":
            value["type"] = "string"
        if not ("nullable" in value and value["nullable"]):
            required.append(key)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def remove_stop_sequences(content: str, stop_sequences: List[str]) -> str:
    for stop_seq in stop_sequences:
        if content[-len(stop_seq):] == stop_seq:
            content = content[:-len(stop_seq)]
    return content


def get_clean_message_list(
    message_list: List[Dict[str, str]],
    role_conversions: Dict = {},
    convert_images_to_image_urls: bool = False,
    flatten_messages_as_text: bool = False,
) -> List[Dict[str, str]]:
    """
    Subsequent messages with the same role will be concatenated to a single message.
    """
    output_message_list = []
    message_list = deepcopy(message_list)
    for message in message_list:
        role = message["role"]
        if role not in MessageRole.roles():
            raise ValueError(f"Incorrect role {role}, only {MessageRole.roles()} are supported.")

        if role in role_conversions:
            message["role"] = role_conversions[role]

        if isinstance(message["content"], list):
            for element in message["content"]:
                if element.get("type") == "image":
                    assert not flatten_messages_as_text
                    if convert_images_to_image_urls:
                        from .utils import encode_image_base64, make_image_url
                        element.update({
                            "type": "image_url",
                            "image_url": {"url": make_image_url(encode_image_base64(element.pop("image")))},
                        })

        if len(output_message_list) > 0 and message["role"] == output_message_list[-1]["role"]:
            assert isinstance(message["content"], list), "Error: wrong content:" + str(message["content"])
            if flatten_messages_as_text:
                output_message_list[-1]["content"] += message["content"][0]["text"]
            else:
                output_message_list[-1]["content"] += message["content"]
        else:
            if flatten_messages_as_text:
                content = message["content"][0]["text"]
            else:
                content = message["content"]
            output_message_list.append({"role": message["role"], "content": content})
    return output_message_list


class Model:
    def __init__(self, **kwargs):
        self.last_input_token_count = None
        self.last_output_token_count = None
        self.total_input_token_count = 0
        self.total_output_token_count = 0
        self.kwargs = kwargs

    @staticmethod
    def _normalize_token_count(value: Any) -> Optional[int]:
        try:
            iv = int(value)
            return iv if iv >= 0 else None
        except Exception:
            return None

    def _record_token_usage(
        self,
        input_token_count: Any,
        output_token_count: Any,
        estimated: bool = False,
    ) -> None:
        input_count = self._normalize_token_count(input_token_count)
        output_count = self._normalize_token_count(output_token_count)

        self.last_input_token_count = input_count
        self.last_output_token_count = output_count

        if input_count is not None:
            self.total_input_token_count += input_count
        if output_count is not None:
            self.total_output_token_count += output_count

    def reset_token_counters(self) -> None:
        self.last_input_token_count = None
        self.last_output_token_count = None
        self.total_input_token_count = 0
        self.total_output_token_count = 0

    def _prepare_completion_kwargs(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from=None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        convert_images_to_image_urls: bool = False,
        flatten_messages_as_text: bool = False,
        **kwargs,
    ) -> Dict:
        messages = get_clean_message_list(
            messages,
            role_conversions=custom_role_conversions or tool_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            flatten_messages_as_text=flatten_messages_as_text,
        )

        completion_kwargs = {
            **self.kwargs,
            "messages": messages,
        }

        if stop_sequences is not None:
            completion_kwargs["stop"] = stop_sequences
        if grammar is not None:
            completion_kwargs["grammar"] = grammar

        if tools_to_call_from:
            completion_kwargs.update({
                "tools": [get_tool_json_schema(tool) for tool in tools_to_call_from],
                "tool_choice": "required",
            })

        completion_kwargs.update(kwargs)
        return completion_kwargs

    def get_token_counts(self) -> Dict[str, int]:
        return {
            "input_token_count": self.last_input_token_count,
            "output_token_count": self.last_output_token_count,
        }

    def get_total_token_counts(self) -> Dict[str, int]:
        return {
            "input_token_count": self.total_input_token_count,
            "output_token_count": self.total_output_token_count,
        }

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from=None,
        **kwargs,
    ) -> ChatMessage:
        pass  # Implemented in subclasses


__all__ = [
    "MessageRole", "tool_role_conversions", "get_clean_message_list",
    "Model", "ChatMessage", "ChatMessageToolCall", "ChatMessageToolCallDefinition",
    "get_tool_json_schema", "EmptyContentError",
]
