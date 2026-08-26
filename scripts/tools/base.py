"""
Tool base class and FinalAnswerTool.
Adapted from Flash-Searcher FlashOAgents/tools.py
"""

import inspect
import logging
import time
from functools import wraps
from typing import Dict, Union, Any

logger = logging.getLogger(__name__)


def validate_after_init(cls):
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.validate_arguments()

    cls.__init__ = new_init
    return cls


AUTHORIZED_TYPES = [
    "dict", "string", "boolean", "integer", "number",
    "image", "audio", "array", "object", "any", "null", "Tuple",
]

CONVERSION_DICT = {"str": "string", "int": "integer", "float": "number"}


class Tool:
    """Base class for agent tools.

    Subclass this and implement the `forward` method.
    Set class attributes: name, description, inputs, output_type.
    """

    name: str
    description: str
    inputs: Dict[str, Dict[str, Union[str, type, bool]]]
    output_type: str
    _total_latency_sec: float = 0.0

    def __init__(self, *args, **kwargs):
        self.is_initialized = False
        self.total_latency_sec = 0.0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        validate_after_init(cls)

    def validate_arguments(self):
        required_attributes = {
            "description": str,
            "name": str,
            "inputs": dict,
            "output_type": str,
        }
        for attr, expected_type in required_attributes.items():
            attr_value = getattr(self, attr, None)
            if attr_value is None:
                raise TypeError(f"You must set an attribute {attr}.")
            if not isinstance(attr_value, expected_type):
                raise TypeError(
                    f"Attribute {attr} should have type {expected_type.__name__}, got {type(attr_value)} instead."
                )
        for input_name, input_content in self.inputs.items():
            assert isinstance(input_content, dict), f"Input '{input_name}' should be a dictionary."
            assert "type" in input_content and "description" in input_content, (
                f"Input '{input_name}' should have keys 'type' and 'description'."
            )
            if input_content["type"] not in AUTHORIZED_TYPES:
                raise Exception(
                    f"Input '{input_name}': type '{input_content['type']}' is not authorized, "
                    f"should be one of {AUTHORIZED_TYPES}."
                )
        assert getattr(self, "output_type", None) in AUTHORIZED_TYPES

    def forward(self, *args, **kwargs):
        return NotImplementedError("Write this method in your subclass of `Tool`.")

    def __call__(self, *args, sanitize_inputs_outputs: bool = False, **kwargs):
        if not self.is_initialized:
            self.setup()

        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            potential_kwargs = args[0]
            if all(key in self.inputs for key in potential_kwargs):
                args = ()
                kwargs = potential_kwargs

        started = time.perf_counter()
        outputs = self.forward(*args, **kwargs)
        elapsed = time.perf_counter() - started
        self.total_latency_sec += elapsed
        Tool._total_latency_sec += elapsed
        return outputs

    def setup(self):
        """Override for expensive initialization (e.g., loading a model)."""
        self.is_initialized = True

    @classmethod
    def reset_total_latency(cls) -> None:
        Tool._total_latency_sec = 0.0

    @classmethod
    def get_total_latency(cls) -> float:
        return float(Tool._total_latency_sec)


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Gives a clear, accurate final answer to the given task."
    inputs = {"answer": {"type": "string", "description": "The clear, accurate final answer to the task"}}
    output_type = "string"

    def forward(self, answer: Any) -> Any:
        return answer


__all__ = ["AUTHORIZED_TYPES", "Tool", "FinalAnswerTool"]
