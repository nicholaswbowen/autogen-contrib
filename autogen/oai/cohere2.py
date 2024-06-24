"""
Create an OpenAI-compatible client for the Cohere API.

Example usage:
Install the `cohere` package by running `python -m pip install cohere --upgrade`.
- https://github.com/cohere-ai/cohere-python

import autogen

config_list = [
    {
        "model": "claude-3-sonnet-20240229",
        "api_key": os.getenv("COHERE_API_KEY"),
        "api_type": "cohere",
        "force_single_step": false,
        "tools": ['web_search']
    }
]

assistant = autogen.AssistantAgent("assistant", llm_config={"config_list": config_list})
"""

from __future__ import annotations

import copy
import inspect
import json
import os
import warnings
import cohere
from typing import Any, Dict, List, Tuple, Union


from typing import List, Optional, Any
from client_utils import validate_parameter
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion import ChatCompletionMessage, Choice
from typing_extensions import Annotated

class OpenAIRequest:
    def __init__(self):
        self.model: str = ""
        self.messages: List[dict] = []
        self.stream: bool = False
        self.max_tokens: int = 0

class CohereRequest:
    def __init__(self):
        self.model: str = ""
        self.chat_history: List['ChatMessage'] = []
        self.message: str = ""
        self.stream: bool = False
        self.max_tokens: int = 0

class ChatMessage:
    def __init__(self):
        self.role: str = ""
        self.message: str = ""

class CohereResponse:
    def __init__(self):
        self.is_finished: bool = False
        self.event_type: str = ""
        self.text: Optional[str] = None
        self.finish_reason: Optional[str] = None

class OpenAIResponse:
    def __init__(self):
        self.id: str = ""
        self.object: str = ""
        self.created: int = 0
        self.model: str = ""
        self.choices: List['OpenAIChoice'] = []

class OpenAIChoice:
    def __init__(self):
        self.index: int = 0
        self.delta: 'OpenAIDelta' = OpenAIDelta()
        self.logprobs: Any = None
        self.finish_reason: Optional[str] = None

# Shouldnt need te stream responses since autogen doesn't support them.
class OpenAINonStreamResponse:
    def __init__(self):
        self.id: str = ""
        self.object: str = ""
        self.created: int = 0
        self.model: str = ""
        self.choices: List['OpenAINonStreamChoice'] = []

class OpenAINonStreamChoice:
    def __init__(self):
        self.index: int = 0
        self.message: 'OpenAIDelta' = OpenAIDelta()
        self.finish_reason: Optional[str] = None

class OpenAIDelta:
    def __init__(self):
        self.role: Optional[str] = None
        self.content: Optional[str] = None

COHERE_PRICING_1k = {
    # "command-r-plus": (0.003, 0.015),
    # "command-r": (0.015, 0.075),
    # "claude-2.0": (0.008, 0.024),
    # "claude-2.1": (0.008, 0.024),
    # "claude-3.0-opus": (0.015, 0.075),
    # "claude-3.0-haiku": (0.00025, 0.00125),
}

class CohereClient:
    def __init__(self, **kwargs: Any):
        """
        Initialize the Cohere API client.
        Args:
            api_key (str): The API key for the Cohere API or set the `COHERE_API_KEY` environment variable.
        """
        self._api_key = kwargs.get("api_key", None)

        if not self._api_key:
            self._api_key = os.getenv("COHERE_API_KEY")

        if self._api_key is None:
            raise ValueError("API key is required to use the Cohere API.")

        self._client = cohere.client(self._api_key)
        self._last_tooluse_status = {}

    def load_config(self, params: Dict[str, Any]):
        """Load the configuration for the Cohere API client."""
        cohere_params = {}

        cohere_params["model"] = params.get("model", None)
        assert cohere_params["model"], "Please provide a `model` in the config_list to use the Cohere API."

        cohere_params["temperature"] = validate_parameter(
            params, "temperature", (float, int), False, 1.0, (0.0, 1.0), None
        )
        cohere_params["max_tokens"] = validate_parameter(params, "max_tokens", int, False, 4096, (1, None), None)
        cohere_params["top_k"] = validate_parameter(params, "top_k", int, True, None, (1, None), None)
        cohere_params["top_p"] = validate_parameter(params, "top_p", (float, int), True, None, (0.0, 1.0), None)
        cohere_params["stop_sequences"] = validate_parameter(params, "stop_sequences", list, True, None, None, None)
        cohere_params["stream"] = validate_parameter(params, "stream", bool, False, False, None, None)

        if cohere_params["stream"]:
            warnings.warn(
                "Streaming is not currently supported, streaming will be disabled.",
                UserWarning,
            )
            cohere_params["stream"] = False

        return cohere_params

    def cost(self, response) -> float:
        """Calculate the cost of the completion using the Cohere pricing."""
        return response.cost

    @property
    def api_key(self):
        return self._api_key

    def create(self, params: Dict[str, Any]):
        """Create a completion for a given config.

        Args:
            params: The params for the completion.

        Returns:
            The completion.
        """
        if "tools" in params:
            converted_functions = self.convert_tools_to_functions(params["tools"])
            params["functions"] = params.get("functions", []) + converted_functions

        raw_contents = params["messages"]
        cohere_params = self.load_config(params)

        processed_messages = []
        for message in raw_contents:

            if message["role"] == "chatbot":
                params["chatbot"] = message["content"]
            elif message["role"] == "function":
                processed_messages.append(self.return_function_call_result(message["content"]))
            elif "function_call" in message:
                processed_messages.append(self.restore_last_tooluse_status())
            elif message["content"] == "":
                message["content"] = "I'm done. Please send TERMINATE"  # Not sure about this one.
                processed_messages.append(message)
            else:
                processed_messages.append(message)

        # Shouldn't be needed for Cohere. 
        # # Check for interleaving roles and correct, for Anthropic must be: user, chatbot, user, etc.
        # for i, message in enumerate(processed_messages):
        #     if message["role"] is not ("user" if i % 2 == 0 else "chatbot"):
        #         message["role"] = "user" if i % 2 == 0 else "chatbot"

        # # Note: When using reflection_with_llm we may end up with an "chatbot" message as the last message
        # if processed_messages[-1]["role"] != "user":
        #     # If the last role is not user, add a continue message at the end
        #     continue_message = {"content": "continue", "role": "user"}
        #     processed_messages.append(continue_message)

        params["messages"] = processed_messages

        # TODO: support stream
        params = params.copy()
        if "functions" in params:
            tools_configs = params.pop("functions")
            tools_configs = [self.openai_func_to_cohere(tool) for tool in tools_configs]
            params["tools"] = tools_configs

        # Anthropic doesn't accept None values, so we need to use keyword argument unpacking instead of setting parameters.
        # Copy params we need into cohere_params
        # Remove any that don't have values
        cohere_params["messages"] = params["messages"]
        if "chatbot" in params:
            cohere_params["chatbot"] = params["chatbot"]
        if "tools" in params:
            cohere_params["tools"] = params["tools"]
        if cohere_params["top_k"] is None:
            del cohere_params["top_k"]
        if cohere_params["top_p"] is None:
            del cohere_params["top_p"]
        if cohere_params["stop_sequences"] is None:
            del cohere_params["stop_sequences"]

        response = self._client.messages.create(**cohere_params)

        # Calculate and save the cost onto the response
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens
        response.cost = _calculate_cost(prompt_tokens, completion_tokens, cohere_params["model"])

        return response

    def message_retrieval(self, response: Union[Message]) -> Union[List[str], List[ChatCompletionMessage]]:
        """Retrieve the messages from the response."""
        messages = response.content
        if len(messages) == 0:
            return [None]
        res = []\
        
        # All cohere models are designed for tool use, shouldn't be neccessary. 
        # if TOOL_ENABLED:
        #     for choice in messages:
        #         if choice.type == "tool_use":
        #             res.insert(0, self.response_to_openai_message(choice))
        #             self._last_tooluse_status["tool_use"] = choice.model_dump()
        #         else:
        #             res.append(choice.text)
        #             self._last_tooluse_status["think"] = choice.text

        #     return res

        # else:
        #     return [  # type: ignore [return-value]
        #         choice.text if choice.message.function_call is not None else choice.message.content  # type: ignore [union-attr]
        #         for choice in messages
        #     ]

    def response_to_openai_message(self, response) -> ChatCompletionMessage:
        """Convert the client response to OpenAI ChatCompletion Message"""
        dict_response = response.model_dump()
        return ChatCompletionMessage(
            content=None,
            role="assistant",
            function_call={"name": dict_response["name"], "arguments": json.dumps(dict_response["input"])},
        )

    def restore_last_tooluse_status(self) -> Dict:
        cached_content = []
        if "think" in self._last_tooluse_status:
            cached_content.append({"type": "text", "text": self._last_tooluse_status["think"]})
        cached_content.append(self._last_tooluse_status["tool_use"])
        res = {"role": "assistant", "content": cached_content}
        return res

    def return_function_call_result(self, result: str) -> Dict:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": self._last_tooluse_status["tool_use"]["id"],
                    "content": result,
                }
            ],
        }

    @staticmethod
    def openai_func_to_cohere(openai_func: dict) -> dict:
        res = openai_func.copy()
        res["input_schema"] = res.pop("parameters")
        return res

    @staticmethod
    def get_usage(response: Message) -> Dict:
        """Get the usage of tokens and their cost information."""
        return {
            "prompt_tokens": response.usage.input_tokens if response.usage is not None else 0,
            "completion_tokens": response.usage.output_tokens if response.usage is not None else 0,
            "total_tokens": (
                response.usage.input_tokens + response.usage.output_tokens if response.usage is not None else 0
            ),
            "cost": response.cost if hasattr(response, "cost") else 0.0,
            "model": response.model,
        }

    @staticmethod
    def convert_tools_to_functions(tools: List) -> List:
        functions = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                functions.append(tool["function"])

        return functions


def _calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Calculate the cost of the completion using the Cohere pricing."""
    total = 0.0

    if model in COHERE_PRICING_1k:
        input_cost_per_1k, output_cost_per_1k = COHERE_PRICING_1k[model]
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        total = input_cost + output_cost
    else:
        warnings.warn(f"Cost calculation not available for model {model}", UserWarning)

    return total
