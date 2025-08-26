"""
Handles transforming from Responses API -> LiteLLM completion  (Chat Completion API)
"""

import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

from openai.types.responses.tool_param import FunctionToolParam
from typing_extensions import TypedDict

from litellm.caching import InMemoryCache
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.responses.litellm_completion_transformation.session_handler import (
    ResponsesSessionHandler,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionImageObject,
    ChatCompletionImageUrlObject,
    ChatCompletionResponseMessage,
    ChatCompletionSystemMessage,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolMessage,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    ChatCompletionUserMessage,
    GenericChatCompletionMessage,
    OpenAIMcpServerTool,
    OpenAIWebSearchOptions,
    OpenAIWebSearchUserLocation,
    Reasoning,
    ResponseAPIUsage,
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
)
from litellm.types.responses.main import (
    GenericResponseOutputItem,
    GenericResponseOutputItemContentAnnotation,
    OutputFunctionToolCall,
    OutputText,
)
from litellm.types.utils import (
    ChatCompletionAnnotation,
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
    Usage,
)

########### Initialize Classes used for Responses API  ###########
TOOL_CALLS_CACHE = InMemoryCache()


class ChatCompletionSession(TypedDict, total=False):
    messages: List[
        Union[
            AllMessageValues,
            GenericChatCompletionMessage,
            ChatCompletionMessageToolCall,
            ChatCompletionResponseMessage,
            Message,
        ]
    ]
    litellm_session_id: Optional[str]


########### End of Initialize Classes used for Responses API  ###########


class LiteLLMCompletionResponsesConfig:
    @staticmethod
    def get_supported_openai_params(model: str) -> list:
        """
        LiteLLM Adapter from OpenAI Responses API to Chat Completion API supports a subset of OpenAI Responses API params
        """
        return [
            "input",
            "model",
            "instructions",
            "max_output_tokens",
            "metadata",
            "parallel_tool_calls",
            "previous_response_id",
            "stream",
            "temperature",
            "tool_choice",
            "tools",
            "top_p",
            "user",
        ]

    @staticmethod
    def transform_responses_api_request_to_chat_completion_request(
        model: str,
        input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
        custom_llm_provider: Optional[str] = None,
        stream: Optional[bool] = None,
        **kwargs,
    ) -> dict:
        """
        Transform a Responses API request into a Chat Completion request
        """
        tools, web_search_options = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            responses_api_request.get("tools") or []  # type: ignore
        )
        litellm_completion_request: dict = {
            "messages": LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
                input=input,
                responses_api_request=responses_api_request,
            ),
            "model": model,
            "tool_choice": responses_api_request.get("tool_choice"),
            "tools": tools,
            "top_p": responses_api_request.get("top_p"),
            "user": responses_api_request.get("user"),
            "temperature": responses_api_request.get("temperature"),
            "parallel_tool_calls": responses_api_request.get("parallel_tool_calls"),
            "max_tokens": responses_api_request.get("max_output_tokens"),
            "stream": stream,
            "metadata": kwargs.get("metadata"),
            "service_tier": kwargs.get("service_tier"),
            "web_search_options": web_search_options,
            # litellm specific params
            "custom_llm_provider": custom_llm_provider,
        }

        # Responses API `Completed` events require usage, we pass `stream_options` to litellm.completion to include usage
        if stream is True:
            stream_options = {
                "include_usage": True,
            }
            litellm_completion_request["stream_options"] = stream_options
            litellm_logging_obj: Optional[LiteLLMLoggingObj] = kwargs.get(
                "litellm_logging_obj"
            )
            if litellm_logging_obj:
                litellm_logging_obj.stream_options = stream_options

        # only pass non-None values
        litellm_completion_request = {
            k: v for k, v in litellm_completion_request.items() if v is not None
        }

        return litellm_completion_request

    @staticmethod
    def transform_responses_api_input_to_messages(
        input: Union[str, ResponseInputParam],
        responses_api_request: Union[ResponsesAPIOptionalRequestParams, dict],
    ) -> List[
        Union[
            AllMessageValues,
            GenericChatCompletionMessage,
            ChatCompletionMessageToolCall,
            ChatCompletionResponseMessage,
            Message,
        ]
    ]:
        """
        Transform a Responses API input into a list of messages
        """
        messages: List[
            Union[
                AllMessageValues,
                GenericChatCompletionMessage,
                ChatCompletionMessageToolCall,
                ChatCompletionResponseMessage,
                Message,
            ]
        ] = []
        if responses_api_request.get("instructions"):
            messages.append(
                LiteLLMCompletionResponsesConfig.transform_instructions_to_system_message(
                    responses_api_request.get("instructions")
                )
            )

        messages.extend(
            LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
                input=input,
            )
        )

        return messages

    @staticmethod
    async def async_responses_api_session_handler(
        previous_response_id: str,
        litellm_completion_request: dict,
    ) -> dict:
        """
        Async hook to get the chain of previous input and output pairs and return a list of Chat Completion messages
        """
        chat_completion_session = ChatCompletionSession(
            messages=[], litellm_session_id=None
        )
        if previous_response_id:
            chat_completion_session = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
                previous_response_id=previous_response_id
            )
        _messages = litellm_completion_request.get("messages") or []
        session_messages = chat_completion_session.get("messages") or []
        litellm_completion_request["messages"] = session_messages + _messages
        litellm_completion_request[
            "litellm_trace_id"
        ] = chat_completion_session.get("litellm_session_id")
        return litellm_completion_request

    @staticmethod
    def _transform_response_input_param_to_chat_completion_message(
        input: Union[str, ResponseInputParam],
    ) -> List[
        Union[
            AllMessageValues,
            GenericChatCompletionMessage,
            ChatCompletionMessageToolCall,
            ChatCompletionResponseMessage,
        ]
    ]:
        """
        Transform a ResponseInputParam into a Chat Completion message
        """
        messages: List[
            Union[
                AllMessageValues,
                GenericChatCompletionMessage,
                ChatCompletionMessageToolCall,
                ChatCompletionResponseMessage,
            ]
        ] = []
        tool_call_output_messages: List[
            Union[
                AllMessageValues,
                GenericChatCompletionMessage,
                ChatCompletionMessageToolCall,
                ChatCompletionResponseMessage,
            ]
        ] = []

        if isinstance(input, str):
            messages.append(ChatCompletionUserMessage(role="user", content=input))
        elif isinstance(input, list):
            # Group items by response ID to handle responses API output items properly
            grouped_items = LiteLLMCompletionResponsesConfig._group_responses_api_items_by_response_id(input)
            
            
            for group in grouped_items:
                if len(group) == 1:
                    # Single item, process normally
                    _input = group[0]
                    chat_completion_messages = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
                        input_item=_input
                    )

                    #########################################################
                    # If Input Item is a Tool Call Output, add it to the tool_call_output_messages list
                    #########################################################
                    if LiteLLMCompletionResponsesConfig._is_input_item_tool_call_output(
                        input_item=_input
                    ):
                        tool_call_output_messages.extend(chat_completion_messages)
                    else:
                        messages.extend(chat_completion_messages)
                else:
                    # Multiple items from same response, combine them properly
                    combined_messages = LiteLLMCompletionResponsesConfig._transform_grouped_responses_api_items_to_chat_completion_message(
                        grouped_items=group
                    )
                    messages.extend(combined_messages)

        messages.extend(tool_call_output_messages)
        return messages

    @staticmethod
    def _group_responses_api_items_by_response_id(input_items: List[Any]) -> List[List[Any]]:
        """
        Group responses API items by their response ID to handle multi-part responses properly.
        Items from the same response (reasoning, message, function_call) should be grouped together.
        """
        groups = []
        current_response_group = []
        
        for item in input_items:
            item_type = None
            item_role = None
            
            # Identify item type and role
            if isinstance(item, dict):
                item_type = item.get("type")
                item_role = item.get("role")
            elif hasattr(item, 'type'):
                item_type = getattr(item, 'type', None)
                item_role = getattr(item, 'role', None)
            
            # Start a new group if we hit a user message or if this is the first item
            if item_role == "user" or (len(current_response_group) == 0 and item_role != "assistant"):
                if current_response_group:
                    groups.append(current_response_group)
                current_response_group = [item]
            # Group assistant-related items together (reasoning, message, function_call)
            elif (item_type in ["reasoning", "message", "function_call"] or 
                  item_role == "assistant" or 
                  (hasattr(item, '__class__') and 'Function' in item.__class__.__name__)):
                current_response_group.append(item)
            # Function call output should also be grouped with the function call
            elif item_type == "function_call_output":
                # If current group has function calls, add to it; otherwise start new group
                if current_response_group:
                    current_response_group.append(item)
                else:
                    current_response_group = [item]
            else:
                # Other items get their own group
                if current_response_group:
                    groups.append(current_response_group)
                    current_response_group = []
                groups.append([item])
        
        # Don't forget the last group
        if current_response_group:
            groups.append(current_response_group)
        
        return groups

    @staticmethod
    def _transform_grouped_responses_api_items_to_chat_completion_message(
        grouped_items: List[Any]
    ) -> List[Union[AllMessageValues, GenericChatCompletionMessage, ChatCompletionResponseMessage]]:
        """
        Transform a group of related responses API items (reasoning, message, function_call) 
        into a single coherent chat completion message with proper thinking content.
        """
        # Separate the different types of items
        reasoning_items = []
        message_items = []
        function_call_items = []
        other_items = []
        
        for item in grouped_items:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "reasoning":
                    reasoning_items.append(item)
                elif item_type == "message":
                    message_items.append(item)
                elif item_type == "function_call":
                    function_call_items.append(item)
                else:
                    other_items.append(item)
            else:
                other_items.append(item)
        
        # Separate different types of items
        user_messages = []
        assistant_items = []
        function_call_items = []
        function_call_output_items = []
        
        for item in grouped_items:
            item_type = None
            item_role = None
            
            if isinstance(item, dict):
                item_type = item.get("type")
                item_role = item.get("role")
            elif hasattr(item, 'type'):
                item_type = getattr(item, 'type', None)
                item_role = getattr(item, 'role', None)
            
            if item_type == "function_call_output":
                function_call_output_items.append(item)
            elif item_type == "function_call" or (hasattr(item, '__class__') and 'Function' in item.__class__.__name__):
                function_call_items.append(item)
            elif item_role == "user":
                user_messages.append(item)
            else:
                assistant_items.append(item)
        
        result = []
        
        # Process user messages first
        for user_item in user_messages:
            messages = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
                input_item=user_item
            )
            result.extend(messages)
        
        # Build combined assistant message from reasoning, message, and function call items
        if assistant_items:
            role = "assistant"
            content = []
            
            # Add thinking content from reasoning items first
            for item in assistant_items:
                if (isinstance(item, dict) and item.get("type") == "reasoning") or \
                   (hasattr(item, 'type') and getattr(item, 'type') == "reasoning"):
                    reasoning_content = item.get("content", []) if isinstance(item, dict) else getattr(item, 'content', [])
                    for reasoning_content_item in reasoning_content:
                        if hasattr(reasoning_content_item, 'text'):
                            thinking_text = getattr(reasoning_content_item, 'text')
                        else:
                            thinking_text = reasoning_content_item.get("text", "")
                        
                        if thinking_text:
                            # Try to find signature from attached data
                            signature = None
                            
                            # Check if signature was attached to the reasoning item
                            if hasattr(item, '_extracted_signature'):
                                signature = getattr(item, '_extracted_signature')
                            elif hasattr(item, '_hidden_params') and item._hidden_params.get('extracted_signature'):
                                signature = item._hidden_params['extracted_signature']
                            
                            if signature:
                                # Convert reasoning to thinking format for Anthropic with real signature
                                thinking_block = {
                                    "type": "thinking",
                                    "thinking": thinking_text,
                                    "signature": signature
                                }
                                content.append(thinking_block)
            
            # Add text content from message items
            for item in assistant_items:
                if (isinstance(item, dict) and item.get("type") == "message") or \
                   (hasattr(item, 'type') and getattr(item, 'type') == "message"):
                    message_content = item.get("content", []) if isinstance(item, dict) else getattr(item, 'content', [])
                    for msg_content_item in message_content:
                        if hasattr(msg_content_item, 'text'):
                            text_content = getattr(msg_content_item, 'text')
                        else:
                            text_content = msg_content_item.get("text", "")
                            
                        if text_content:  # Only add non-empty text
                            content.append({
                                "type": "text",
                                "text": text_content
                            })
            
            # Add tool_use blocks for function call items (from both assistant_items and function_call_items)
            all_function_calls = []
            
            # Check assistant_items for any function calls
            for item in assistant_items:
                if (isinstance(item, dict) and item.get("type") == "function_call") or \
                   (hasattr(item, 'type') and getattr(item, 'type') == "function_call"):
                    all_function_calls.append(item)
            
            # Add the separate function_call_items
            all_function_calls.extend(function_call_items)
            
            # Process all function calls
            for item in all_function_calls:
                call_id = item.get("call_id") or item.get("id") if isinstance(item, dict) else \
                          getattr(item, 'call_id', None) or getattr(item, 'id', None)
                name = item.get("name") if isinstance(item, dict) else getattr(item, 'name', None)
                arguments = item.get("arguments") if isinstance(item, dict) else getattr(item, 'arguments', None)
                
                if call_id and name and arguments:
                    tool_use_block = {
                        "type": "tool_use",
                        "id": call_id,
                        "name": name,
                        "input": json.loads(arguments)
                    }
                    content.append(tool_use_block)
            
            # Create the combined assistant message
            from litellm.types.llms.openai import ChatCompletionAssistantMessage
            result.append(
                ChatCompletionAssistantMessage(
                    role=role,
                    content=LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(
                        content if content else None
                    )
                )
            )
        
        # Process function_call_output items as tool messages
        for function_output in function_call_output_items:
            messages = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
                input_item=function_output
            )
            result.extend(messages)
        
        return result

    @staticmethod
    def _ensure_tool_call_output_has_corresponding_tool_call(
        messages: List[Union[AllMessageValues, GenericChatCompletionMessage]],
    ) -> bool:
        """
        If any tool call output is present, ensure there is a corresponding tool call/tool_use block
        """
        for message in messages:
            if message.get("role") == "tool":
                return True
        return False

    @staticmethod
    def _transform_responses_api_input_item_to_chat_completion_message(
        input_item: Any,
    ) -> List[
        Union[
            AllMessageValues,
            GenericChatCompletionMessage,
            ChatCompletionResponseMessage,
        ]
    ]:
        """
        Transform a Responses API input item into a Chat Completion message

        - EasyInputMessageParam
        - Message
        - ResponseOutputMessageParam
        - ResponseFileSearchToolCallParam
        - ResponseComputerToolCallParam
        - ComputerCallOutput
        - ResponseFunctionWebSearchParam
        - ResponseFunctionToolCallParam
        - FunctionCallOutput
        - ResponseReasoningItemParam
        - ItemReference
        """
        if LiteLLMCompletionResponsesConfig._is_input_item_tool_call_output(input_item):
            # handle executed tool call results
            return LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
                tool_call_output=input_item
            )
        elif LiteLLMCompletionResponsesConfig._is_input_item_function_call(input_item):
            # handle function call input items
            return LiteLLMCompletionResponsesConfig._transform_responses_api_function_call_to_chat_completion_message(
                function_call=input_item
            )
        else:
            role = input_item.get("role") or "user"
            content = input_item.get("content")
            
            # Special handling for assistant messages with tool_use content blocks
            if role == "assistant" and isinstance(content, list):
                tool_calls = []
                filtered_content = []
                
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_use":
                        # Convert tool_use content to tool_calls
                        tool_calls.append({
                            "id": item.get("id"),
                            "type": "function",
                            "function": {
                                "name": item.get("name"),
                                "arguments": json.dumps(item.get("input", {}))
                            }
                        })
                    else:
                        # Keep other content types
                        filtered_content.append(item)
                
                # If we found tool_use blocks, create message with tool_calls
                if tool_calls:
                    from litellm.types.llms.openai import ChatCompletionAssistantMessage
                    return [
                        ChatCompletionAssistantMessage(
                            role=role,
                            content=LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(
                                filtered_content if filtered_content else None
                            ),
                            tool_calls=tool_calls
                        )
                    ]
            
            # Special handling for user messages with tool_result content blocks
            elif role == "user" and isinstance(content, list):
                from litellm.types.llms.openai import ChatCompletionToolMessage
                messages = []
                remaining_content = []
                
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        # Convert tool_result to a tool message
                        messages.append(
                            ChatCompletionToolMessage(
                                role="tool",
                                content=item.get("content", ""),
                                tool_call_id=item.get("tool_use_id", "")
                            )
                        )
                    else:
                        # Keep other content types for the user message
                        remaining_content.append(item)
                
                # Add user message for any remaining content
                if remaining_content:
                    messages.append(
                        GenericChatCompletionMessage(
                            role=role,
                            content=LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(
                                remaining_content
                            ),
                        )
                    )
                
                # Return tool messages if found, otherwise fall through to default
                if messages:
                    return messages
            
            # Default handling for other message types
            return [
                GenericChatCompletionMessage(
                    role=role,
                    content=LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(
                        content
                    ),
                )
            ]

    @staticmethod
    def _is_input_item_tool_call_output(input_item: Any) -> bool:
        """
        Check if the input item is a tool call output
        """
        return input_item.get("type") in [
            "function_call_output",
            "web_search_call",
            "computer_call_output",
        ]

    @staticmethod
    def _is_input_item_function_call(input_item: Any) -> bool:
        """
        Check if the input item is a function call
        """
        return input_item.get("type") == "function_call"

    @staticmethod
    def _transform_responses_api_tool_call_output_to_chat_completion_message(
        tool_call_output: Dict[str, Any],
    ) -> List[
        Union[
            AllMessageValues,
            GenericChatCompletionMessage,
            ChatCompletionResponseMessage,
        ]
    ]:
        """
        ChatCompletionToolMessage is used to indicate the output from a tool call
        """
        tool_output_message = ChatCompletionToolMessage(
            role="tool",
            content=tool_call_output.get("output") or "",
            tool_call_id=tool_call_output.get("call_id") or "",
        )

        _tool_use_definition = TOOL_CALLS_CACHE.get_cache(
            key=tool_call_output.get("call_id") or "",
        )
        if _tool_use_definition:
            """
            Append the tool use definition to the list of messages


            Providers like Anthropic require the tool use definition to be included with the tool output

            - Input:
                {'function':
                    arguments:'{"command": ["echo","<html>\\n<head>\\n  <title>Hello</title>\\n</head>\\n<body>\\n  <h1>Hi</h1>\\n</body>\\n</html>",">","index.html"]}',
                    name='shell',
                    'id': 'toolu_018KFWsEySHjdKZPdUzXpymJ',
                    'type': 'function'
                }
            - Output:
                {
                    "id": "toolu_018KFWsEySHjdKZPdUzXpymJ",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"latitude\":48.8566,\"longitude\":2.3522}"
                        }
                }

            """
            function: dict = _tool_use_definition.get("function") or {}
            tool_call_chunk = ChatCompletionToolCallChunk(
                id=_tool_use_definition.get("id") or "",
                type=_tool_use_definition.get("type") or "function",
                function=ChatCompletionToolCallFunctionChunk(
                    name=function.get("name") or "",
                    arguments=function.get("arguments") or "",
                ),
                index=0,
            )
            chat_completion_response_message = ChatCompletionResponseMessage(
                tool_calls=[tool_call_chunk],
                role="assistant",
            )
            return [chat_completion_response_message, tool_output_message]

        return [tool_output_message]

    @staticmethod
    def _transform_responses_api_function_call_to_chat_completion_message(
        function_call: Dict[str, Any],
    ) -> List[
        Union[
            AllMessageValues,
            GenericChatCompletionMessage,
            ChatCompletionResponseMessage,
        ]
    ]:
        """
        Transform a Responses API function_call into a Chat Completion message with tool calls

        Handles Input items of this type:
        function_call:
        ```json
        {
            "type": "function_call",
            "arguments":"{\"location\": \"São Paulo, Brazil\"}",
            "call_id": "call_v2wlBzrlTIFl9FxPeY774GHZ",
            "name": "get_weather",
            "id": "fc_685c42deefc0819a822b6936faaa30be0c76bc1491ab6619",
            "status": "completed"
        }
        ```
        """
        # Create a tool call for the function call
        tool_call = ChatCompletionToolCallChunk(
            id=function_call.get("call_id") or function_call.get("id") or "",
            type="function",
            function=ChatCompletionToolCallFunctionChunk(
                name=function_call.get("name") or "",
                arguments=function_call.get("arguments") or "",
            ),
            index=0,
        )
        
        # Create an assistant message with the tool call
        chat_completion_response_message = ChatCompletionResponseMessage(
            tool_calls=[tool_call],
            role="assistant",
            content=None,  # Function calls don't have content
        )
        
        return [chat_completion_response_message]

    @staticmethod
    def _transform_input_file_item_to_file_item(item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a Responses API input_file item to a Chat Completion file item

        Args:
            item: Dictionary containing input_file type with file_id and/or file_data

        Returns:
            Dictionary with transformed file structure for Chat Completion
        """
        file_dict: Dict[str, Any] = {}
        keys = ["file_id", "file_data"]
        for key in keys:
            if item.get(key):
                file_dict[key] = item.get(key)

        new_item: Dict[str, Any] = {"type": "file", "file": file_dict}
        return new_item

    @staticmethod
    def _transform_input_image_item_to_image_item(item: Dict[str, Any]) -> ChatCompletionImageObject:
        """
        Transform a Responses API input_image item to a Chat Completion image item
        """
        image_url_obj = ChatCompletionImageUrlObject(
            url=item.get("image_url") or "",
            detail=item.get("detail") or "auto"
        )

        return ChatCompletionImageObject(
            type="image_url",
            image_url=image_url_obj
        )

    @staticmethod
    def _transform_responses_api_content_to_chat_completion_content(
        content: Any,
    ) -> Union[str, List[Union[str, Dict[str, Any]]]]:
        """
        Transform a Responses API content into a Chat Completion content
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            content_list: List[Union[str, Dict[str, Any]]] = []
            for item in content:
                if isinstance(item, str):
                    content_list.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "input_file":
                        content_list.append(
                            LiteLLMCompletionResponsesConfig._transform_input_file_item_to_file_item(
                                item
                            )
                        )
                    elif item.get("type") == "input_image":
                        content_list.append(
                            dict(
                                LiteLLMCompletionResponsesConfig._transform_input_image_item_to_image_item(
                                    item
                                )
                            )
                        )
                    else:
                        # Handle different content types with their specific field names
                        item_type = item.get("type")
                        transformed_type = LiteLLMCompletionResponsesConfig._get_chat_completion_request_content_type(
                            item_type or "text"
                        )
                        
                        result_item = {"type": transformed_type}
                        
                        # Use the correct field name based on the content type
                        if item_type == "thinking":
                            result_item["thinking"] = item.get("thinking")
                            # Preserve signature if present
                            if item.get("signature"):
                                result_item["signature"] = item.get("signature")
                        elif item_type == "tool_use":
                            result_item["id"] = item.get("id")
                            result_item["name"] = item.get("name") 
                            result_item["input"] = item.get("input")
                        elif item_type == "tool_result":
                            result_item["tool_use_id"] = item.get("tool_use_id")
                            result_item["content"] = item.get("content")
                        else:
                            # Default to text field for other types
                            result_item["text"] = item.get("text")
                        
                        content_list.append(result_item)
            return content_list
        else:
            raise ValueError(f"Invalid content type: {type(content)}")

    @staticmethod
    def _get_chat_completion_request_content_type(content_type: str) -> str:
        """
        Get the Chat Completion request content type
        """
        # Responses API content has `input_` prefix, if it exists, remove it
        if content_type.startswith("input_"):
            return content_type[len("input_") :]
        else:
            return content_type

    @staticmethod
    def transform_instructions_to_system_message(
        instructions: Optional[str],
    ) -> ChatCompletionSystemMessage:
        """
        Transform a Instructions into a system message
        """
        return ChatCompletionSystemMessage(role="system", content=instructions or "")

    @staticmethod
    def transform_responses_api_tools_to_chat_completion_tools(
        tools: Optional[List[Union[FunctionToolParam, OpenAIMcpServerTool]]],
    ) -> Tuple[List[Union[ChatCompletionToolParam, OpenAIMcpServerTool]], Optional[OpenAIWebSearchOptions]]:
        """
        Transform a Responses API tools into a Chat Completion tools
        """
        if tools is None:
            return [], None
        chat_completion_tools: List[
            Union[ChatCompletionToolParam, OpenAIMcpServerTool]
        ] = []
        web_search_options: Optional[OpenAIWebSearchOptions] = None
        for tool in tools:
            if tool.get("type") == "mcp":
                chat_completion_tools.append(cast(OpenAIMcpServerTool, tool))
            elif tool.get("type") == "web_search_preview" or tool.get("type") == "web_search":
                _search_context_size: Literal["low", "medium", "high"] = cast(Literal["low", "medium", "high"], tool.get("search_context_size"))
                _user_location: Optional[OpenAIWebSearchUserLocation] = cast(Optional[OpenAIWebSearchUserLocation], tool.get("user_location") or None)
                web_search_options = OpenAIWebSearchOptions(
                    search_context_size=_search_context_size,
                    user_location=_user_location,
                )
            else:
                typed_tool = cast(FunctionToolParam, tool)
                chat_completion_tools.append(
                    ChatCompletionToolParam(
                        type="function",
                        function=ChatCompletionToolParamFunctionChunk(
                            name=typed_tool.get("name") or "",
                            description=typed_tool.get("description") or "",
                            parameters=dict(typed_tool.get("parameters", {}) or {}),
                            strict=typed_tool.get("strict", False) or False,
                        ),
                    )
                )
        return chat_completion_tools, web_search_options

    @staticmethod
    def transform_chat_completion_tools_to_responses_tools(
        chat_completion_response: ModelResponse,
    ) -> List[OutputFunctionToolCall]:
        """
        Transform a Chat Completion tools into a Responses API tools
        """
        all_chat_completion_tools: List[ChatCompletionMessageToolCall] = []
        for choice in chat_completion_response.choices:
            if isinstance(choice, Choices):
                if choice.message.tool_calls:
                    all_chat_completion_tools.extend(choice.message.tool_calls)
                    for tool_call in choice.message.tool_calls:
                        TOOL_CALLS_CACHE.set_cache(
                            key=tool_call.id,
                            value=tool_call,
                        )

        responses_tools: List[OutputFunctionToolCall] = []
        for tool in all_chat_completion_tools:
            if tool.type == "function":
                function_definition = tool.function
                responses_tools.append(
                    OutputFunctionToolCall(
                        name=function_definition.name or "",
                        arguments=function_definition.get("arguments") or "",
                        call_id=tool.id or "",
                        id=tool.id or "",
                        type="function_call",  # critical this is "function_call" to work with tools like openai codex
                        status=function_definition.get("status") or "completed",
                    )
                )
        return responses_tools

    @staticmethod
    def transform_chat_completion_response_to_responses_api_response(
        request_input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
        chat_completion_response: Union[ModelResponse, dict],
    ) -> ResponsesAPIResponse:
        """
        Transform a Chat Completion response into a Responses API response
        """
        if isinstance(chat_completion_response, dict):
            chat_completion_response = ModelResponse(**chat_completion_response)
        responses_api_response: ResponsesAPIResponse = ResponsesAPIResponse(
            id=chat_completion_response.id,
            created_at=chat_completion_response.created,
            model=chat_completion_response.model,
            object=chat_completion_response.object,
            error=getattr(chat_completion_response, "error", None),
            incomplete_details=getattr(
                chat_completion_response, "incomplete_details", None
            ),
            instructions=getattr(chat_completion_response, "instructions", None),
            metadata=getattr(chat_completion_response, "metadata", {}),
            output=LiteLLMCompletionResponsesConfig._transform_chat_completion_choices_to_responses_output(
                chat_completion_response=chat_completion_response,
                choices=getattr(chat_completion_response, "choices", []),
            ),
            parallel_tool_calls=getattr(
                chat_completion_response, "parallel_tool_calls", False
            ),
            temperature=getattr(chat_completion_response, "temperature", 0),
            tool_choice=getattr(chat_completion_response, "tool_choice", "auto"),
            tools=getattr(chat_completion_response, "tools", []),
            top_p=getattr(chat_completion_response, "top_p", None),
            max_output_tokens=getattr(
                chat_completion_response, "max_output_tokens", None
            ),
            previous_response_id=getattr(
                chat_completion_response, "previous_response_id", None
            ),
            reasoning=Reasoning(),
            status=getattr(chat_completion_response, "status", "completed"),
            text={},
            truncation=getattr(chat_completion_response, "truncation", None),
            usage=LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
                chat_completion_response=chat_completion_response
            ),
            user=getattr(chat_completion_response, "user", None),
        )
        
        # Store raw response data in hidden params if available from the ModelResponse
        if hasattr(chat_completion_response, '_preserved_anthropic_original'):
            # Use the preserved anthropic original response for signature extraction
            responses_api_response._hidden_params['anthropic_original_response'] = chat_completion_response._preserved_anthropic_original
        
        if hasattr(chat_completion_response, '_hidden_params'):
            # Also pass through any anthropic original response from hidden params (fallback)
            anthropic_original = chat_completion_response._hidden_params.get('anthropic_original_response')
            if anthropic_original and 'anthropic_original_response' not in responses_api_response._hidden_params:
                responses_api_response._hidden_params['anthropic_original_response'] = anthropic_original
            
            # Also keep any other hidden params
            responses_api_response._hidden_params['raw_response_data'] = chat_completion_response._hidden_params.get('original_response')
        
        return responses_api_response

    @staticmethod
    def _transform_chat_completion_choices_to_responses_output(
        chat_completion_response: ModelResponse,
        choices: List[Choices],
    ) -> List[Union[GenericResponseOutputItem, OutputFunctionToolCall]]:
        responses_output: List[
            Union[GenericResponseOutputItem, OutputFunctionToolCall]
        ] = []

        responses_output.extend(
            LiteLLMCompletionResponsesConfig._extract_reasoning_output_items(
                chat_completion_response, choices
            )
        )
        responses_output.extend(
            LiteLLMCompletionResponsesConfig._extract_message_output_items(
                chat_completion_response, choices
            )
        )
        responses_output.extend(
            LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(
                chat_completion_response=chat_completion_response
            )
        )
        return responses_output

    @staticmethod
    def _extract_reasoning_output_items(
        chat_completion_response: ModelResponse,
        choices: List[Choices],
    ) -> List[GenericResponseOutputItem]:
        for choice in choices:
            if hasattr(choice, "message") and choice.message:
                message = choice.message
                if hasattr(message, "reasoning_content") and message.reasoning_content:
                    # Only check the first choice for reasoning content
                    return [
                        GenericResponseOutputItem(
                            type="reasoning",
                            id=f"{chat_completion_response.id}_reasoning",
                            status=choice.finish_reason,
                            role="assistant",
                            content=[
                                OutputText(
                                    type="output_text",
                                    text=message.reasoning_content,
                                    annotations=[],
                                )
                            ],
                        )
                    ]
        return []

    @staticmethod
    def _extract_message_output_items(
        chat_completion_response: ModelResponse,
        choices: List[Choices],
    ) -> List[GenericResponseOutputItem]:
        message_output_items = []
        for choice in choices:
            message_output_items.append(
                GenericResponseOutputItem(
                    type="message",
                    id=chat_completion_response.id,
                    status=choice.finish_reason,
                    role=choice.message.role,
                    content=[
                        LiteLLMCompletionResponsesConfig._transform_chat_message_to_response_output_text(
                            choice.message
                        )
                    ],
                )
            )
        return message_output_items

    @staticmethod
    def _transform_responses_api_outputs_to_chat_completion_messages(
        responses_api_output: ResponsesAPIResponse,
    ) -> List[
        Union[
            AllMessageValues,
            GenericChatCompletionMessage,
            ChatCompletionMessageToolCall,
        ]
    ]:
        messages: List[
            Union[
                AllMessageValues,
                GenericChatCompletionMessage,
                ChatCompletionMessageToolCall,
            ]
        ] = []
        output_items = responses_api_output.output
        for _output_item in output_items:
            output_item: dict = dict(_output_item)
            if output_item.get("type") == "function_call":
                # handle function call output
                messages.append(
                    LiteLLMCompletionResponsesConfig._transform_responses_output_tool_call_to_chat_completion_output_tool_call(
                        tool_call=output_item
                    )
                )
            else:
                # transform as generic ResponseOutputItem
                messages.append(
                    GenericChatCompletionMessage(
                        role=str(output_item.get("role")) or "user",
                        content=LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(
                            output_item.get("content")
                        ),
                    )
                )
        return messages

    @staticmethod
    def _transform_responses_output_tool_call_to_chat_completion_output_tool_call(
        tool_call: dict,
    ) -> ChatCompletionMessageToolCall:
        return ChatCompletionMessageToolCall(
            id=tool_call.get("id") or "",
            type="function",
            function=Function(
                name=tool_call.get("name") or "",
                arguments=tool_call.get("arguments") or "",
            ),
        )

    @staticmethod
    def _transform_chat_message_to_response_output_text(
        message: Message,
    ) -> OutputText:
        return OutputText(
            type="output_text",
            text=message.content,
            annotations=LiteLLMCompletionResponsesConfig._transform_chat_completion_annotations_to_response_output_annotations(
                annotations=getattr(message, "annotations", None)
            ),
        )

    @staticmethod
    def _transform_chat_completion_annotations_to_response_output_annotations(
        annotations: Optional[List[ChatCompletionAnnotation]],
    ) -> List[GenericResponseOutputItemContentAnnotation]:
        response_output_annotations: List[
            GenericResponseOutputItemContentAnnotation
        ] = []

        if annotations is None:
            return response_output_annotations

        for annotation in annotations:
            annotation_type = annotation.get("type")
            if annotation_type == "url_citation" and "url_citation" in annotation:
                url_citation = annotation["url_citation"]
                response_output_annotations.append(
                    GenericResponseOutputItemContentAnnotation(
                        type=annotation_type,
                        start_index=url_citation.get("start_index"),
                        end_index=url_citation.get("end_index"),
                        url=url_citation.get("url"),
                        title=url_citation.get("title"),
                    )
                )
            # Handle other annotation types here

        return response_output_annotations


    @staticmethod
    def _transform_chat_completion_usage_to_responses_usage(
        chat_completion_response: Union[ModelResponse, Usage],
    ) -> ResponseAPIUsage:
        if isinstance(chat_completion_response, ModelResponse):
            usage: Optional[Usage] = getattr(chat_completion_response, "usage", None)
        else:
            usage = chat_completion_response
        if usage is None:
            return ResponseAPIUsage(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
            )
        return ResponseAPIUsage(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )
