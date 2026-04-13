import ast
import json
import re
import anthropic
import backoff
import openai
import copy

from llm import create_client, get_model_family_name, get_response_from_llm
from prompts.tooluse_prompt import get_tooluse_prompt
from tools import load_all_tools

CLAUDE_MODEL = 'openrouter/minimax/minimax-m2.5'
OPENAI_MODEL = 'openrouter/minimax/minimax-m2.5'


def is_openai_tool_model(model: str) -> bool:
    model_family = get_model_family_name(model)
    return model_family.startswith('o3-') or model_family.startswith('gpt-5-')


def is_minimax_model(model: str) -> bool:
    return get_model_family_name(model).startswith('minimax-')


def strip_thinking_tags(text: str) -> str:
    """Strip mandatory <think>...</think> reasoning tags from MiniMax responses."""
    if not text:
        return text
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def get_openai_function_call(response):
    for item in getattr(response, 'output', []):
        if getattr(item, 'type', None) == 'function_call':
            return item
    return None


def get_openai_response_items(response):
    return [
        item
        for item in getattr(response, 'output', [])
        if getattr(item, 'type', None) in {'function_call', 'message'}
    ]


def extract_openai_response_text(response):
    output_text = getattr(response, 'output_text', None)
    if output_text:
        return output_text

    text_chunks = []
    for item in getattr(response, 'output', []):
        if getattr(item, 'type', None) != 'message':
            continue
        for block in getattr(item, 'content', []):
            block_type = getattr(block, 'type', None)
            block_text = getattr(block, 'text', None)
            if block_text is None and isinstance(block, dict):
                block_type = block.get('type')
                block_text = block.get('text')
            if block_type in {'output_text', 'text'} and block_text:
                text_chunks.append(block_text)
    return '\n'.join(text_chunks)


def serialize_tool_output(tool_result):
    if isinstance(tool_result, str):
        return tool_result
    try:
        return json.dumps(tool_result, ensure_ascii=False, default=str)
    except TypeError:
        return str(tool_result)


def empty_usage():
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def merge_usage(total_usage, response_usage):
    merged = empty_usage()
    for key in merged:
        merged[key] = int(total_usage.get(key, 0) or 0) + int(response_usage.get(key, 0) or 0)
    return merged


def extract_response_usage(response):
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return empty_usage()

    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
    else:
        input_tokens = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0))
        output_tokens = getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0))
        total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens)
    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }

def process_tool_call(tools_dict, tool_name, tool_input):
    try:
        if tool_name in tools_dict:
            return tools_dict[tool_name]['function'](**tool_input)
        else:
            return f"Error: Tool '{tool_name}' not found"
    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"

@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APITimeoutError, anthropic.RateLimitError, anthropic.APIStatusError),
    max_time=600,
    max_value=60,
)
def get_response_withtools(
    client, model, messages, tools, tool_choice,
    logging=None, max_retry=3
):
    try:
        if 'claude' in model:
            response = client.messages.create(
                model=model,
                messages=messages,
                max_tokens=4096,
                tool_choice=tool_choice,
                tools=tools,
            )
        elif is_minimax_model(model):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=8192,
            )
        elif is_openai_tool_model(model):
            response = client.responses.create(
                model=model,
                input=messages,
                tool_choice=tool_choice,
                tools=tools,
                parallel_tool_calls=False
            )
            response = response
        else:
            raise ValueError(f"Unsupported model: {model}")
        return response
    except Exception as e:
        logging(f"Error in get_response_withtools: {str(e)}")
        if max_retry > 0:
            return get_response_withtools(client, model, messages, tools, tool_choice, logging, max_retry - 1)

        # Hitting the context window limit
        if 'Input is too long for requested model' in str(e):
            pass

        raise  # Re-raise the exception after logging

def check_for_tool_use(response, model=''):
    """
    Checks if the response contains a tool call.
    """
    if 'claude' in model:
        # Claude, check for stop_reason in response
        if response.stop_reason == "tool_use":
            tool_use_block = next(block for block in response.content if block.type == "tool_use")
            return {
                'tool_id': tool_use_block.id,
                'tool_name': tool_use_block.name,
                'tool_input': tool_use_block.input,
            }

    elif is_minimax_model(model):
        # MiniMax via chat.completions: check tool_calls on message
        msg = response.choices[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            try:
                tool_input = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                # MiniMax occasionally returns malformed JSON arguments; skip this tool call
                return None
            return {
                'tool_id': tc.id,
                'tool_name': tc.function.name,
                'tool_input': tool_input,
            }

    elif is_openai_tool_model(model):
        # OpenAI, check for tool_calls in response
        tool_call = get_openai_function_call(response)
        if tool_call is not None:
            return {
                'tool_id': tool_call.call_id,
                'tool_name': tool_call.name,
                'tool_input': json.loads(tool_call.arguments),
            }

    else:
        # Any other LLM, response is str, check for <tool_use> tag in response
        pattern = r'<tool_use>(.*?)</tool_use>'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            tool_use_str = match.group(1).strip()
            try:
                tool_use_dict = ast.literal_eval(tool_use_str)
                if isinstance(tool_use_dict, dict) and 'tool_name' in tool_use_dict and 'tool_input' in tool_use_dict:
                    return tool_use_dict
            except Exception:
                pass

    # No tool use found
    return None

def convert_tool_info(tool_info, model=None):
    """
    Converts tool_info from Claude format to the given model's format.
    """
    if 'claude' in model:
        # should have no change
        return {
            'name': tool_info['name'],
            'description': tool_info['description'],
            'input_schema': tool_info['input_schema'],
        }
    elif is_minimax_model(model):
        # MiniMax via OpenRouter: standard chat.completions tool format, no strict mode
        return {
            'type': 'function',
            'function': {
                'name': tool_info['name'],
                'description': tool_info['description'],
                'parameters': tool_info['input_schema'],
            },
        }
    elif is_openai_tool_model(model):
        def add_additional_properties(d):
            if isinstance(d, dict):
                if 'properties' in d:
                    d['additionalProperties'] = False
                for k, v in d.items():
                    add_additional_properties(v)
        add_additional_properties(tool_info['input_schema'])
        for p in tool_info['input_schema']['properties'].keys():
            if not p in tool_info['input_schema']['required']:
                tool_info['input_schema']['required'].append(p)
                t = copy.deepcopy(tool_info['input_schema']['properties'][p]["type"])
                if isinstance(t, str):
                    tool_info['input_schema']['properties'][p]["type"] = [t, "null"]
                elif isinstance(t, list):
                    tool_info['input_schema']['properties'][p]["type"] = t + ["null"]

        return {
            'type': 'function',
            'name': tool_info['name'],
            'description': tool_info['description'],
            'parameters': tool_info['input_schema'],
            "strict": True,
        }
    else:
        return tool_info

def convert_block_claude(block):
    """
    Convert a single block of content from Claude into a standard format.
    """
    if isinstance(block, dict):
        block_type = block.get('type')
        text = block.get('text')
        tool_name = block.get('name')
        tool_input = block.get('input')
        tool_result = block.get('content')
    else:
        block_type = getattr(block, 'type', None)
        text = getattr(block, 'text', None)
        tool_name = getattr(block, 'name', None)
        tool_input = getattr(block, 'input', None)
        tool_result = getattr(block, 'content', None)

    text = text or ""

    if block_type == "text":
        return {
            "type": "text",
            "text": text
        }
    elif block_type == "tool_use":
        # Convert to the manual tool calling format
        return {
            "type": "text",
            "text": f"<tool_use>\n{{'tool_name': {tool_name}, 'tool_input': {tool_input}}}\n</tool_use>"
        }
    elif block_type == "tool_result":
        return {
            "type": "text",
            "text": f"Tool Result: {tool_result}"
        }
    else:
        # Fallback if we ever encounter an unknown block type
        return {
            "type": "text",
            "text": str(block)
        }

def convert_msg_history_claude(msg_history):
    """
    Convert Claude-style message history into a generic format.
    """
    new_msg_history = []

    for msg in msg_history:
        role = msg.get('role', '')
        content_blocks = msg.get('content', [])
        new_content = []

        for block in content_blocks:
            new_content.append(convert_block_claude(block))

        new_msg_history.append({
            "role": role,
            "content": new_content
        })

    return new_msg_history

def normalize_openai_content(content):
    if content is None:
        return [{"type": "text", "text": ""}]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        normalized_content = []
        for block in content:
            if isinstance(block, dict):
                text = block.get('text')
            else:
                text = getattr(block, 'text', None)
            if text is not None:
                normalized_content.append({
                    "type": "text",
                    "text": text,
                })
        if normalized_content:
            return normalized_content
    return [{"type": "text", "text": str(content)}]


def convert_msg_history_openai(msg_history):
    """
    Convert OpenAI-style message history into a generic format.
    """
    new_msg_history = []

    for msg in msg_history:
        if isinstance(msg, dict):
            role = msg.get('role', '')
            content = msg.get('content', '')

            if msg.get('type') == 'function_call_output':
                new_msg = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Tool Result: {msg.get('output', '')}",
                        }
                    ],
                }
            elif role == 'tool':
                new_msg = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Tool Result: {content}",
                        }
                    ],
                }
            else:
                new_msg = {
                    "role": role,
                    "content": normalize_openai_content(content),
                }
        else:
            item_type = getattr(msg, 'type', None)
            role = getattr(msg, 'role', None)
            content = getattr(msg, 'content', None)
            tool_calls = getattr(msg, 'tool_calls', None)

            if item_type == 'function_call':
                function_name = getattr(msg, 'name', '')
                function_args = getattr(msg, 'arguments', '')
                new_msg = {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"<tool_use>\n{{'tool_name': {function_name}, 'tool_input': {function_args}}}\n</tool_use>",
                        }
                    ],
                }
            elif item_type == 'message':
                new_msg = {
                    "role": role or "assistant",
                    "content": normalize_openai_content(content),
                }
            elif tool_calls:
                tool_call = tool_calls[0]
                function_name = getattr(tool_call.function, 'name', '')
                function_args = getattr(tool_call.function, 'arguments', '')
                # Convert to the manual tool calling format
                new_msg = {
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": f"<tool_use>\n{{'tool_name': {function_name}, 'tool_input': {function_args}}}\n</tool_use>",
                        }
                    ],
                }
            else:
                new_msg = {
                    "role": role,
                    "content": normalize_openai_content(content),
                }

        new_msg_history.append(new_msg)

    return new_msg_history

def convert_msg_history(msg_history, model=None):
    """
    Convert message history from the model-specific format to a generic format.
    """
    if 'claude' in model:
        return convert_msg_history_claude(msg_history)
    elif is_openai_tool_model(model):
        return convert_msg_history_openai(msg_history)
    else:
        return msg_history

def chat_with_agent_manualtools(msg, model, msg_history=None, logging=print, return_usage=False):
    # Construct message
    if msg_history is None:
        msg_history = []
    system_message = f'You are a coding agent.\n\n{get_tooluse_prompt()}'
    new_msg_history = msg_history

    total_usage = empty_usage()
    try:
        # Load all tools
        all_tools = load_all_tools(logging=logging)
        tools_dict = {tool['info']['name']: tool for tool in all_tools}
        
        # Create client
        client, client_model = create_client(model)

        # Call API
        logging(f"Input: {msg}")
        response, new_msg_history = get_response_from_llm(
            msg=msg,
            client=client,
            model=client_model,
            system_message=system_message,
            print_debug=False,
            msg_history=new_msg_history,
        )
        logging(f"Output: {response}")

        # Tool use
        tool_use = check_for_tool_use(response, model=client_model)
        while tool_use:
            # Process tool call
            tool_name = tool_use['tool_name']
            tool_input = tool_use['tool_input']
            tool_result = process_tool_call(tools_dict, tool_name, tool_input)

            # Get tool response
            tool_msg = f'Tool Used: {tool_name}\nTool Input: {tool_input}\nTool Result: {tool_result}'
            logging(tool_msg)
            response, new_msg_history = get_response_from_llm(
                msg=tool_msg,
                client=client,
                model=client_model,
                system_message=system_message,
                print_debug=False,
                msg_history=new_msg_history,
            )
            logging(f"Output: {response}")

            # Check for next tool use
            tool_use = check_for_tool_use(response, model=client_model)

    except Exception as e:
        logging(f"Error in chat_with_agent_manualtools: {str(e)}")
        raise

    if return_usage:
        return new_msg_history, total_usage
    return new_msg_history

def chat_with_agent_claude(
        msg,
        model='bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        msg_history=None,
        logging=print,
        return_usage=False,
    ):
    # Construct message
    if msg_history is None:
        msg_history = []
    new_msg_history = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": msg,
                }
            ],
        }
    ]

    total_usage = empty_usage()
    try:
        # Create client
        client, client_model = create_client(model)

        # Load all tools
        all_tools = load_all_tools(logging=logging)
        tools_dict = {tool['info']['name']: tool for tool in all_tools}
        tools = [convert_tool_info(tool['info'], model=client_model) for tool in all_tools]

        # Call API
        response = get_response_withtools(
            client=client,
            model=client_model,
            messages=msg_history + new_msg_history,
            tool_choice={"type": "auto"},
            tools=tools,
            logging=logging,
        )
        total_usage = merge_usage(total_usage, extract_response_usage(response))

        # Check for tool use
        tool_use = check_for_tool_use(response, model=client_model)
        while tool_use:
            # Process tool call
            tool_name = tool_use['tool_name']
            tool_input = tool_use['tool_input']
            tool_result = process_tool_call(tools_dict, tool_name, tool_input)

            # Get tool response
            new_msg_history.append({"role": "assistant", "content": response.content})
            new_msg_history.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use['tool_id'],
                        "content": tool_result,
                    }
                ],
            })
            response = get_response_withtools(
                client=client,
                model=client_model,
                messages=msg_history + new_msg_history,
                tool_choice={"type": "auto"},
                tools=tools,
                logging=logging,
            )
            total_usage = merge_usage(total_usage, extract_response_usage(response))

            # Check for next tool use
            tool_use = check_for_tool_use(response, model=client_model)

        # Get final response
        final_response = next((block.text for block in response.content if hasattr(block, "text")), None)
        new_msg_history.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": final_response,
                }
            ],
        })

    except Exception as e:
        logging(f"Error in chat_with_agent_claude: {str(e)}")
        raise

    if return_usage:
        return new_msg_history, total_usage
    return new_msg_history

def chat_with_agent_openai(
        msg,
        model=OPENAI_MODEL,
        msg_history=None,
        logging=print,
        return_usage=False,
    ):
    # Construct message
    if msg_history is None:
        msg_history = []
    new_msg_history = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": msg,
                }
            ],
        }
    ]
    separator = '=' * 10
    logging(f"\n{separator} User Instruction {separator}\n{msg}")
    total_usage = empty_usage()
    try:
        # Create client
        client, client_model = create_client(model)

        # Load all tools
        all_tools = load_all_tools(logging=logging)
        tools_dict = {tool['info']['name']: tool for tool in all_tools}
        tools = [convert_tool_info(tool['info'], model=client_model) for tool in all_tools]

        # Call API
        response = get_response_withtools(
            client=client,
            model=client_model,
            messages=msg_history + new_msg_history,
            tool_choice="auto",
            tools=tools,
            logging=logging,
        )
        total_usage = merge_usage(total_usage, extract_response_usage(response))
        logging(f"\n{separator} Agent Response {separator}\n{response}")

        # Check for tool use
        tool_use = check_for_tool_use(response, model=client_model)
        logging(tool_use)
        while tool_use:
            # Process tool call
            tool_name = tool_use['tool_name']
            tool_input = tool_use['tool_input']
            tool_result = process_tool_call(tools_dict, tool_name, tool_input)

            logging(f"Tool Used: {tool_name}")
            logging(f"Tool Input: {tool_input}")
            logging(f"Tool Result: {tool_result}")

            # Get tool response
            new_msg_history.extend(get_openai_response_items(response))
            new_msg_history.append({
                "type": "function_call_output",
                "id": f"{tool_use['tool_id']}_output",
                "call_id": tool_use['tool_id'],
                "output": serialize_tool_output(tool_result),
            })
            response = get_response_withtools(
                client=client,
                model=client_model,
                messages=msg_history + new_msg_history,
                tool_choice="auto",
                tools=tools,
                logging=logging,
            )
            total_usage = merge_usage(total_usage, extract_response_usage(response))

            # Check for next tool use
            tool_use = check_for_tool_use(response, model=client_model)

            logging(f"Tool Response: {response}")

        # Get final response
        new_msg_history.append({
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": extract_openai_response_text(response),
                }
            ],
        })

    except Exception as e:
        logging(f"Error in chat_with_agent_openai: {str(e)}")
        raise

    if return_usage:
        return new_msg_history, total_usage
    return new_msg_history

def chat_with_agent_minimax(
        msg,
        model=OPENAI_MODEL,
        msg_history=None,
        logging=print,
        return_usage=False,
    ):
    """
    MiniMax via OpenRouter: uses chat.completions with standard tool_calls format.
    Strips mandatory <think>...</think> tags from all content before storing.
    """
    if msg_history is None:
        msg_history = []

    system_message = f'You are a coding agent.\n\n{get_tooluse_prompt()}'
    new_msg_history = [{"role": "user", "content": msg}]

    # Build full messages list (system + history + new user turn)
    messages = [{"role": "system", "content": system_message}] + msg_history + new_msg_history

    total_usage = empty_usage()
    try:
        client, client_model = create_client(model)
        all_tools = load_all_tools(logging=logging)
        tools_dict = {tool['info']['name']: tool for tool in all_tools}
        tools = [convert_tool_info(tool['info'], model=client_model) for tool in all_tools]

        response = get_response_withtools(
            client=client,
            model=client_model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            logging=logging,
        )
        total_usage = merge_usage(total_usage, extract_response_usage(response))

        tool_use = check_for_tool_use(response, model=client_model)
        while tool_use:
            tool_name = tool_use['tool_name']
            tool_input = tool_use['tool_input']
            tool_result = process_tool_call(tools_dict, tool_name, tool_input)
            logging(f"Tool Used: {tool_name}\nTool Input: {tool_input}\nTool Result: {tool_result}")

            # Append full assistant message (with tool_calls) — required by MiniMax for reasoning chain
            raw_msg = response.choices[0].message
            assistant_msg = {
                "role": "assistant",
                "content": strip_thinking_tags(raw_msg.content or ""),
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in (raw_msg.tool_calls or [])
                ],
            }
            tool_result_msg = {
                "role": "tool",
                "tool_call_id": tool_use['tool_id'],
                "content": serialize_tool_output(tool_result),
            }
            messages.append(assistant_msg)
            messages.append(tool_result_msg)
            new_msg_history.append(assistant_msg)
            new_msg_history.append(tool_result_msg)

            response = get_response_withtools(
                client=client,
                model=client_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                logging=logging,
            )
            total_usage = merge_usage(total_usage, extract_response_usage(response))
            tool_use = check_for_tool_use(response, model=client_model)

        # Final assistant message (no tool calls)
        final_content = strip_thinking_tags(response.choices[0].message.content or "")
        final_msg = {"role": "assistant", "content": final_content}
        messages.append(final_msg)
        new_msg_history.append(final_msg)

    except Exception as e:
        logging(f"Error in chat_with_agent_minimax: {str(e)}")
        raise

    if return_usage:
        return new_msg_history, total_usage
    return new_msg_history


def chat_with_agent(
    msg,
    model=CLAUDE_MODEL,
    msg_history=None,
    logging=print,
    convert=False,  # Convert the message history to a generic format, so that msg_history can be used across models
    return_usage=False,
):
    if msg_history is None:
        msg_history = []

    if 'claude' in model:
        # Claude models
        new_msg_history = chat_with_agent_claude(
            msg,
            model=model,
            msg_history=msg_history,
            logging=logging,
            return_usage=return_usage,
        )
        usage = None
        if return_usage:
            new_msg_history, usage = new_msg_history
        conv_msg_history = convert_msg_history(new_msg_history, model=model)
        logging(conv_msg_history)
        if convert:
            new_msg_history = conv_msg_history
        new_msg_history = msg_history + new_msg_history

    elif is_minimax_model(model):
        # MiniMax models via OpenRouter (chat.completions with tool_calls)
        new_msg_history = chat_with_agent_minimax(
            msg,
            model=model,
            msg_history=msg_history,
            logging=logging,
            return_usage=return_usage,
        )
        usage = None
        if return_usage:
            new_msg_history, usage = new_msg_history
        new_msg_history = msg_history + new_msg_history

    elif is_openai_tool_model(model):
        # OpenAI models
        new_msg_history = chat_with_agent_openai(
            msg,
            model=model,
            msg_history=msg_history,
            logging=logging,
            return_usage=return_usage,
        )
        usage = None
        if return_usage:
            new_msg_history, usage = new_msg_history
        # Current version does not support cross-model conversion
        # new_msg_history = convert_msg_history(new_msg_history, model=model)
        new_msg_history = msg_history + new_msg_history

    else:
        # Models without in-built tool calling
        new_msg_history = chat_with_agent_manualtools(
            msg,
            model=model,
            msg_history=msg_history,
            logging=logging,
            return_usage=return_usage,
        )
        usage = None
        if return_usage:
            new_msg_history, usage = new_msg_history
        conv_msg_history = convert_msg_history(new_msg_history, model=model)
        if convert:
            new_msg_history = conv_msg_history

    if return_usage:
        return new_msg_history, usage or empty_usage()
    return new_msg_history


if __name__ == "__main__":
    # Test the tool calling functionality
    msg = "hello!"
    chat_with_agent(msg)
