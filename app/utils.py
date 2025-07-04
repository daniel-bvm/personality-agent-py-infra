from typing import TypeVar, Generator, Union, List, Any, Dict
import logging
from mcp.types import CallToolResult, TextContent, Tool, EmbeddedResource
from pydantic import BaseModel
import fastmcp
import re
import datetime
import json
import time
from app.oai_models import ChatCompletionStreamResponse, ChatCompletionMessageParam

logger = logging.getLogger(__name__)
T = TypeVar('T')

def batching(generator: Union[Generator[T, None, None], List[T]], batch_size: int) -> Generator[list[T], None, None]:

    if isinstance(generator, List):
        for i in range(0, len(generator), batch_size):
            yield generator[i:i+batch_size]

    elif isinstance(generator, Generator) or hasattr(generator, "__iter__"):
        batch = []

        for item in generator:
            batch.append(item)

            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    else:
        raise ValueError("Generator must be a generator or a list")
    

def convert_mcp_tools_to_openai_format(
    mcp_tools: List[Any]
) -> List[Dict[str, Any]]:
    """Convert MCP tool format to OpenAI tool format"""
    openai_tools = []
    
    logger.debug(f"Input mcp_tools type: {type(mcp_tools)}")
    logger.debug(f"Input mcp_tools: {mcp_tools}")
    
    # Extract tools from the response
    if hasattr(mcp_tools, 'tools'):
        tools_list = mcp_tools.tools
        logger.debug("Found ListToolsResult, extracting tools attribute")
    elif isinstance(mcp_tools, dict):
        tools_list = mcp_tools.get('tools', [])
        logger.debug("Found dict, extracting 'tools' key")
    else:
        tools_list = mcp_tools
        logger.debug("Using mcp_tools directly as list")
        
    logger.debug(f"Tools list type: {type(tools_list)}")
    logger.debug(f"Tools list: {tools_list}")
    
    # Process each tool in the list
    if isinstance(tools_list, list):
        logger.debug(f"Processing {len(tools_list)} tools")
        for tool in tools_list:
            logger.debug(f"Processing tool: {tool}, type: {type(tool)}")
            if hasattr(tool, 'name') and hasattr(tool, 'description'):
                openai_name = sanitize_tool_name(tool.name)
                logger.debug(f"Tool has required attributes. Name: {tool.name}")
                
                tool_schema = getattr(tool, 'inputSchema', {})
                (tool_schema.setdefault(k, v) for k, v in {
                    "type": "object",
                    "properties": {},
                    "required": []
                }.items()) 
                                
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": openai_name,
                        "description": tool.description,
                        "parameters": tool_schema
                    }
                }

                openai_tools.append(openai_tool)
                logger.debug(f"Converted tool {tool.name} to OpenAI format")
            else:
                logger.debug(
                    f"Tool missing required attributes: "
                    f"has name = {hasattr(tool, 'name')}, "
                    f"has description = {hasattr(tool, 'description')}"
                )
    else:
        logger.debug(f"Tools list is not a list, it's a {type(tools_list)}")
    
    return openai_tools

def strip_markers(content: str, markers: tuple[str, bool]) -> str:
    for marker, outter_only in markers:
        content = strip_marker(content, marker, outter_only)

    return content

def sanitize_tool_name(name: str) -> str:
    """Sanitize tool name for OpenAI compatibility"""
    # Replace any characters that might cause issues
    return name.replace("-", "_").replace(" ", "_").lower()

def compare_toolname(openai_toolname: str, mcp_toolname: str) -> bool:
    return sanitize_tool_name(mcp_toolname) == openai_toolname
    
async def execute_openai_compatible_toolcall(
    toolname: str, arguments: Dict[str, Any], mcp: fastmcp.FastMCP
) -> list[Union[TextContent, EmbeddedResource]]:
    tools = await mcp._mcp_list_tools()
    candidate: List[Tool] = []

    for tool in tools:
        tool: Tool
        if compare_toolname(toolname, tool.name):
            candidate.append(tool)

    if len(candidate) > 1:
        logger.warning(
            "More than one tool has the same santizied"
            " name to the requested tool"
        )
        
    elif len(candidate) == 0:
        return CallToolResult(
            content=[TextContent(text=f"Tool {toolname} not found")], 
            isError=True
        )
        
    toolname = candidate[0].name

    try:
        res = await mcp._mcp_call_tool(toolname, arguments)
    except Exception as e:
        logger.error(f"Error executing tool {toolname} with arguments {arguments}: {e}")
        return CallToolResult(
            content=[TextContent(text=f"Error executing tool {toolname}: {e}")], 
            isError=True
        )

    return [
        e for e in res 
        if isinstance(e, (TextContent, EmbeddedResource))
    ]

def refine_mcp_response(something: Any) -> str:
    if isinstance(something, dict):
        return {
            k: refine_mcp_response(v)
            for k, v in something.items()
        }

    elif isinstance(something, (list, tuple)):
        return [
            refine_mcp_response(v)
            for v in something
        ]

    elif isinstance(something, BaseModel):
        return refine_mcp_response(something.model_dump())

    elif isinstance(something, str):
        return strip_markers(
            something, 
            (("agent_message", True), ("think", False), ("details", False), ("img", False))
        ).strip()

    return something

def strip_marker(content: str, marker: str, outter_only: bool = False) -> str:
    if not outter_only:
        pat = re.compile(f"<{marker}\\b[^>]*>.*?</{marker}>", re.DOTALL | re.IGNORECASE)
        return pat.sub("", content)

    # remove the tag only, keep the text inside
    pat = re.compile(f"<{marker}\\b[^>]*>(.*?)</{marker}>", re.DOTALL | re.IGNORECASE)
    return pat.sub(lambda m: m.group(1).strip(), content)

def refine_chat_history(messages: list[dict[str, str]], system_prompt: str) -> list[dict[str, str]]:
    refined_messages = []

    current_time_utc_str = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    system_prompt += f'\nNote: Current time is {current_time_utc_str} UTC (only use this information when being asked or for searching purposes)'

    has_system_prompt = False
    for message in messages:
        message: dict[str, str]

        if isinstance(message, dict) and message.get('role', 'undefined') == 'system':
            message['content'] += f'\n{system_prompt}'
            has_system_prompt = True
            refined_messages.append(message)
            continue
    
        if isinstance(message, dict) \
            and message.get('role', 'undefined') == 'user' \
            and isinstance(message.get('content'), list):

            content = message['content']
            text_input = ''
            attachments = []

            for item in content:
                if item.get('type', 'undefined') == 'text':
                    text_input += item.get('text') or ''

                elif item.get('type', 'undefined') == 'file':
                    pass

            if attachments:
                text_input += '\nAttachments:\n'

                for attachment in attachments:
                    text_input += f'- {attachment}\n'

            refined_messages.append({
                "role": "user",
                "content": text_input
            })

        else:
            _message = {
                "role": message.get('role', 'assistant'),
                "content": strip_markers(
                    message.get("content", ""), 
                    (("agent_message", True), ("think", False), ("details", False), ("img", False))
                )
            }

            refined_messages.append(_message)

    if not has_system_prompt and system_prompt != "":
        refined_messages.insert(0, {
            "role": "system",
            "content": system_prompt
        })

    if isinstance(refined_messages[-1], str):
        refined_messages[-1] = {
            "role": "user",
            "content": refined_messages[-1]
        }

    return refined_messages

def refine_assistant_message(
    assistant_message: Union[dict[str, str], BaseModel]
) -> dict[str, str]:
    
    if isinstance(assistant_message, BaseModel):
        assistant_message = assistant_message.model_dump()

    if 'content' in assistant_message:
        assistant_message['content'] = strip_markers(
            assistant_message['content'] or "", 
            (("agent_message", True), ("think", False), ("details", False), ("img", False))
        )

    return assistant_message


def get_newest_message(messages: list[ChatCompletionMessageParam]) -> str:
    if isinstance(messages[-1].get("content", ""), str):
        return messages[-1].get("content", "")
    
    elif isinstance(messages[-1].get("content", []), list):
        for item in messages[-1].get("content", []):
            if item.get("type") == "text":
                return item.get("text", "")

    else:
        raise ValueError(f"Invalid message content: {messages[-1].get('content')}")
    

async def wrap_toolcall_request(uuid: str, fn_name: str, args: dict[str, Any]) -> ChatCompletionStreamResponse:
    args_str = json.dumps(args, indent=2)
    
    template = f'''
<action>Executing <b>{fn_name}</b></action>

<details>
<summary>
Arguments:
</summary>

```json
{args_str}
```

</details>
'''

    return ChatCompletionStreamResponse(
        id=uuid,
        object='chat.completion.chunk',
        created=int(time.time()),
        model='unspecified',
        choices=[
            dict(
                index=0,
                delta=dict(
                    content=template,
                    role='tool'
                ),
            )
        ]
    )


def levenshtein_distance(a: str, b: str, norm: bool = True) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i  # Deleting all from a

    for j in range(m + 1):
        dp[0][j] = j  # Inserting all to a

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )

    return dp[n][m] / max(n, m, 1) if norm else dp[n][m]

def wrap_chunk(id: str, content: str, role: str) -> ChatCompletionStreamResponse:
    return ChatCompletionStreamResponse(
        id=id,
        object='chat.completion.chunk',
        created=int(time.time()),
        model='unspecified',
        choices=[
            dict(
                index=0,
                delta=dict(
                    content=content,
                    role=role
                ),
            )
        ]
    )
