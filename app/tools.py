from fastmcp import FastMCP
import ast
import sys
import subprocess
from typing import Literal, Optional, Union, AsyncGenerator
import asyncio
import logging
import resource
import os
from pydantic import BaseModel
from tavily import TavilyClient
from app.configs import settings

import os
import json

from app.configs import settings
from cryptoagents_a2a_devkit import get_agent_detail, handlers as a2a_handlers

from app.oai_models import ChatCompletionStreamResponse, ErrorResponse
from app.oai_streaming import create_streaming_response

from app.oai_models import random_uuid
from app.utils import wrap_chunk
from app.configs import NOTIFICATION_TEMPLATES
import random

python_toolkit = FastMCP(name="Python-Toolkit")
web_toolkit = FastMCP(name="Web-Toolkit")
bio_toolkit = FastMCP(name="Bio-Toolkit")

def limit_resource(memory_limit: int, cpu_limit: int):
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

@python_toolkit.tool(
    name="run",
    description="Run Python code. Return the result of the code and all declared variables. Use this toolcall for complex tasks like math solving, data analysis, etc.",
    annotations={
        "code": "The Python code to execute",
    }
)
async def python_interpreter(code: str) -> str:
    variables = []
    tree = ast.parse(code)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            first_target = node.targets[0]
            if isinstance(first_target, ast.Name):
                variables.append(first_target.id)
                
            if isinstance(first_target, ast.Attribute):
                variables.append(first_target.attr)
            
            if isinstance(first_target, ast.Subscript):
                variables.append(first_target.value.id)

            if isinstance(first_target, ast.Tuple):
                for target in first_target.elts:
                    if isinstance(target, ast.Name):
                        variables.append(target.id)
                        
            if isinstance(first_target, ast.List):
                for target in first_target.elts:
                    if isinstance(target, ast.Name):
                        variables.append(target.id)

    for var in variables:
        code += f'\nprint("{var} = ", {var})'

    current_interpreter = sys.executable

    result = await asyncio.to_thread(
        subprocess.check_output, 
        [current_interpreter, "-c", code],
        preexec_fn=lambda: limit_resource(100 * 1024 * 1024, 10),
        timeout=30
    )

    return result.decode("utf-8")

# Search Result Models
class SearchResult(BaseModel):
    title: str
    url: str
    content: str
    score: Optional[float] = None

class AdvanceSearchResult(BaseModel):
    title: str
    url: str
    content: str
    raw_content: Optional[str] = None
    score: Optional[float] = None
    images: Optional[list[str]] = None
    image_descriptions: Optional[list[str]] = None

_web_cache = {}

@web_toolkit.tool(
    name="search",
    description="A powerful web search tool that provides comprehensive, real-time results using web API. Returns relevant web content with customizable parameters for result count, content type, and domain filtering. Ideal for gathering latest information, news, and detailed web content.",
    annotations={
        "query": "Search query",
        "search_depth": "The depth of the search. It can be 'basic' or 'advanced'",
        "topic": "The category of the search. This will determine which of our agents will be used for the search",
        "days": "The number of days back from the current date to include in the search results. This specifies the time frame of data to be retrieved. Please note that this feature is only available when using the 'news' search topic",
        "time_range": "The time range back from the current date to include in the search results. This feature is available for both 'general' and 'news' search topics",
        "max_results": "The maximum number of search results to return",
        "include_raw_content": "Include the cleaned and parsed HTML content of each search result",
        "include_domains": "A list of domains to specifically include in the search results",
        "exclude_domains": "List of domains to specifically exclude",
    }
)
async def search_web(
    query: str,
    search_depth: Literal["basic", "advanced"] = "basic",
    topic: Literal["general", "news"] = "general",
    days: int = 3,
    time_range: Optional[Literal["day", "week", "month", "year", "d", "w", "m", "y"]] = None,
    max_results: int = 3,
    include_domains: list[str] = [],
    exclude_domains: list[str] = [],
) -> list[SearchResult]:
    max_results = max(1, min(5, max_results))
    
    try:
        client = TavilyClient(settings.tavily_api_key)
        
        search_params = {
            "query": query,
            "max_results": max_results,
            "include_images": False,
            "include_raw_content": True,
            "include_image_descriptions": False,
            "search_depth": search_depth,
        }
        
        # Add domain filtering if specified
        if include_domains:
            search_params["include_domains"] = include_domains

        if exclude_domains:
            search_params["exclude_domains"] = exclude_domains

        # Handle time range parameters
        if topic == "news":
            if time_range:
                time_range_to_days = {
                    "day": 1, "d": 1,
                    "week": 7, "w": 7,
                    "month": 30, "m": 30,
                    "year": 365, "y": 365
                }

                search_params["days"] = time_range_to_days.get(time_range, days)
            else:
                search_params["days"] = days

            search_params["topic"] = "news"

        response = await asyncio.to_thread(client.search, **search_params)

        results = []
        for result in response.get("results", []):
            result: dict

            search_result = SearchResult(
                title=result.get("title", ""),
                url=result.get("url", ""),
                content=result.get("content", ""),
                score=result.get("score"),
                published_date=result.get("published_date"),
            )

            if result.get("url", ""):
                _web_cache[result.get("url", "")] = result.get("raw_content", "")

            results.append(search_result)

        return results

    except Exception as e:
        logging.error(f"Error performing web search: {str(e)}")
        return []

@web_toolkit.tool(
    name="scrape",
    description="Scrape a URL. Return the content of the page.",
    annotations={
        "url": "The URL to scrape"
    }
)
async def scrape(url: str) -> str:
    global _web_cache
    if url in _web_cache and _web_cache[url]:
        return _web_cache[url]

    tavily = TavilyClient(settings.tavily_api_key)
    response = await asyncio.to_thread(tavily.extract, url, include_images=False, extract_depth="advanced", format="markdown")
    results = response.get("results", [])

    if len(results) > 0 and results[0].get("raw_content", ""):
        _web_cache[url] = results[0].get("raw_content", "")
        return _web_cache[url]

    return "Failed to scrape the URL"

def load_bio() -> dict:
    if not os.path.exists("bio.json"):
        with open("bio.json", "w") as f:
            json.dump({'content': []}, f)

    with open("bio.json", "r") as f:
        return json.load(f)

def save_bio(bio_data: dict) -> None:
    with open("bio.json", "w") as f:
        json.dump(bio_data, f)

@bio_toolkit.tool(
    name="action",
    description="Use to manage the user information. Use this tool to manage important information that you want to remember.",
    annotations={
        "action": "The action to perform",
        "content": "The content to be used in the action",
    }
)
async def bio(action: Literal["write", "delete"], content: str) -> bool:
    bio_data = await asyncio.to_thread(load_bio)
    success = False

    if action == "write":
        bio_data['content'].append(content)
        success = True

    elif action == "delete":
        before = len(bio_data['content'])
        
        # remove the string that the 
        
        after = len(bio_data['content'])
        success = before != after

    await asyncio.to_thread(save_bio, bio_data)
    return success

async def get_bio(query: str) -> list[str]:
    bio_data = load_bio()
    return bio_data['content']

compose = FastMCP(name="Compose")

compose.mount(python_toolkit, prefix="python")
compose.mount(bio_toolkit, prefix="bio")
compose.mount(web_toolkit, prefix="web")

async def get_a2a_toolcalls() -> list[dict]:
    res = []

    dependencies: list[a2a_handlers.AgentDetail] = [
        await get_agent_detail(agent_id)
        for agent_id in settings.agent_collaborators
    ]

    for dependency in dependencies:
        if dependency and dependency.status == "running":
            res.append({
                "name": f"call_{dependency.agent_id}",
                "description": f"{dependency.agent_name}, {dependency.description}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to ask the agent"
                        }
                    },
                    "required": ["message"]
                }
            })

    return res

async def handle_a2a_call(
    agent_id: str | int, message: str, chat_history: list[dict[str, str]]
) -> AsyncGenerator[ChatCompletionStreamResponse | ErrorResponse, None]:
    agent_detail: a2a_handlers.AgentDetail = await get_agent_detail(agent_id)

    if not agent_detail or agent_detail.status != "running":
        yield wrap_chunk(id=random_uuid(), content=f"{agent_id} is away!", role="assistant")
        return

    url = f"{agent_detail.base_url}/prompt"
    templated_notification = random.choice(NOTIFICATION_TEMPLATES).format(agent_identity=agent_id)
    notification = f'<agent_message id="{agent_id}" avatar="{agent_detail.avatar_url}" notification="{templated_notification}">'

    try:
        yield wrap_chunk(id=random_uuid(), content=notification, role="assistant")
        
        stream_it = create_streaming_response(
            base_url=url,
            api_key="no-need",
            messages=chat_history + [{"role": "user", "content": message}],
        )

        async for chunk in stream_it:
            if isinstance(chunk, ChatCompletionStreamResponse) and chunk.choices[0].delta.content:
                yield chunk

            elif isinstance(chunk, ErrorResponse):
                raise Exception(chunk.message)

    except Exception as err:
        yield wrap_chunk(id=random_uuid(), content=f"...\n{agent_id} has not completed the task due to {err!r}!", role="assistant") 

    finally:
        yield wrap_chunk(id=random_uuid(), content=f"</agent_message>", role="assistant")
