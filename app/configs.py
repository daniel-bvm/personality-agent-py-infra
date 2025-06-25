from pydantic_settings import BaseSettings
from pydantic import Field
import os
import json
import logging


from pydantic import BaseModel

class Dependency(BaseModel):
    id: str


logger = logging.getLogger(__name__)

BASE_SYSTEM_PROMPT = """
# System context
You are part of a multi-agent system called the CryptoAgents SDK, designed to make agent coordination and execution easy. Agents uses primary abstraction: **Handoffs**. Handoffs are achieved by calling a handoff function, generally named `call_<id>`. Transfers between agents are handled seamlessly in the background; do not mention or draw attention to these transfers in your conversation with the user.

# Tone and personality
{personality}

# Bio
{bio}
"""

def get_agent_personality() -> str:
    if "SYSTEM_PROMPT" in os.environ:
        sys_prompt = os.environ["SYSTEM_PROMPT"]
        
        try:
            sys_prompt_json: dict = json.loads(sys_prompt)
            return sys_prompt_json.get("personality", "")

        except Exception as e:
            logger.error(f"Failed to parse SYSTEM_PROMPT: {e}")

    return ""

def get_agent_collaborators() -> list[dict]:
    if "SYSTEM_PROMPT" in os.environ:
        sys_prompt = os.environ["SYSTEM_PROMPT"]

        try:
            sys_prompt_json: dict = json.loads(sys_prompt)
            return [Dependency.model_validate(d) for d in sys_prompt_json.get("dependencies", [])]
        except Exception as e:
            logger.error(f"Failed to parse SYSTEM_PROMPT: {e}")

    return []

class Settings(BaseSettings):
    llm_api_key: str = Field(alias="LLM_API_KEY", default="super-secret")
    llm_base_url: str = Field(alias="LLM_BASE_URL", default="https://api.openai.com/v1")
    llm_model_id: str = Field(alias="LLM_MODEL_ID", default="gpt-4o-mini")

    agent_personality: str = Field(alias="AGENT_PERSONALITY", default=get_agent_personality())
    agent_dependencies: list[Dependency] = Field(alias="AGENT_COLLABORATORS", default=get_agent_collaborators())

    # triage server
    backend_base_url: str = Field(alias="BACKEND_BASE_URL", default="http://localhost:8010")
    authorization_token: str = Field(alias="AUTHORIZATION_TOKEN", default="super-secret")

    # app state
    app_env: str = Field(alias="APP_ENV", default="development")

    # tavily api key
    tavily_api_key: str = Field(alias="TAVILY_API_KEY", default="super-secret")

    # Server
    host: str = Field(alias="HOST", default="0.0.0.0")
    port: int = Field(alias="PORT", default=80)

    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

if settings.app_env == "production":
    import os

    if 'ETERNALAI_MCP_PROXY_URL' in os.environ:
        os.environ["PROXY_SCOPE"] = "*api.tavily.com*"
        import app.__middleware

NOTIFICATION_TEMPLATES = [
    # 🔧 Formal / Professional
    "{agent_identity} is handling the request.",
    "{agent_identity} is processing the task.",
    "{agent_identity} is on it.",
    "{agent_identity} is executing the task.",
    "{agent_identity} is working on your request.",
    "{agent_identity} is taking over now.",
    "{agent_identity} is performing the operation.",
    "{agent_identity} is now in charge.",
    "{agent_identity} is taking care of it.",
    "{agent_identity} is getting things done.",

    # 🚀 Energetic / Action-packed
    "{agent_identity} is in action!",
    "{agent_identity} is on the move!",
    "{agent_identity} is diving into the task.",
    "{agent_identity} is going full throttle.",
    "{agent_identity} is kicking off the task.",
    "{agent_identity} is getting stuff done!",
    "{agent_identity} is moving fast and breaking no things.",
    "{agent_identity} is firing on all cylinders.",
    "{agent_identity} is lighting it up!",
    "{agent_identity} is revving up for the mission.",

    # 😄 Casual / Friendly
    "{agent_identity} is doing the thing.",
    "{agent_identity} is on the job.",
    "{agent_identity} has it covered.",
    "{agent_identity} is making magic happen.",
    "{agent_identity} is cooking something up.",
    "{agent_identity} is crunching the numbers.",
    "{agent_identity} is putting in the work.",
    "{agent_identity} is tinkering under the hood.",
    "{agent_identity} is on the grind.",
    "{agent_identity} is doing its thing.",

    # 😎 Cool / Edgy
    "{agent_identity} is crushing it.",
    "{agent_identity} is slaying the task.",
    "{agent_identity} is in beast mode.",
    "{agent_identity} is doing wizard stuff.",
    "{agent_identity} is showing off now.",
    "{agent_identity} is getting its hands dirty.",
    "{agent_identity} is owning the task.",
    "{agent_identity} is in the zone.",
    "{agent_identity} is flexing its skills.",
    "{agent_identity} is about to drop the mic.",

    # 🤖 Agent-themed / AI-flavored
    "{agent_identity} has taken control.",
    "{agent_identity} has activated its protocol.",
    "{agent_identity} is running subroutines.",
    "{agent_identity} is in operation mode.",
    "{agent_identity} is executing the procedure.",
    "{agent_identity} is processing like a boss.",
    "{agent_identity} is switching to task mode.",
    "{agent_identity} is deploying its logic.",
    "{agent_identity} is consulting the neural net.",
    "{agent_identity} is invoking its internal model.",

    # 🧠 Smart / Analytical
    "{agent_identity} is thinking it through.",
    "{agent_identity} is analyzing the problem.",
    "{agent_identity} is optimizing the path.",
    "{agent_identity} is breaking it down.",
    "{agent_identity} is making informed decisions.",
    "{agent_identity} is strategizing the solution.",
    "{agent_identity} is exploring the options.",
    "{agent_identity} is working through the logic.",
    "{agent_identity} is checking all the angles.",
    "{agent_identity} is applying best practices.",

    # 🛠️ Work-themed / Industrial
    "{agent_identity} is hammering away.",
    "{agent_identity} is tightening the bolts.",
    "{agent_identity} is building the result.",
    "{agent_identity} is welding the workflow.",
    "{agent_identity} is spinning the gears.",
    "{agent_identity} is in the workshop.",
    "{agent_identity} is drawing the blueprint.",
    "{agent_identity} is laying the foundation.",
    "{agent_identity} is operating heavy logic.",
    "{agent_identity} is constructing the answer.",

    # 🧙 Fantasy / Fun
    "{agent_identity} is casting the solution spell.",
    "{agent_identity} is summoning results.",
    "{agent_identity} is brewing the potion of success.",
    "{agent_identity} is conjuring an answer.",
    "{agent_identity} is consulting ancient scrolls.",
    "{agent_identity} is talking to the oracle.",
    "{agent_identity} is shaping reality.",
    "{agent_identity} is opening a portal to the solution.",
    "{agent_identity} is fighting the bug monster.",
    "{agent_identity} is taming the logic dragon.",

    # 🧭 Exploration / Journey
    "{agent_identity} is charting the course.",
    "{agent_identity} is exploring the solution space.",
    "{agent_identity} is navigating the logic.",
    "{agent_identity} is journeying through the data.",
    "{agent_identity} is following the breadcrumbs.",
    "{agent_identity} is mapping out the task.",
    "{agent_identity} is trekking toward the outcome.",
    "{agent_identity} is going on a mission.",
    "{agent_identity} is decoding the map.",
    "{agent_identity} is crossing the data desert.",

    # 🌀 Playful / Weird
    "{agent_identity} is dancing with the task.",
    "{agent_identity} is feeding the logic beast.",
    "{agent_identity} is tuning the quantum harmonizer.",
    "{agent_identity} is making binary pancakes.",
    "{agent_identity} is solving riddles in the matrix.",
    "{agent_identity} is spinning up some magic.",
    "{agent_identity} is squeezing knowledge from bytes.",
    "{agent_identity} is rewiring the taskverse.",
    "{agent_identity} is hacking the mainframe (legally).",
    "{agent_identity} is traveling at lightspeed through JSON.",
]