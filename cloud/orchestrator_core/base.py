import asyncio
import json
from typing import Any, Callable, Optional, Dict, List
import structlog
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

class AgentState(BaseModel):
    """Represents the shared state passed between agents in a pipeline."""
    task: str
    current_input: str
    history: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    is_finished: bool = False

class BaseAgent:
    """Base class for all orchestrator agents."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.log = logger.bind(agent=self.name)

    async def execute(self, state: AgentState, client: Any) -> AgentState:
        """Execute the agent's logic. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement execute()")

class WorkerAgent(BaseAgent):
    """A standard LLM-powered agent that performs a specific task."""
    
    def __init__(self, name: str, description: str, system_instruction: str, model: str = "gemini-3-flash-preview", tools: list = None, response_schema: Any = None):
        super().__init__(name, description)
        self.system_instruction = system_instruction
        self.model = model
        self.tools = tools or []
        self.response_schema = response_schema

    async def execute(self, state: AgentState, client: Any) -> AgentState:
        self.log.info("agent.execute.start", task=state.task)
        
        # Build prompt
        prompt = f"Task: {state.task}\n\nCurrent Input:\n{state.current_input}\n"
        
        try:
            from google.genai import types
            
            # Prepare config with strict schema if defined
            config_kwargs = {
                "system_instruction": self.system_instruction,
                "temperature": 0.2,
            }
            
            if self.response_schema:
                config_kwargs["response_mime_type"] = "application/json"
                config_kwargs["response_schema"] = self.response_schema
                
            if self.tools:
                config_kwargs["tools"] = self.tools
                
            config = types.GenerateContentConfig(**config_kwargs)
            
            # Execute
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config
            )
            
            result_text = response.text
            
            # If JSON schema was requested, ensure it's valid
            if self.response_schema:
                try:
                    parsed = json.loads(result_text)
                    # We store the raw text but it's guaranteed to be JSON
                except json.JSONDecodeError:
                    self.log.error("agent.execute.schema_error", result=result_text)
                    result_text = json.dumps({"error": "Failed to parse structured output", "raw": result_text})

            state.history.append({
                "agent": self.name,
                "input": state.current_input,
                "output": result_text
            })
            
            state.current_input = result_text
            self.log.info("agent.execute.complete")
            
        except Exception as e:
            self.log.error("agent.execute.failed", error=str(e))
            state.current_input = f"ERROR from {self.name}: {str(e)}"
            
        return state
