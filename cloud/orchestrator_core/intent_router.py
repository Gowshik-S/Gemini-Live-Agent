import asyncio
from typing import Any, Callable
import structlog
from pydantic import BaseModel

from .base import AgentState

logger = structlog.get_logger(__name__)

class IntentRouter:
    """
    Front-door router that classifies user requests into:
    1. A strict deterministic pipeline (Sequential/Loop)
    2. The dynamic orchestrator (ReAct)
    """

    def __init__(self, model: str = "gemini-3-flash-preview"):
        self.model = model
        self.log = logger.bind(component="intent_router")
        
        # In a real implementation, these would be loaded from a config or database
        self.registered_pipelines = {
            "research_report": "A deep research workflow generating a structured document.",
            "code_audit": "A comprehensive code review and refactoring loop."
        }

    async def route(self, user_request: str, client: Any) -> str:
        """
        Classifies the request. 
        Returns the ID of a registered pipeline, or "dynamic" if ad-hoc.
        """
        self.log.info("router.classifying", request=user_request[:50])
        
        prompt = (
            f"You are the routing layer for an AI assistant. Analyze the user request below.\n"
            f"If the request exactly matches one of the following registered strict pipelines, output ONLY the pipeline ID.\n"
            f"If it is a general, ad-hoc, or desktop control request, output ONLY the word 'dynamic'.\n\n"
            f"Pipelines:\n"
        )
        for p_id, desc in self.registered_pipelines.items():
            prompt += f"- {p_id}: {desc}\n"
            
        prompt += f"\nRequest: {user_request}\n"
        
        try:
            from google.genai import types
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=10
                )
            )
            
            decision = response.text.strip().lower()
            
            # Clean up potential markdown formatting
            if "dynamic" in decision:
                decision = "dynamic"
            else:
                for p_id in self.registered_pipelines:
                    if p_id in decision:
                        decision = p_id
                        break
                else:
                    decision = "dynamic" # Default fallback
                    
            self.log.info("router.decision", decision=decision)
            return decision
            
        except Exception as e:
            self.log.error("router.failed", error=str(e))
            return "dynamic" # Always fail open to the dynamic lane
