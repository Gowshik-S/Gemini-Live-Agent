import asyncio
import json
from typing import Any, List
from pydantic import BaseModel

from .base import BaseAgent, AgentState

class SequentialAgent(BaseAgent):
    """Executes a series of agents in a strict sequence (DAG)."""

    def __init__(self, name: str, description: str, sequence: List[BaseAgent]):
        super().__init__(name, description)
        self.sequence = sequence

    async def execute(self, state: AgentState, client: Any) -> AgentState:
        self.log.info("sequential.start", agents=[a.name for a in self.sequence])
        
        for agent in self.sequence:
            if state.is_finished:
                break
            
            self.log.info("sequential.handoff", to=agent.name)
            state = await agent.execute(state, client)
            
        self.log.info("sequential.complete")
        return state

class LoopAgent(BaseAgent):
    """Executes an Actor-Critic loop until a condition is met or max iterations reached."""
    
    def __init__(self, name: str, description: str, worker: BaseAgent, judge: BaseAgent, max_iterations: int = 3):
        super().__init__(name, description)
        self.worker = worker
        self.judge = judge
        self.max_iterations = max_iterations

    async def execute(self, state: AgentState, client: Any) -> AgentState:
        self.log.info("loop.start", worker=self.worker.name, judge=self.judge.name, max_iter=self.max_iterations)
        
        for iteration in range(self.max_iterations):
            self.log.info("loop.iteration", iteration=iteration+1)
            
            # 1. Worker acts
            state = await self.worker.execute(state, client)
            
            # 2. Judge evaluates
            # We preserve the worker's output as the input to the judge
            worker_output = state.current_input
            state = await self.judge.execute(state, client)
            
            # 3. Parse Judge's verdict (expected to be Pydantic JSON)
            try:
                verdict = json.loads(state.current_input)
                status = verdict.get("status", "fail").lower()
                feedback = verdict.get("feedback", "No feedback provided.")
                
                if status == "pass":
                    self.log.info("loop.passed", iteration=iteration+1)
                    # Restore the worker's approved output as the final output of this loop
                    state.current_input = worker_output
                    break
                else:
                    self.log.info("loop.failed", feedback=feedback)
                    # Feed the judge's feedback back into the worker for the next iteration
                    state.current_input = f"PREVIOUS ATTEMPT FAILED.\nFeedback: {feedback}\nPlease revise and try again."
            except json.JSONDecodeError:
                self.log.error("loop.judge_invalid_json", raw=state.current_input)
                state.current_input = "Judge returned invalid format. Please try again."
                
        else:
            self.log.warning("loop.max_iterations_reached", max=self.max_iterations)
            # Circuit breaker tripped. Escalate or mark as degraded.
            state.metadata["escalation"] = True
            state.current_input = f"[ESCALATION] Task failed quality checks after {self.max_iterations} attempts.\nLast Output: {worker_output}"
            
        return state
