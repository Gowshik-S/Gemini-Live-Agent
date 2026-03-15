import asyncio
from typing import Any

from orchestrator_core.base import AgentState, WorkerAgent
from orchestrator_core.pipelines import SequentialAgent, LoopAgent

# --- Mock Agent Definitions (In reality, these would be loaded from config/skills) ---

researcher_agent = WorkerAgent(
    name="Researcher",
    description="Finds relevant information and summarizes it.",
    system_instruction="You are an expert researcher. Given a topic, gather the most important facts. Do NOT output markdown, just plain text facts.",
)

judge_agent = WorkerAgent(
    name="Judge",
    description="Evaluates the quality of research.",
    system_instruction=(
        "You are a strict judge. Evaluate the input research. "
        "If it is comprehensive and clear, status is 'pass'. "
        "If it is vague or missing details, status is 'fail' and provide feedback."
    ),
    response_schema={
        "type": "object", 
        "properties": {
            "status": {"type": "string", "enum": ["pass", "fail"]}, 
            "feedback": {"type": "string"}
        },
        "required": ["status", "feedback"]
    }
)

writer_agent = WorkerAgent(
    name="Writer",
    description="Formats facts into a professional report.",
    system_instruction="You are a professional technical writer. Take the facts provided and turn them into a polished 3-paragraph report.",
)

# --- Define the Pipelines ---

# The "Research Loop": Researcher drafts -> Judge reviews. Repeats until Judge says 'pass'.
research_loop = LoopAgent(
    name="ResearchLoop",
    description="Researches and verifies facts until they pass quality control.",
    worker=researcher_agent,
    judge=judge_agent,
    max_iterations=3
)

# The "Research Report Pipeline": Run the Research Loop, then pass the verified facts to the Writer.
research_report_pipeline = SequentialAgent(
    name="ResearchReportPipeline",
    description="Generates a fully verified research report.",
    sequence=[research_loop, writer_agent]
)

# --- Registry Map for IntentRouter ---

PIPELINE_REGISTRY = {
    "research_report": research_report_pipeline,
    # "code_audit": code_audit_pipeline, # To be implemented
}

async def execute_pipeline(pipeline_id: str, goal: str, client: Any) -> str:
    """Executes a registered pipeline and returns the final output."""
    pipeline = PIPELINE_REGISTRY.get(pipeline_id)
    if not pipeline:
        raise ValueError(f"Pipeline '{pipeline_id}' not found in registry.")
        
    state = AgentState(
        task=goal,
        current_input=goal
    )
    
    final_state = await pipeline.execute(state, client)
    
    if final_state.metadata.get("escalation"):
        return f"[ESCALATION WARNING] Pipeline did not complete successfully.\n\n{final_state.current_input}"
        
    return final_state.current_input
