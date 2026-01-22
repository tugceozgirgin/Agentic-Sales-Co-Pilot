from typing import TypedDict, Any, List, Annotated, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel


class CRMAgentOutput(BaseModel):
    output_message: str
    rationale: str
    irrelevant_query: bool


class ReflectionOutput(BaseModel):
    correctness: bool
    rationale: str


class MainState(TypedDict):
    input: str
    output: str
    stop: bool
    reflection_count: int
    last_tool_calls: Any
    reflection_output: str
    messages: Annotated[List[BaseMessage], add_messages]
    crm_agent_output: Optional[CRMAgentOutput]
