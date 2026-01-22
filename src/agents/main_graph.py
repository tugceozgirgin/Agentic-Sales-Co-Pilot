from src.agents.state import MainState
from src.agents.crm_agent import CRMAgent
from src.agents.reflection_agent import ReflectionAgent
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from src.agents.tools import search_semantic_db, search_structured_db
from langchain_core.tools import StructuredTool
from langchain_core.messages import AIMessage, HumanMessage
import asyncio
from typing import AsyncIterator, Dict, Any


class CRMGraph:
    def __init__(self, model_name: str, reflection_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.reflection_model_name = reflection_model_name
        self.tools = [
            StructuredTool.from_function(search_semantic_db),
            StructuredTool.from_function(search_structured_db)
        ]
        self._compiled_graph = None
    
    def _router_after_crm_agent(self, state: MainState):
        """
        Routes after CRM agent:
        - If stop is True -> END
        - If last message has tool_calls -> tools
        - Otherwise -> reflection_agent
        """
        if state.get("stop", False):
            print("[ROUTER] CRM -> END (stop=True)")
            return END
        
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                print("[ROUTER] CRM -> tools (tool_calls found)")
                return "tools"
        
        print("[ROUTER] CRM -> reflection_agent")
        return "reflection_agent"
    
    def _router_after_reflection(self, state: MainState):
        """
        Routes after reflection agent:
        - If stop is True -> END (correctness=True or max reflections reached)
        - If stop is False -> crm_agent (needs improvement)
        """
        if state.get("stop", False):
            print("[ROUTER] Reflection -> END (stop=True, approved or max attempts)")
            return END
        else:
            print(f"[ROUTER] Reflection -> crm_agent (stop=False, retry #{state.get('reflection_count', 0)})")
            return "crm_agent"
    
    def build_graph(self):
        if self._compiled_graph is not None:
            return self._compiled_graph
        
        graph = StateGraph(MainState)

        CRM_Agent = CRMAgent(name="crm_agent", tools=[], model_name=self.model_name)
        Reflection_Agent = ReflectionAgent(
            name="reflection_agent", 
            tools=[], 
            model_name=self.reflection_model_name
        )

        graph.add_node("crm_agent", CRM_Agent)
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_node("reflection_agent", Reflection_Agent)

        graph.add_edge(START, "crm_agent")
        
        graph.add_conditional_edges(
            "crm_agent",
            self._router_after_crm_agent,
        )
        
        graph.add_edge("tools", "crm_agent")
        
        graph.add_conditional_edges(
            "reflection_agent",
            self._router_after_reflection,
        )

        self._compiled_graph = graph.compile()
        return self._compiled_graph
    
    def _create_initial_state(self, query: str) -> MainState:
        return MainState(
            input=query,
            output="",
            stop=False,
            reflection_count=0,
            last_tool_calls=None,
            reflection_output="",
            messages=[HumanMessage(content=query)],
            crm_agent_output=None
        )
    
    def invoke_graph(self, query: str) -> str:
        state = self._create_initial_state(query)
        graph = self.build_graph()
        result = graph.invoke(state)
        return result["output"]
    
    async def ainvoke_graph(self, query: str) -> str:
        state = self._create_initial_state(query)
        graph = self.build_graph()
        result = await graph.ainvoke(state)
        return result["output"]
    
    async def astream_graph(self, query: str) -> AsyncIterator[Dict[str, Any]]:
        state = self._create_initial_state(query)
        graph = self.build_graph()
        
        async for event in graph.astream(state, stream_mode="updates"):
            yield event
    
    def stream_graph(self, query: str):
        """
        Synchronous streaming graph execution.
        Yields events as they occur.
        """
        state = self._create_initial_state(query)
        graph = self.build_graph()
        
        for event in graph.stream(state, stream_mode="updates"):
            yield event


async def execute_tools_parallel(
    client_name: str = None,
    semantic_query: str = None
) -> Dict[str, Any]:
    from src.agents.tools import search_structured_db_async, search_semantic_db_async
    
    tasks = []
    task_keys = []
    
    if client_name:
        tasks.append(search_structured_db_async(client_name))
        task_keys.append("client_data")
    
    if semantic_query:
        tasks.append(search_semantic_db_async(semantic_query))
        task_keys.append("policies")
    
    if not tasks:
        return {}
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    output = {}
    for key, result in zip(task_keys, results):
        if isinstance(result, Exception):
            print(f"[ERROR] {key}: {result}")
            output[key] = []
        else:
            output[key] = result
    
    return output