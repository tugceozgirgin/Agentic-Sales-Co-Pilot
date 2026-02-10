from src.agents.base_agent import BaseAgent
from src.agents.state import MainState, ReflectionOutput
from src.models.chat_models import Models
from src.agents.prompts.reflection_prompts import ReflectionPrompts
from typing import Dict, Any, List
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
import json
import re
from pydantic import ValidationError


class ReflectionAgent(BaseAgent):
    def __init__(self, name: str, tools: list, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        super().__init__(name, tools, model_name)
        self.llm = Models.get_openai_model("gpt-4o-mini", temperature=0.0)
        #self.llm = Models.get_vicuna_model(model_name, temperature=0.0)

    def forward(self, state: MainState) -> MainState:
        reflection_count = state.get("reflection_count", 0) + 1
        state["reflection_count"] = reflection_count
        
        print(f"[REFLECTION] Attempt {reflection_count}/3")
        
        if reflection_count >= 3:
            state["stop"] = True
            state["reflection_output"] = "Max reflection attempts reached."
            return state
        
        crm_output = state.get("crm_agent_output")
        if crm_output is None:
            state["stop"] = True
            state["reflection_output"] = "No CRM output to evaluate."
            return state
        
        tool_calls_made = self._extract_tool_calls(state.get("messages", []))
        
        prompt = ReflectionPrompts.build_reflection_prompt(
            user_input=state["input"],
            crm_output_message=crm_output.output_message,
            crm_rationale=crm_output.rationale,
            crm_irrelevant_query=crm_output.irrelevant_query,
            tool_calls_made=tool_calls_made
        )
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            
            reflection_output = self._parse_reflection_output(response_text, prompt)
            
            if reflection_output.correctness:
                state["stop"] = True
                state["reflection_output"] = reflection_output.rationale
            else:
                state["stop"] = False
                state["reflection_output"] = reflection_output.rationale
            
            return state
            
        except Exception as e:
            print(f"[REFLECTION] Error: {str(e)}")
            state["stop"] = True
            state["reflection_output"] = f"Reflection error: {str(e)}"
            return state
    
    def _extract_tool_calls(self, messages: List) -> List[str]:
        tool_names = []
        for msg in messages:
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_names.append(tc.get('name', 'unknown'))
        return tool_names
    
    def _parse_reflection_output(self, llm_output: str, prompt: str) -> ReflectionOutput:
        if not llm_output or not llm_output.strip():
            raise ValueError("Reflection LLM output is empty.")
        
        response_text = llm_output
        if "Your evaluation:" in llm_output:
            parts = llm_output.split("Your evaluation:")
            response_text = parts[-1].strip()
        
        json_match = re.search(r'\{[^{}]*"correctness"[^{}]*\}', response_text, re.IGNORECASE)
        if json_match:
            try:
                json_str = json_match.group()
                json_str = re.sub(r'\bTrue\b', 'true', json_str)
                json_str = re.sub(r'\bFalse\b', 'false', json_str)
                parsed_json = json.loads(json_str)
                return ReflectionOutput(**parsed_json)
            except (json.JSONDecodeError, ValidationError):
                pass
        
        correctness = None
        rationale = ""
        
        correctness_match = re.search(r'correctness[:\s]*(true|false|yes|no)', response_text, re.IGNORECASE)
        if correctness_match:
            val = correctness_match.group(1).lower()
            correctness = val in ('true', 'yes')
        
        rationale_match = re.search(r'rationale[:\s]*["\']?(.+?)(?:["\']?\s*$|\n|$)', response_text, re.IGNORECASE | re.DOTALL)
        if rationale_match:
            rationale = rationale_match.group(1).strip().strip('"\'')
        
        if correctness is not None:
            return ReflectionOutput(correctness=correctness, rationale=rationale or "Evaluation complete.")
        
        positive_indicators = ['correct', 'appropriate', 'proper', 'both tools', 'correctly']
        negative_indicators = ['incorrect', 'wrong', 'missing', 'should have', 'only used one']
        
        response_lower = response_text.lower()
        has_positive = any(ind in response_lower for ind in positive_indicators)
        has_negative = any(ind in response_lower for ind in negative_indicators)
        
        if has_negative and not has_positive:
            return ReflectionOutput(correctness=False, rationale=response_text[:200])
        elif has_positive:
            return ReflectionOutput(correctness=True, rationale=response_text[:200])
        
        return ReflectionOutput(correctness=True, rationale="Unable to parse response, defaulting to approved.")
