from src.agents.base_agent import BaseAgent
from src.agents.state import MainState, CRMAgentOutput
from typing import Dict, Any, Union
from src.models.chat_models import Models
from src.agents.tools import search_semantic_db, search_structured_db
from src.agents.prompts.crm_prompts import CRMPrompts
from langchain_core.tools import StructuredTool
from langchain_core.messages import  AIMessage, SystemMessage
import json
import re
from pydantic import ValidationError


class CRMAgent(BaseAgent):
    def __init__(self, name: str, tools: list, model_name: str):
        super().__init__(name, tools, model_name)
        self.llm = Models.get_openai_model(model_name, temperature=0.0)

        self.tools = [
            StructuredTool.from_function(search_semantic_db),
            StructuredTool.from_function(search_structured_db)
        ]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def forward(self, state: MainState) -> MainState:
        messages = list(state.get("messages", []))
        has_system_message = any(isinstance(m, SystemMessage) for m in messages)
        if not has_system_message:
            user_input = state["input"]
            reflection_output = state.get("reflection_output", "")
            
            prompt_messages = CRMPrompts.build_crm_prompt(user_input, reflection_output)
            
            system_msg = prompt_messages[0]
            
            messages = [system_msg] + messages
        
        try:    
            response = self.llm_with_tools.invoke(messages)
            
            new_messages = [response]
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                state["stop"] = False
            else:
                crm_agent_output = self.parse_crm_agent_output(response.content)
                state["crm_agent_output"] = crm_agent_output
                state["output"] = crm_agent_output.output_message
                state["stop"] = False
            
            state["messages"] = new_messages
            return state
        
        except Exception as e:
            state["output"] = f"An error occurred: {str(e)}"
            state["stop"] = True
            return state
    
    def parse_crm_agent_output(self, llm_output: Union[str, AIMessage]) -> CRMAgentOutput:
        """
        Parse the CRM agent LLM output into a CRMAgentOutput object.
        Raises a clear error if parsing or validation fails.
        """
        if isinstance(llm_output, AIMessage):
            raw_text = llm_output.content
        else:
            raw_text = llm_output

        if not raw_text or not raw_text.strip():
            raise ValueError("LLM output is empty.")

        cleaned = re.sub(r"```(?:json)?", "", raw_text)
        cleaned = cleaned.replace("```", "").strip()

        json_start = cleaned.find('{')
        if json_start != -1:
            brace_count = 0
            json_end = json_start
            for i in range(json_start, len(cleaned)):
                if cleaned[i] == '{':
                    brace_count += 1
                elif cleaned[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            json_str = cleaned[json_start:json_end]
        else:
            json_str = cleaned

        try:
            parsed_json = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse LLM output as JSON.\n"
                f"Raw output:\n{raw_text}\n"
                f"Extracted JSON string:\n{json_str}"
            ) from e

        try:
            return CRMAgentOutput(**parsed_json)
        except ValidationError as e:
            raise ValueError(
                f"LLM output JSON does not match CRMAgentOutput schema.\n"
                f"Parsed JSON: {parsed_json}"
            ) from e
