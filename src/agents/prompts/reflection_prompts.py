from typing import List


class ReflectionPrompts:
    
    @staticmethod
    def build_reflection_prompt(
        user_input: str,
        crm_output_message: str,
        crm_rationale: str,
        crm_irrelevant_query: bool,
        tool_calls_made: List[str]
    ) -> str:
        """
        Build the reflection prompt for the QA agent to evaluate CRM agent's response.
        """
        tools_used = ", ".join(tool_calls_made) if tool_calls_made else "None"
        
        prompt = f"""You are a QA agent evaluating a CRM Sales Agent's response.

CRM Agent's Task:
- Answer sales-related questions using two tools:
  1. search_structured_db(client_name): Get client facts (spend, industry, account manager)
  2. search_semantic_db(query): Get sales rules and policies
- For recommendations, CRM should use BOTH tools
- For client facts only, use search_structured_db
- For policies only, use search_semantic_db
- Mark irrelevant (non-sales) queries as irrelevant_query=True

User Query: {user_input}

CRM Agent's Response:
- Output Message: {crm_output_message}
- Rationale: {crm_rationale}
- Marked as Irrelevant: {crm_irrelevant_query}
- Tools Used: {tools_used}

Evaluate if the CRM agent:
1. Used the correct tool(s) for this query type
2. Provided accurate information based on retrieved data
3. Correctly identified if query was irrelevant

OUTPUT FORMAT (valid JSON only):
{{"correctness": true/false, "rationale": "brief explanation of your evaluation"}}

Examples:
- If user asks for product recommendation but CRM only used one tool: {{"correctness": false, "rationale": "Recommendation queries require both search_structured_db and search_semantic_db, but only one was used."}}
- If CRM correctly answered with proper tools: {{"correctness": true, "rationale": "CRM used appropriate tools and provided accurate response."}}
- If user asked non-sales question and CRM marked it irrelevant: {{"correctness": true, "rationale": "Query was correctly identified as out of scope."}}

Your evaluation:"""
        
        return prompt
