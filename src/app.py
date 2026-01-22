import streamlit as st
import os
import asyncio
from src.agents.main_graph import CRMGraph
from src.agents.tools import structured_db_cache, semantic_db_cache

st.set_page_config(
    page_title="CRM Chatbot",
    page_icon="💬",
    layout="wide"
)

if "graph" not in st.session_state:
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    st.session_state.graph = CRMGraph(model_name=model_name)
    st.session_state.messages = []

st.title("💬 CRM Chatbot")
st.markdown("Ask me anything about your CRM data!")

with st.sidebar:
    st.header("⚡ Performance")
    
    if st.button("Show Cache Stats"):
        struct_stats = structured_db_cache.stats()
        semantic_stats = semantic_db_cache.stats()
        
        st.subheader("Structured DB Cache")
        st.json(struct_stats)
        
        st.subheader("Semantic DB Cache")
        st.json(semantic_stats)
    
    if st.button("Clear Caches"):
        structured_db_cache.clear()
        semantic_db_cache.clear()
        st.success("Caches cleared!")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def stream_response(graph: CRMGraph, query: str):
    """
    Stream response from the graph with progress indicators.
    Falls back to non-streaming if streaming not available.
    """
    try:
        compiled_graph = graph.build_graph()
        from langchain_core.messages import HumanMessage
        
        state = {
            "input": query,
            "output": "",
            "stop": False,
            "reflection_count": 0,
            "last_tool_calls": None,
            "reflection_output": "",
            "messages": [HumanMessage(content=query)],
            "crm_agent_output": None
        }
        
        full_response = ""
        current_node = ""
        

        for event in compiled_graph.stream(state, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_name != current_node:
                    current_node = node_name
                    yield f"\n🔄 *{node_name}*\n"

                if "output" in node_output and node_output["output"]:
                    full_response = node_output["output"]

        if full_response:
            yield f"\n---\n{full_response}"
        
    except Exception as e:
        yield graph.invoke_graph(query)


if prompt := st.chat_input("Enter your query here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Processing..."):
                for chunk in stream_response(st.session_state.graph, prompt):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
            if "---" in full_response:
                final_output = full_response.split("---")[-1].strip()
            else:
                final_output = full_response
            
            st.session_state.messages.append({"role": "assistant", "content": final_output})
            
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
