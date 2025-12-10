import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="ArXivCode: Research Paper to Code",
    layout="wide"
)

st.title("ArXivCode: From Theory to Implementation")
st.markdown("Bridge the gap between AI research and practical implementation. Search for theoretical concepts from papers and get explained code snippets with annotations.")

# Sidebar
with st.sidebar:
    st.header("Example Queries")
    examples = [
        "attention mechanism implementation",
        "transformer encoder layer",
        "self-attention pytorch",
        "LoRA fine-tuning",
        "gradient descent optimizer",
        "batch normalization layer",
        "dropout regularization",
        "cross-entropy loss function"
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state.query = ex
    
    st.divider()
    st.header("System Stats")
    try:
        stats = requests.get(f"{API_URL}/stats").json()
        st.metric("Code Snippets", stats["total_snippets"])
        st.metric("Embedding Model", "CodeBERT")
    except:
        st.warning("API not connected")

# Main interface
query = st.text_input(
    "Search theoretical concepts and get related code:",
    value=st.session_state.get("query", ""),
    placeholder="e.g., attention mechanism implementation"
)

top_k = st.slider("Number of results", 1, 10, 5)

if st.button("Search", type="primary") and query:
    with st.spinner("Searching..."):
        try:
            response = requests.post(
                f"{API_URL}/search",
                json={"query": query, "top_k": top_k}
            )
            results = response.json()["results"]
            
            st.success(f"Found {len(results)} results")
            
            for i, result in enumerate(results):
                with st.expander(
                    f"**{i+1}. {result.get('function_name', 'Unknown')}** "
                    f"(Score: {result['score']:.3f})"
                ):
                    st.markdown(f"**Paper:** {result.get('paper_title', 'N/A')}")
                    st.markdown(f"**File:** `{result.get('file_path', 'N/A')}`")
                    
                    # Display code
                    st.code(result.get('code_text', 'No code available'), language="python")
                    
                    # Action buttons in a row
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("Explain", key=f"explain_{i}", type="primary", use_container_width=True):
                            with st.spinner("Generating explanation..."):
                                explain_response = requests.post(
                                    f"{API_URL}/explain",
                                    json={
                                        "query": query,
                                        "code_snippet": result.get('code_text', ''),
                                        "paper_title": result.get('paper_title', ''),
                                        "paper_context": ""
                                    }
                                )
                                explanation = explain_response.json()["explanation"]
                                st.info(explanation)
                    with col2:
                        if result.get('paper_url'):
                            st.link_button("View Paper", result['paper_url'], use_container_width=True)
                    with col3:
                        if result.get('repo_url'):
                            st.link_button("View Repo", result['repo_url'], use_container_width=True)
                        
        except Exception as e:
            st.error(f"Error: {str(e)}")