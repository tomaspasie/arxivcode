import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="ArXivCode: Research Paper to Code",
    layout="wide"
)

# Initialize session state for explanations and results
if 'explanations' not in st.session_state:
    st.session_state.explanations = {}
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""

st.title("ArXivCode: From Theory to Implementation")
st.markdown("Bridge the gap between AI research and practical implementation. Search for theoretical concepts from papers and get explained code snippets with annotations.")

# Sidebar
with st.sidebar:
    st.header("Example Queries")
    examples = [
        "how to implement LoRA",
        "transformer attention mechanism",
        "BERT fine-tuning",
        "flash attention",
        "PPO reinforcement learning",
        "vision transformer",
        "knowledge distillation",
        "prompt tuning"
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
            st.session_state.search_results = response.json()["results"]
            st.session_state.last_query = query
            st.session_state.explanations = {}  # Clear old explanations
        except Exception as e:
            st.error(f"Search error: {str(e)}")

# Display results (persisted in session state)
if st.session_state.search_results:
    results = st.session_state.search_results
    st.success(f"Found {len(results)} results for: **{st.session_state.last_query}**")
    
    for i, result in enumerate(results):
        with st.expander(
            f"**{i+1}. {result.get('function_name', 'Unknown')}** "
            f"(Score: {result['score']:.3f})",
            expanded=(i == 0)  # First result expanded by default
        ):
            st.markdown(f"**Paper:** {result.get('paper_title', 'N/A')}")
            st.markdown(f"**File:** `{result.get('file_path', 'N/A')}`")
            
            # Action buttons in a row
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Explain", key=f"explain_{i}", type="primary", use_container_width=True):
                    with st.spinner("Generating explanation..."):
                        try:
                            explain_response = requests.post(
                                f"{API_URL}/explain",
                                json={
                                    "query": st.session_state.last_query,
                                    "code_snippet": result.get('code_text', ''),
                                    "paper_title": result.get('paper_title', ''),
                                    "paper_context": ""
                                }
                            )
                            st.session_state.explanations[i] = explain_response.json()["explanation"]
                        except Exception as e:
                            st.session_state.explanations[i] = f"Error generating explanation: {str(e)}"
            with col2:
                if result.get('paper_url'):
                    st.link_button("View Paper", result['paper_url'], use_container_width=True)
            with col3:
                if result.get('repo_url'):
                    st.link_button("View Repo", result['repo_url'], use_container_width=True)
            
            # Show explanation box if it exists for this result
            if i in st.session_state.explanations:
                st.markdown("---")
                st.markdown("### Explanation")
                st.info(st.session_state.explanations[i])
            
            # Display code (below buttons and explanation)
            st.markdown("### Code")
            st.code(result.get('code_text', 'No code available'), language="python")