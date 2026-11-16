import os, requests, streamlit as st
st.set_page_config(page_title="AtlasRAG", page_icon="🔎", layout="wide")
st.title("🔎 AtlasRAG")
st.caption("Grounded answers · Hybrid retrieval · Source citations")
api = os.getenv("API_BASE_URL","http://localhost:8000")
with st.sidebar:
    st.subheader("API"); st.code(api)
    st.caption("Start the API and index documents before asking questions.")
question = st.text_area("Ask your knowledge base", height=110,
    placeholder="How long are application audit logs retained?")
k = st.slider("Evidence chunks",1,10,5)
if st.button("Ask AtlasRAG", type="primary", disabled=len(question.strip())<3):
    try:
        with st.spinner("Searching and generating..."):
            res = requests.post(api+"/query",json={"question":question,"top_k":k},timeout=240)
            res.raise_for_status(); data=res.json()
        st.subheader("Answer"); st.write(data["answer"])
        if data["abstained"]: st.warning("The system abstained: retrieved evidence was insufficient.")
        st.caption(f"Latency: {data['latency_ms']} ms")
        st.subheader("Sources")
        for c in data["citations"]:
            with st.expander(f"[{c['citation_id']}] {c['source']} · score {c['score']}"):
                st.write(c["excerpt"]); st.caption(c["chunk_id"])
    except requests.RequestException as e: st.error(f"API request failed: {e}")
