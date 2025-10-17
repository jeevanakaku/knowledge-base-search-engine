# app_streamlit.py
# Run: streamlit run app_streamlit.py

import streamlit as st
import os
import json
import numpy as np
import faiss
from search_and_qa import embed_query_st, embed_query_openai, search_index, load_meta, synthesize_answer_with_openai

st.set_page_config(page_title="KB Search Engine", layout="wide")
st.title("Knowledge-base Search Engine")

index_path = st.text_input("FAISS index path", value="faiss.index")
meta_path = st.text_input("Metadata JSON path", value="docs_meta.json")
method = st.selectbox("Embedding method", ["sentence_transformer","openai"])
query = st.text_area("Your question", height=120)
k = st.slider("Top K", min_value=1, max_value=10, value=5)
synth = st.checkbox("Synthesize answer with OpenAI (requires API key)")
openai_key = st.text_input("OpenAI API Key (optional)", type="password")

if st.button("Search"):
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        st.error("Index or meta file not found. Run ingest.py first.")
    else:
        meta = load_meta(meta_path)
        if method=="sentence_transformer":
            q_emb = embed_query_st(query)
        else:
            q_emb = embed_query_openai(query, openai_api_key=openai_key)
        distances, idxs = search_index(index_path, q_emb, k=k)
        st.subheader("Top results")
        for i,(d,idx) in enumerate(zip(distances, idxs)):
            m = meta[idx]
            st.markdown(f"**{i+1}. {m.get('title','-')}** — score: {d:.4f}")
            st.write(m['text'][:1000])
            st.markdown(f"_source: {m.get('source')}_")
        if synth and openai_key:
            with st.spinner("Calling LLM to synthesize answer..."):
                results = [{"title": load_meta(meta_path)[idx]['title'], "text": load_meta(meta_path)[idx]['text'], "source": load_meta(meta_path)[idx]['source']} for idx in idxs]
                ans = synthesize_answer_with_openai(query, results, openai_api_key=openai_key, model="gpt-4o-mini")
                st.subheader("Synthesized Answer")
                st.write(ans)
