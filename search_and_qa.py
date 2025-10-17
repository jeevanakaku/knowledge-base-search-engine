# search_and_qa.py
# Usage:
# python search_and_qa.py --index_path faiss.index --meta_path docs_meta.json --query "what is RAG?" --method sentence_transformer
# Set OPENAI_API_KEY env var if using OpenAI synth or embeddings

import os, json, argparse
import numpy as np
import faiss

def load_meta(meta_path):
    with open(meta_path,'r',encoding='utf-8') as fh:
        return json.load(fh)

def embed_query_st(text, model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name)
    v = m.encode([text], convert_to_numpy=True)
    return np.array(v).astype('float32')

def embed_query_openai(text, engine="text-embedding-3-small", openai_api_key=None):
    import openai, numpy as np
    if openai_api_key:
        openai.api_key=openai_api_key
    resp = openai.Embeddings.create(model=engine, input=[text])
    return np.array(resp['data'][0]['embedding'], dtype=np.float32).reshape(1,-1)

def search_index(index_path, q_emb, k=5):
    index = faiss.read_index(index_path)
    faiss.normalize_L2(q_emb)
    distances, idxs = index.search(q_emb, k)
    return distances[0], idxs[0]

def synthesize_answer_with_openai(question, contexts, openai_api_key=None, model="gpt-4o-mini"):
    import openai
    if openai_api_key:
        openai.api_key=openai_api_key
    prompt = "You are an assistant. Use the following extracted document snippets to answer the question concisely and cite sources by filename or title.\n\n"
    for i,c in enumerate(contexts):
        prompt += f"### Source {i+1} ({c.get('title')})\n{c['text']}\n\n"
    prompt += f"Question: {question}\n\nAnswer (short, cite which source lines you used):"
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        max_tokens=350,
        temperature=0.0,
    )
    return resp['choices'][0]['message']['content'].strip()

def main(args):
    meta = load_meta(args.meta_path)
    if args.method=="sentence_transformer":
        q_emb = embed_query_st(args.query, model_name=args.st_model)
    else:
        q_emb = embed_query_openai(args.query, engine=args.openai_model, openai_api_key=args.openai_key)
    distances, idxs = search_index(args.index_path, q_emb, k=args.k)
    results=[]
    for dist, idx in zip(distances, idxs):
        m = meta[idx]
        results.append({"score": float(dist), "title": m.get("title"), "source": m.get("source"), "text": m.get("text")[:1200]})
    print("Top results:")
    for r in results:
        print("SCORE:", r['score'], "TITLE:", r['title'], "SOURCE:", r['source'])
        print(r['text'][:400].replace("\n"," ") + "\n---\n")
    if args.synth and args.openai_key:
        answer = synthesize_answer_with_openai(args.query, results, openai_api_key=args.openai_key, model=args.openai_model_qa)
        print("\n=== Synthesized Answer ===\n")
        print(answer)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index_path", default="faiss.index")
    p.add_argument("--meta_path", default="docs_meta.json")
    p.add_argument("--query", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--method", choices=["sentence_transformer","openai"], default="sentence_transformer")
    p.add_argument("--st_model", default="all-MiniLM-L6-v2")
    p.add_argument("--openai_model", default="text-embedding-3-small")
    p.add_argument("--openai_key", default=os.getenv("OPENAI_API_KEY"))
    p.add_argument("--synth", action="store_true", help="Call LLM to synthesize an answer (requires OPENAI_API_KEY)")
    p.add_argument("--openai_model_qa", default="gpt-4o-mini")
    args = p.parse_args()
    main(args)
