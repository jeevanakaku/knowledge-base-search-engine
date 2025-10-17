# ingest.py
# Usage: python ingest.py --docs_dir ./docs --index_path ./faiss.index --meta_path ./docs_meta.json --method sentence_transformer
# method: sentence_transformer | openai

import os
import argparse
import json
from tqdm import tqdm
import numpy as np

# PDF text extraction
import fitz  # PyMuPDF

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    texts=[]
    for page in doc:
        texts.append(page.get_text())
    return "\n".join(texts)

def load_documents(docs_dir):
    docs=[]
    for root,_,files in os.walk(docs_dir):
        for f in files:
            if f.lower().endswith(('.pdf','.txt')):
                path=os.path.join(root,f)
                if f.lower().endswith('.pdf'):
                    text=extract_text_from_pdf(path)
                else:
                    with open(path,'r',encoding='utf-8',errors='ignore') as fh:
                        text=fh.read()
                if text.strip():
                    docs.append({"id": len(docs), "source_path": path, "text": text[:100000], "title": f})
    return docs

# Chunking simple (split by paragraphs to limit embedding size)
def chunk_text(text, max_tokens=1000):
    # naive chunk by sentences/paragraphs; tune if needed
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks=[]
    cur=""
    for p in paras:
        if len(cur)+len(p) > 3000:  # approx char limit
            chunks.append(cur)
            cur=p
        else:
            cur = (cur + "\n\n" + p).strip() if cur else p
    if cur:
        chunks.append(cur)
    return chunks

# Embedding helpers
def embed_with_sentence_transformer(texts, model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return emb

def embed_with_openai(texts, engine="text-embedding-3-small", openai_api_key=None):
    import openai
    if openai_api_key:
        openai.api_key=openai_api_key
    embs=[]
    BATCH=16
    for i in range(0,len(texts),BATCH):
        batch = texts[i:i+BATCH]
        resp = openai.Embeddings.create(model=engine, input=batch)
        for r in resp['data']:
            embs.append(np.array(r['embedding'], dtype=np.float32))
    return np.vstack(embs)

def build_faiss_index(embeddings, index_path):
    import faiss
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product; we will normalize vectors
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, index_path)

def save_meta(meta, meta_path):
    with open(meta_path,'w',encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

def main(args):
    docs = load_documents(args.docs_dir)
    print(f"Found {len(docs)} documents.")
    # split docs into chunks
    chunks=[]
    for doc in docs:
        cks = chunk_text(doc['text'])
        for i,c in enumerate(cks):
            chunks.append({
                "doc_id": doc['id'],
                "chunk_id": f"{doc['id']}_{i}",
                "text": c,
                "source": doc['source_path'],
                "title": doc['title']
            })
    print(f"Created {len(chunks)} chunks.")
    texts = [c['text'] for c in chunks]

    if args.method == "sentence_transformer":
        emb = embed_with_sentence_transformer(texts, model_name=args.st_model)
        emb = np.array(emb).astype('float32')
    else:
        emb = embed_with_openai(texts, engine=args.openai_model, openai_api_key=args.openai_key)
        emb = np.array(emb).astype('float32')

    # normalize for cosine similarity with IndexFlatIP
    faiss.normalize_L2(emb)

    # save index
    import faiss
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)
    faiss.write_index(index, args.index_path)
    print("FAISS index saved to", args.index_path)

    # save metadata aligned with embeddings
    save_meta(chunks, args.meta_path)
    print("Metadata saved to", args.meta_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", required=True)
    parser.add_argument("--index_path", default="faiss.index")
    parser.add_argument("--meta_path", default="docs_meta.json")
    parser.add_argument("--method", choices=["sentence_transformer","openai"], default="sentence_transformer")
    # sentence-transformer option
    parser.add_argument("--st_model", default="all-MiniLM-L6-v2")
    # openai option
    parser.add_argument("--openai_model", default="text-embedding-3-small")
    parser.add_argument("--openai_key", default=None)
    args = parser.parse_args()
    main(args)
