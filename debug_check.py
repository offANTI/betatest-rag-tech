import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.retrieval.dense_search import cosine_similarity, load_data
from src.retrieval.bm25 import build_bm25_index, tokenize, batch_chunks

chunks, dense_embeddings = load_data()
bm25 = build_bm25_index(chunks)
model = SentenceTransformer("all-MiniLM-L6-v2")

# найти индекс чанка с isinstance в заголовке
target_idx = None
for i, chunk in enumerate(chunks):
    if "isinstance" in chunk["heading"].lower():
        target_idx = i
        print(f"Found at index {i}: {chunk['heading']}")
        break

if target_idx is None:
    print("isinstance chunk NOT FOUND in corpus at all")
else:
    query = "how to check if a variable is a list"
    query_vec = model.encode(query)
    dense_scores = cosine_similarity(query_vec, dense_embeddings)
    bm25_scores = bm25.get_scores(tokenize(query))

    dense_rank = int(np.argsort(np.argsort(-dense_scores))[target_idx])
    bm25_rank = int(np.argsort(np.argsort(-bm25_scores))[target_idx])
    print(f"dense_rank={dense_rank}, bm25_rank={bm25_rank} (out of {len(chunks)})")