import os
import random
import pickle
import re
import json
import numpy as np
import pandas as pd
import torch
import faiss
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer

# -----------------------------
# 0) 시드 고정
# -----------------------------
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# -----------------------------
# 1) 데이터 로드 및 전처리
# -----------------------------
file_path = "/home/food/people/subin/data/식품안전정보DB-url 추가(2014~2024).xls"
df = pd.read_excel(file_path, sheet_name="2023", usecols=["제목", "내용"])

def compose_text(title: str, content: str) -> str:
    title = title or ""
    content = content or ""
    return (title + " " + content).strip()

DATA = df.apply(lambda r: compose_text(r["제목"], r["내용"]), axis=1).tolist()

# -----------------------------
# 2) 리트리버 로딩
# -----------------------------
with open("/home/food/people/minju/embedding/bm25/bm25_2023.pkl", "rb") as f:
    kiwi_retriever = pickle.load(f)
    kiwi_retriever.k = 5

with open("/home/food/people/subin/data/embeddings/LaBSE/embeddings_2023.pkl", "rb") as f:
    EMBEDDINGS = pickle.load(f)

faiss.normalize_L2(EMBEDDINGS)
index = faiss.IndexFlatIP(EMBEDDINGS.shape[1])
index.add(EMBEDDINGS)

EMBED_MODEL = SentenceTransformer("sentence-transformers/LaBSE")

# -----------------------------
# 3) 하이브리드 검색 함수
# -----------------------------
def hybrid_retrieval(query: str, k=10, bm25_weight=0.6, faiss_weight=0.4):
    bm25_docs = kiwi_retriever.get_relevant_documents(query)
    bm25_result = {doc.page_content: bm25_weight * (len(bm25_docs) - rank)
                   for rank, doc in enumerate(bm25_docs)}

    q_emb = EMBED_MODEL.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(q_emb, 5)
    faiss_result = {DATA[idx]: faiss_weight * (len(indices[0]) - rank)
                    for rank, idx in enumerate(indices[0])}

    merged_result = bm25_result.copy()
    for doc, score in faiss_result.items():
        merged_result[doc] = max(merged_result.get(doc, 0), score)

    sorted_docs = sorted(merged_result.items(), key=lambda x: x[1], reverse=True)[:k]
    return sorted_docs

# -----------------------------
# 4) JSON 입력 → 실행 및 저장
# -----------------------------
def run_and_save_from_json(input_json_path: str, out_path: str):
    with open(input_json_path, "r", encoding="utf-8") as f:
        qa_items = json.load(f)

    results = []
    for item in qa_items:
        qid = item.get("id")
        title = item.get("title", "")
        content = item.get("content", "")
        gold = item.get("answer", "")
        query = item.get("query")

        target_full = compose_text(title, content)
        docs_scored = hybrid_retrieval(query)

        row = {
            "id": qid,
            "제목": title,
            "질문": query,
            "예상답변": "(LLM 비활성화: 리트리버 결과만 출력)",
            "정답": gold
        }

        for i in range(10):
            if i < len(docs_scored):
                doc_i, sc_i = docs_scored[i]
                snippet = doc_i[:300].replace("\n", " ") + "..."
                row[f"문서{i+1}"] = f"[{i+1}] {snippet}"
                row[f"점수{i+1}"] = sc_i
            else:
                row[f"문서{i+1}"] = ""
                row[f"점수{i+1}"] = ""

        results.append(row)

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[DONE] 결과가 '{out_path}'에 저장되었습니다.")

# -----------------------------
# 5) 실행
# -----------------------------
if __name__ == "__main__":
    input_json = "/home/food/people/minju/final/queries_generated_from_list.json"
    output_csv = "./original_hybrid_real.csv"
    run_and_save_from_json(input_json, output_csv)