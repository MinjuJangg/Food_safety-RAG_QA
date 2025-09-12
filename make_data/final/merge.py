import os, random, pickle, re, json
import numpy as np
import pandas as pd
import torch, faiss
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# (선택) LLM 프롬프트 유틸이 있다면 사용
try:
    from llm import prompting_answer  # 없으면 LLM 비활성화 상태에서 무시됨
except Exception:
    prompting_answer = None

# -----------------------------
# 0) 공통 세팅
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
# 1) 데이터 로드 (제목 + 내용)로 인덱싱 텍스트 구성
# -----------------------------
import pandas as pd
from typing import List

# -----------------------------
# 1) 전체 시트 합치기
# -----------------------------
file_path = "/home/food/people/subin/data/식품안전정보DB-url 추가(2014~2024).xls"
xls = pd.ExcelFile(file_path)

# 연도순 정렬
sorted_sheets = sorted(xls.sheet_names, key=lambda x: int(x))

dfs = []
for sheet in sorted_sheets:
    df_tmp = pd.read_excel(file_path, sheet_name=sheet, usecols=["제목", "내용"])
    dfs.append(df_tmp)

df = pd.concat(dfs, ignore_index=True)
df = df[~df['내용'].isnull()].reset_index(drop=True)




# -----------------------------
# 2) 제목+내용 합치기
# -----------------------------
def compose_title_content(title: str, content: str, use_title: bool = True) -> str:
    title = str(title) if pd.notna(title) else ""
    content = str(content) if pd.notna(content) else ""
    return (title + " " + content).strip() if use_title else content.strip()

df["검색텍스트"] = df.apply(
    lambda r: compose_title_content(r["제목"], r["내용"], use_title=True), axis=1
)

# -----------------------------
# 3) 검색용 DATA 리스트 생성
# -----------------------------
DATA: List[str] = df["검색텍스트"].tolist()
print(f"[INFO] 데이터 총 개수: {len(DATA)}")


# -----------------------------
# 2) 구성 옵션
# -----------------------------
@dataclass
class SearchConfig:
    mode: str = "embedding"     # ["embedding", "bm25", "hybrid_sum", "hybrid_max"]
    k: int = 5                # 최종 반환 문서 수
    k_bm25: int = 5            # BM25 후보 수
    k_embed: int = 5           # 임베딩 후보 수
    bm25_weight: float = 0.6    # 하이브리드(합산) 가중치
    embed_weight: float = 0.4   # 하이브리드(합산) 가중치
    use_reranker: bool = False  # ko-reranker 사용여부
    reranker_top_k: Optional[int] = None  # 리랭크 이후 상위 n (None=자동 k)
    enable_llm: bool = False    # 기본 False: LLM 전혀 안 씀

CFG = SearchConfig(
    mode="embedding",
    k=5, k_bm25=5, k_embed=5,
    bm25_weight=0.6, embed_weight=0.4,
    use_reranker=False, reranker_top_k=None,
    enable_llm=False
)

# -----------------------------
# 3) 리트리버 로드 (옵셔널)
# -----------------------------
# 3-1) BM25 (Kiwi)
kiwi_retriever = None
try:
    with open(f"/home/food/people/minju/embedding/bm25/bm25_all.pkl", "rb") as f:
        kiwi_retriever = pickle.load(f)
    kiwi_retriever.k = 10
    print("[INFO] KiwiBM25 로드 완료")
except Exception as e:
    print(f"[WARN] KiwiBM25 로드 실패 → BM25 모드 비활성화: {e}")

# 3-2) FAISS + 임베딩 (LaBSE)
with open(f"/home/food/people/minju/embedding/labse2/embeddings_all.pkl", "rb") as f:
    EMBEDDINGS = pickle.load(f)   # shape: (N, D)

DIM = EMBEDDINGS.shape[1]
faiss.normalize_L2(EMBEDDINGS)
FAISS_INDEX = faiss.IndexFlatIP(DIM)
FAISS_INDEX.add(EMBEDDINGS)
EMBED_MODEL = SentenceTransformer("sentence-transformers/LaBSE")
print("[INFO] FAISS + LaBSE 준비 완료")

# sample_queries = [100, 5000, 20000]
# for idx in sample_queries:
#     q = EMBED_MODEL.encode([DATA[idx]], convert_to_numpy=True, normalize_embeddings=True)
#     D, I = FAISS_INDEX.search(q, 1)
#     print(f"Query={idx} | Retrieved={I[0][0]} | Distance={D[0][0]:.4f}")



# 3-3) (옵션) ko-reranker
reranker_tokenizer, reranker_model = None, None
try:
    reranker_model_name = "Dongjin-kr/ko-reranker"
    reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
    reranker_model.eval()
    print("[INFO] ko-reranker 로드 완료")
except Exception as e:
    print(f"[WARN] ko-reranker 로드 실패 → 리랭크 비활성화: {e}")

# 3-4) (옵션) EXAONE (답변 생성)
GEN = None
if CFG.enable_llm:
    exaone_model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    tokenizer_exaone = AutoTokenizer.from_pretrained(exaone_model_name, trust_remote_code=True, timeout=60)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True
    )
    exaone_model = AutoModelForCausalLM.from_pretrained(
        exaone_model_name, device_map="auto",
        quantization_config=bnb_config, trust_remote_code=True
    )
from transformers import pipeline
import os, random, pickle, re, json
import numpy as np
import pandas as pd
import torch, faiss
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# (선택) LLM 프롬프트 유틸이 있다면 사용
try:
    from llm import prompting_answer  # 없으면 LLM 비활성화 상태에서 무시됨
except Exception:
    prompting_answer = None

# -----------------------------
# 0) 공통 세팅
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
# 1) 데이터 로드 (제목 + 내용)로 인덱싱 텍스트 구성
# -----------------------------
import pandas as pd
from typing import List

# -----------------------------
# 1) 전체 시트 합치기
# -----------------------------
file_path = "/home/food/people/subin/data/식품안전정보DB-url 추가(2014~2024).xls"
xls = pd.ExcelFile(file_path)

# 연도순 정렬
sorted_sheets = sorted(xls.sheet_names, key=lambda x: int(x))

dfs = []
for sheet in sorted_sheets:
    df_tmp = pd.read_excel(file_path, sheet_name=sheet, usecols=["제목", "내용"])
    dfs.append(df_tmp)

df = pd.concat(dfs, ignore_index=True)
df = df[~df['내용'].isnull()].reset_index(drop=True)




# -----------------------------
# 2) 제목+내용 합치기
# -----------------------------
def compose_title_content(title: str, content: str, use_title: bool = True) -> str:
    title = str(title) if pd.notna(title) else ""
    content = str(content) if pd.notna(content) else ""
    return (title + " " + content).strip() if use_title else content.strip()

df["검색텍스트"] = df.apply(
    lambda r: compose_title_content(r["제목"], r["내용"], use_title=True), axis=1
)

# -----------------------------
# 3) 검색용 DATA 리스트 생성
# -----------------------------
DATA: List[str] = df["검색텍스트"].tolist()
print(f"[INFO] 데이터 총 개수: {len(DATA)}")


# -----------------------------
# 2) 구성 옵션
# -----------------------------
@dataclass
class SearchConfig:
    mode: str = "embedding"     # ["embedding", "bm25", "hybrid_sum", "hybrid_max"]
    k: int = 5                # 최종 반환 문서 수
    k_bm25: int = 5            # BM25 후보 수
    k_embed: int = 5           # 임베딩 후보 수
    bm25_weight: float = 0.6    # 하이브리드(합산) 가중치
    embed_weight: float = 0.4   # 하이브리드(합산) 가중치
    use_reranker: bool = False  # ko-reranker 사용여부
    reranker_top_k: Optional[int] = None  # 리랭크 이후 상위 n (None=자동 k)
    enable_llm: bool = False    # 기본 False: LLM 전혀 안 씀

CFG = SearchConfig(
    mode="embedding",
    k=10, k_bm25=5, k_embed=5,
    bm25_weight=0.6, embed_weight=0.4,
    use_reranker=False, reranker_top_k=None,
    enable_llm=False
)

# -----------------------------
# 3) 리트리버 로드 (옵셔널)
# -----------------------------
# 3-1) BM25 (Kiwi)
kiwi_retriever = None
try:
    with open(f"/home/food/people/minju/embedding/bm25/bm25_all.pkl", "rb") as f:
        kiwi_retriever = pickle.load(f)
    kiwi_retriever.k = 10
    print("[INFO] KiwiBM25 로드 완료")
except Exception as e:
    print(f"[WARN] KiwiBM25 로드 실패 → BM25 모드 비활성화: {e}")

# 3-2) FAISS + 임베딩 (LaBSE)
with open(f"/home/food/people/minju/embedding/labse2/embeddings_all.pkl", "rb") as f:
    EMBEDDINGS = pickle.load(f)   # shape: (N, D)

DIM = EMBEDDINGS.shape[1]
faiss.normalize_L2(EMBEDDINGS)
FAISS_INDEX = faiss.IndexFlatIP(DIM)
FAISS_INDEX.add(EMBEDDINGS)
EMBED_MODEL = SentenceTransformer("sentence-transformers/LaBSE")
print("[INFO] FAISS + LaBSE 준비 완료")

# sample_queries = [100, 5000, 20000]
# for idx in sample_queries:
#     q = EMBED_MODEL.encode([DATA[idx]], convert_to_numpy=True, normalize_embeddings=True)
#     D, I = FAISS_INDEX.search(q, 1)
#     print(f"Query={idx} | Retrieved={I[0][0]} | Distance={D[0][0]:.4f}")



# 3-3) (옵션) ko-reranker
reranker_tokenizer, reranker_model = None, None
try:
    reranker_model_name = "Dongjin-kr/ko-reranker"
    reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
    reranker_model.eval()
    print("[INFO] ko-reranker 로드 완료")
except Exception as e:
    print(f"[WARN] ko-reranker 로드 실패 → 리랭크 비활성화: {e}")

# 3-4) (옵션) EXAONE (답변 생성)
GEN = None
if CFG.enable_llm:
    exaone_model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    tokenizer_exaone = AutoTokenizer.from_pretrained(exaone_model_name, trust_remote_code=True, timeout=60)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True
    )
    exaone_model = AutoModelForCausalLM.from_pretrained(
        exaone_model_name, device_map="auto",
        quantization_config=bnb_config, trust_remote_code=True
    )
    from transformers import pipeline
    GEN = pipeline("text-generation", model=exaone_model, tokenizer=tokenizer_exaone, max_new_tokens=300)

# -----------------------------
# 4) 리트리버 함수
# -----------------------------
def retrieve_bm25(query: str, k: int) -> List[Tuple[str, float]]:
    if kiwi_retriever is None:
        return []
    kiwi_retriever.k = k
    docs = kiwi_retriever.get_relevant_documents(query)
    # 점수 정보가 없다면 순위 기반 점수(상위일수록 점수↑)
    return [(doc.page_content, (k - rank)) for rank, doc in enumerate(docs, start=1)]

def retrieve_embedding(query: str, k: int) -> List[Tuple[str, float]]:
    q = EMBED_MODEL.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = FAISS_INDEX.search(q, k)  # Inner Product → 코사인 유사도처럼 사용
    idxs = indices[0].tolist()
    sims = distances[0].tolist()
    return [(DATA[i], sims[r]) for r, i in enumerate(idxs)]

# -----------------------------
# 5) 하이브리드 머지
# -----------------------------
def merge_scores_sum(bm25: List[Tuple[str, float]], emb: List[Tuple[str, float]], w_bm25: float, w_emb: float, k: int):
    scores: Dict[str, float] = {}
    for doc, s in bm25:
        scores[doc] = scores.get(doc, 0.0) + w_bm25 * s
    for doc, s in emb:
        scores[doc] = scores.get(doc, 0.0) + w_emb * s
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return merged  # List[(doc, score)]

def merge_scores_max_weighted(
    bm25: List[Tuple[str, float]],
    emb: List[Tuple[str, float]],
    w_bm25: float,
    w_emb: float,
    k: int
    ) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}


    for doc, s in bm25:
        weighted_score = w_bm25 * s
        scores[doc] = max(scores.get(doc, float("-inf")), weighted_score)


    for doc, s in emb:
        weighted_score = w_emb * s
        scores[doc] = max(scores.get(doc, float("-inf")), weighted_score)


    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return merged

# def merge_scores_max(bm25: List[Tuple[str, float]], emb: List[Tuple[str, float]], k: int):
#     scores: Dict[str, float] = {}
#     for doc, s in bm25:
#         scores[doc] = max(scores.get(doc, float("-inf")), s)
#     for doc, s in emb:
#         scores[doc] = max(scores.get(doc, float("-inf")), s)
#     merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
#     return merged

# -----------------------------
# 6) 리랭커
# -----------------------------
def rerank_with_ko_reranker(query: str, docs: List[Tuple[str, float]], top_k: Optional[int]) -> List[Tuple[str, float]]:
    if reranker_model is None or reranker_tokenizer is None:
        return docs
    texts = [d[0] for d in docs]
    pairs = [[query, t] for t in texts]
    inputs = reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        outputs = reranker_model(**inputs)
        r_scores = outputs.logits.view(-1).cpu().numpy().tolist()
    reranked = list(zip(texts, r_scores))
    reranked.sort(key=lambda x: x[1], reverse=True)
    if top_k is None:
        top_k = len(reranked)
    return reranked[:top_k]

# -----------------------------
# 7) 통합 검색
# -----------------------------
def retrieve_docs(query: str, cfg: SearchConfig) -> List[Tuple[str, float]]:
    mode = cfg.mode.lower()
    if mode == "embedding":
        return retrieve_embedding(query, cfg.k)
    elif mode == "bm25":
        return retrieve_bm25(query, cfg.k)
    elif mode == "hybrid_sum":
        bm25 = retrieve_bm25(query, cfg.k_bm25)
        emb  = retrieve_embedding(query, cfg.k_embed)
        return merge_scores_sum(bm25, emb, cfg.bm25_weight, cfg.embed_weight, cfg.k)
    elif mode == "hybrid_max":
        bm25 = retrieve_bm25(query, cfg.k_bm25)
        emb = retrieve_embedding(query, cfg.k_embed)
        return merge_scores_max_weighted(bm25, emb, cfg.bm25_weight, cfg.embed_weight, cfg.k)
        # bm25 = retrieve_bm25(query, cfg.k_bm25)
        # emb  = retrieve_embedding(query, cfg.k_embed)
        # return merge_scores_max(bm25, emb, cfg.k)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

# -----------------------------
# 8) 답변 생성 (LLM off 시 근거만)
# -----------------------------
def make_answer(query: str, docs_scored: List[Tuple[str, float]], cfg: SearchConfig) -> Tuple[str, List[Tuple[str, float]]]:
    if cfg.use_reranker and len(docs_scored) > 0:
        docs_scored = rerank_with_ko_reranker(query, docs_scored, top_k=cfg.reranker_top_k or cfg.k)

    top_texts = [d for d, _ in docs_scored[:cfg.k]]
    context = "\n".join(top_texts)

    if not cfg.enable_llm or GEN is None or prompting_answer is None:
        return "(LLM 비활성화: 근거 문서만 제공)", docs_scored[:cfg.k]

    messages = prompting_answer(query, context, rag=True)
    prompt = "\n\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages]) + "\n\nAssistant:"
    resp = GEN(prompt, num_return_sequences=1)[0]["generated_text"]
    answer = resp.split("Assistant:")[-1].strip()
    answer = re.sub(rf"^{re.escape(query)}[\s:：]*", "", answer).strip()
    if not answer:
        answer = "답변을 생성할 수 없습니다."
    return answer, docs_scored[:cfg.k]

# -----------------------------
# 9) 타깃 매칭 유틸 (제목+내용 기준으로 동일 비교)
# -----------------------------
def _normalize_text(t: str) -> str:
    if t is None:
        return ""
    return re.sub(r"\s+", " ", str(t)).strip()

def _find_target_in_used(target_text: str, used_docs: List[Tuple[str, float]]) -> Tuple[bool, Optional[int], Optional[float]]:
    """
    used_docs: [(doc_text, score)] (1위부터)
    비교는 제목+내용 정규화 문자열로 수행.
    1) 완전 일치
    2) 부분 포함(길이 큰 쪽의 앞 200자 기준)
    """
    norm_target = _normalize_text(target_text)
    short_probe = norm_target[:200] if len(norm_target) > 200 else norm_target

    for idx, (doc, sc) in enumerate(used_docs):
        norm_doc = _normalize_text(doc)
        if norm_doc == norm_target:
            return True, idx + 1, sc
        if short_probe and (short_probe in norm_doc or norm_doc[:200] in norm_target):
            return True, idx + 1, sc
    return False, None, None

# -----------------------------
# 10) JSON 입력 → 실행/저장
# -----------------------------
def run_and_save_from_json(input_json_path: str, cfg: SearchConfig, out_path: str = "./qa_result_from_json.csv"):
    with open(input_json_path, "r", encoding="utf-8") as f:
        qa_items = json.load(f)
    print(len(qa_items))
    results = []
    for item in qa_items:
        qid   = item.get("id")
        title = item.get("title", "")
        gold  = item.get("answer", "")
        query = item["query"]
        target_content = item.get("content", "")

        # 타깃을 “제목 + 내용”으로 합쳐 동일 기준으로 비교
        target_full = compose_title_content(title, target_content, use_title=True)

        docs_scored = retrieve_docs(query, cfg)
        pred, used  = make_answer(query, docs_scored, cfg)

        # 최종 Top-k에서 타깃 등장 여부/순위/점수
        target_found, target_rank, target_score = _find_target_in_used(target_full, used)

        row = {
            "id": qid,
            "제목": title,
            "질문": query,
            "예상답변": pred,
            "정답": gold,
            "target_found": bool(target_found),
            "target_rank": target_rank if target_rank is not None else "",
            "target_score": target_score if target_score is not None else "",
        }

        for i in range(cfg.k):
            if i < len(used):
                doc_i, sc_i = used[i]
                snippet = doc_i[:300].replace("\n", " ") + "..."
                row[f"문서{i+1}"] = f"[{i+1}] {snippet}"
                row[f"점수{i+1}"] = sc_i
            else:
                row[f"문서{i+1}"] = ""
                row[f"점수{i+1}"] = ""

        results.append(row)

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[DONE] JSON 입력 결과가 '{out_path}'로 저장되었습니다.")

# -----------------------------
# 11) 실행
# -----------------------------
if __name__ == "__main__":
    # # 모드 설정
    # CFG.mode = "embedding"      # "embedding" / "bm25" / "hybrid_sum" / "hybrid_max"
    # CFG.use_reranker = True
    # CFG.enable_llm = False      # 리트리버만 사용
    # CFG.k = 10
    # CFG.reranker_top_k = 10
                   
    # #    BM25 단일
    # CFG.mode = "bm25"
    # CFG.use_reranker = True
    # CFG.enable_llm = False      # 리트리버만 사용
    # CFG.k = 10
    # CFG.reranker_top_k = 10
    
    
    # print("DATA size:", len(DATA))
    # print("Embedding size:", EMBEDDINGS.shape[0])

    # 예) 하이브리드(합산)
    CFG.mode = "hybrid_sum"; CFG.bm25_weight = 0.6; CFG.embed_weight = 0.4
    CFG.use_reranker = True
    CFG.enable_llm = True      # 리트리버만 사용
    CFG.k = 10
    CFG.reranker_top_k = 10
    
    
    # CFG.k = 5               # ⬅️ 혹시 위에서 바꾼 적이 있다면 여기서도 최종 고정
    # CFG.reranker_top_k = 5  # ⬅️ 리랭커 사용할 땐 함께 지정

    # # 예) 하이브리드(max) + 리랭커
    # CFG.mode = "hybrid_max"
    # CFG.bm25_weight = 0.6
    # CFG.embed_weight = 0.4
    # CFG.use_reranker = False
    # CFG.enable_llm = False

    

    input_json = f"/home/food/people/minju/make_data/merged.json"
    output_csv = f"/home/food/people/minju/final/top5/hybrid_reranker.csv"

    run_and_save_from_json(input_json, CFG, out_path=output_csv)


    GEN = pipeline("text-generation", model=exaone_model, tokenizer=tokenizer_exaone, max_new_tokens=300)

# -----------------------------
# 4) 리트리버 함수
# -----------------------------
def retrieve_bm25(query: str, k: int) -> List[Tuple[str, float]]:
    if kiwi_retriever is None:
        return []
    kiwi_retriever.k = k
    docs = kiwi_retriever.get_relevant_documents(query)
    # 점수 정보가 없다면 순위 기반 점수(상위일수록 점수↑)
    return [(doc.page_content, (k - rank)) for rank, doc in enumerate(docs, start=1)]

def retrieve_embedding(query: str, k: int) -> List[Tuple[str, float]]:
    q = EMBED_MODEL.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = FAISS_INDEX.search(q, k)  # Inner Product → 코사인 유사도처럼 사용
    idxs = indices[0].tolist()
    sims = distances[0].tolist()
    return [(DATA[i], sims[r]) for r, i in enumerate(idxs)]

# -----------------------------
# 5) 하이브리드 머지
# -----------------------------
def merge_scores_sum(bm25: List[Tuple[str, float]], emb: List[Tuple[str, float]], w_bm25: float, w_emb: float, k: int):
    scores: Dict[str, float] = {}
    for doc, s in bm25:
        scores[doc] = scores.get(doc, 0.0) + w_bm25 * s
    for doc, s in emb:
        scores[doc] = scores.get(doc, 0.0) + w_emb * s
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return merged  # List[(doc, score)]

def merge_scores_max_weighted(
    bm25: List[Tuple[str, float]],
    emb: List[Tuple[str, float]],
    w_bm25: float,
    w_emb: float,
    k: int
    ) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}


    for doc, s in bm25:
        weighted_score = w_bm25 * s
        scores[doc] = max(scores.get(doc, float("-inf")), weighted_score)


    for doc, s in emb:
        weighted_score = w_emb * s
        scores[doc] = max(scores.get(doc, float("-inf")), weighted_score)


    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return merged

# def merge_scores_max(bm25: List[Tuple[str, float]], emb: List[Tuple[str, float]], k: int):
#     scores: Dict[str, float] = {}
#     for doc, s in bm25:
#         scores[doc] = max(scores.get(doc, float("-inf")), s)
#     for doc, s in emb:
#         scores[doc] = max(scores.get(doc, float("-inf")), s)
#     merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
#     return merged

# -----------------------------
# 6) 리랭커
# -----------------------------
def rerank_with_ko_reranker(query: str, docs: List[Tuple[str, float]], top_k: Optional[int]) -> List[Tuple[str, float]]:
    if reranker_model is None or reranker_tokenizer is None:
        return docs
    texts = [d[0] for d in docs]
    pairs = [[query, t] for t in texts]
    inputs = reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        outputs = reranker_model(**inputs)
        r_scores = outputs.logits.view(-1).cpu().numpy().tolist()
    reranked = list(zip(texts, r_scores))
    reranked.sort(key=lambda x: x[1], reverse=True)
    if top_k is None:
        top_k = len(reranked)
    return reranked[:top_k]

# -----------------------------
# 7) 통합 검색
# -----------------------------
def retrieve_docs(query: str, cfg: SearchConfig) -> List[Tuple[str, float]]:
    mode = cfg.mode.lower()
    if mode == "embedding":
        return retrieve_embedding(query, cfg.k)
    elif mode == "bm25":
        return retrieve_bm25(query, cfg.k)
    elif mode == "hybrid_sum":
        bm25 = retrieve_bm25(query, cfg.k_bm25)
        emb  = retrieve_embedding(query, cfg.k_embed)
        return merge_scores_sum(bm25, emb, cfg.bm25_weight, cfg.embed_weight, cfg.k)
    elif mode == "hybrid_max":
        bm25 = retrieve_bm25(query, cfg.k_bm25)
        emb = retrieve_embedding(query, cfg.k_embed)
        return merge_scores_max_weighted(bm25, emb, cfg.bm25_weight, cfg.embed_weight, cfg.k)
        # bm25 = retrieve_bm25(query, cfg.k_bm25)
        # emb  = retrieve_embedding(query, cfg.k_embed)
        # return merge_scores_max(bm25, emb, cfg.k)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

# -----------------------------
# 8) 답변 생성 (LLM off 시 근거만)
# -----------------------------
def make_answer(query: str, docs_scored: List[Tuple[str, float]], cfg: SearchConfig) -> Tuple[str, List[Tuple[str, float]]]:
    if cfg.use_reranker and len(docs_scored) > 0:
        docs_scored = rerank_with_ko_reranker(query, docs_scored, top_k=cfg.reranker_top_k or cfg.k)

    top_texts = [d for d, _ in docs_scored[:cfg.k]]
    context = "\n".join(top_texts)

    if not cfg.enable_llm or GEN is None or prompting_answer is None:
        return "(LLM 비활성화: 근거 문서만 제공)", docs_scored[:cfg.k]

    messages = prompting_answer(query, context, rag=True)
    prompt = "\n\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages]) + "\n\nAssistant:"
    resp = GEN(prompt, num_return_sequences=1)[0]["generated_text"]
    answer = resp.split("Assistant:")[-1].strip()
    answer = re.sub(rf"^{re.escape(query)}[\s:：]*", "", answer).strip()
    if not answer:
        answer = "답변을 생성할 수 없습니다."
    return answer, docs_scored[:cfg.k]

# -----------------------------
# 9) 타깃 매칭 유틸 (제목+내용 기준으로 동일 비교)
# -----------------------------
def _normalize_text(t: str) -> str:
    if t is None:
        return ""
    return re.sub(r"\s+", " ", str(t)).strip()

def _find_target_in_used(target_text: str, used_docs: List[Tuple[str, float]]) -> Tuple[bool, Optional[int], Optional[float]]:
    """
    used_docs: [(doc_text, score)] (1위부터)
    비교는 제목+내용 정규화 문자열로 수행.
    1) 완전 일치
    2) 부분 포함(길이 큰 쪽의 앞 200자 기준)
    """
    norm_target = _normalize_text(target_text)
    short_probe = norm_target[:200] if len(norm_target) > 200 else norm_target

    for idx, (doc, sc) in enumerate(used_docs):
        norm_doc = _normalize_text(doc)
        if norm_doc == norm_target:
            return True, idx + 1, sc
        if short_probe and (short_probe in norm_doc or norm_doc[:200] in norm_target):
            return True, idx + 1, sc
    return False, None, None

# -----------------------------
# 10) JSON 입력 → 실행/저장
# -----------------------------
def run_and_save_from_json(input_json_path: str, cfg: SearchConfig, out_path: str = "./qa_result_from_json.csv"):
    with open(input_json_path, "r", encoding="utf-8") as f:
        qa_items = json.load(f)
    print(len(qa_items))
    results = []
    for item in qa_items:
        qid   = item.get("id")
        title = item.get("title", "")
        gold  = item.get("answer", "")
        query = item["query"]
        target_content = item.get("content", "")

        # 타깃을 “제목 + 내용”으로 합쳐 동일 기준으로 비교
        target_full = compose_title_content(title, target_content, use_title=True)

        docs_scored = retrieve_docs(query, cfg)
        pred, used  = make_answer(query, docs_scored, cfg)

        # 최종 Top-k에서 타깃 등장 여부/순위/점수
        target_found, target_rank, target_score = _find_target_in_used(target_full, used)

        row = {
            "id": qid,
            "제목": title,
            "질문": query,
            "예상답변": pred,
            "정답": gold,
            "target_found": bool(target_found),
            "target_rank": target_rank if target_rank is not None else "",
            "target_score": target_score if target_score is not None else "",
        }

        for i in range(cfg.k):
            if i < len(used):
                doc_i, sc_i = used[i]
                snippet = doc_i[:300].replace("\n", " ") + "..."
                row[f"문서{i+1}"] = f"[{i+1}] {snippet}"
                row[f"점수{i+1}"] = sc_i
            else:
                row[f"문서{i+1}"] = ""
                row[f"점수{i+1}"] = ""

        results.append(row)

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[DONE] JSON 입력 결과가 '{out_path}'로 저장되었습니다.")

# -----------------------------
# 11) 실행
# -----------------------------
if __name__ == "__main__":
    # # 모드 설정
    CFG.mode = "embedding"      # "embedding" / "bm25" / "hybrid_sum" / "hybrid_max"
    CFG.use_reranker = True
    CFG.enable_llm = False      # 리트리버만 사용
    CFG.k = 20
    CFG.reranker_top_k = 10
                   
    # #    BM25 단일
    # CFG.mode = "bm25"
    # CFG.use_reranker = True
    # CFG.enable_llm = False      # 리트리버만 사용
    # CFG.k = 20
    # CFG.reranker_top_k = 10
    
    
    # print("DATA size:", len(DATA))
    # print("Embedding size:", EMBEDDINGS.shape[0])

    # 예) 하이브리드(합산)
    # CFG.mode = "hybrid_sum"; CFG.bm25_weight = 0.6; CFG.embed_weight = 0.4
    # CFG.use_reranker = True
    # CFG.enable_llm = False  # 리트리버만 사용
    # CFG.k = 20
    # CFG.reranker_top_k = 10
    
    
    # CFG.k = 5              
    # CFG.reranker_top_k = 5  

    # # 예) 하이브리드(max) + 리랭커
    # CFG.mode = "hybrid_max"
    # CFG.bm25_weight = 0.6
    # CFG.embed_weight = 0.4
    # CFG.use_reranker = False
    # CFG.enable_llm = False

    

    input_json = f"/home/food/people/minju/make_data/merged.json"
    output_csv = f"/home/food/people/minju/final/top5/labse_reranker20.csv"

    run_and_save_from_json(input_json, CFG, out_path=output_csv)

