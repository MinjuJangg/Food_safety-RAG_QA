# -*- coding: utf-8 -*-
"""
식품안전 RAG 평가 파이프라인 (리팩토링)
- Excel 시트 병합(연도 탭), 제목+내용 결합
- BM25 / LaBSE+FAISS / 하이브리드(sum, max-weighted)
- ko-reranker(옵션), LLM 생성(옵션: EXAONE)
- 진행률 표시(tqdm), 디바이스/FAISS GPU 자동 설정
- JSON QA 스위트 → CSV 결과 저장
"""

import os
import re
import json
import pickle
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import faiss

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)

from sentence_transformers import SentenceTransformer

# -- (선택) LLM 프롬프트 유틸
try:
    from llm import prompting_answer  # 없으면 LLM 비활성화
except Exception:
    prompting_answer = None

# -- tqdm (미설치/미지원 안전 처리)
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x=None, total=None, **kwargs):
        return x if x is not None else range(total or 0)


# =============================================================================
# 0) 공통 유틸
# =============================================================================
def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class SearchConfig:
    # 리트리버/하이브리드
    mode: str = "embedding"     # ["embedding", "bm25", "hybrid_sum", "hybrid_max"]
    k: int = 5
    k_bm25: int = 5
    k_embed: int = 5
    bm25_weight: float = 0.6
    embed_weight: float = 0.4

    # 리랭커/LLM
    use_reranker: bool = False
    reranker_top_k: Optional[int] = None
    enable_llm: bool = False

    # 진행률/로깅
    show_progress: bool = True
    progress_desc: str = "Evaluating"

    # 디바이스/FAISS
    device_id: int = 0
    use_faiss_gpu: bool = True

    # 경로
    excel_path: str = "/home/food/people/subin/data/식품안전정보DB-url 추가(2014~2024).xls"
    bm25_pickle: str = "/home/food/people/minju/embedding/bm25/bm25_all.pkl"
    embeddings_pickle: str = "/home/food/people/minju/embedding/labse2/embeddings_all.pkl"
    input_json: str = "/home/food/people/minju/make_data/merged.json"
    output_csv: str = "/home/food/people/minju/final/top5/hybrid_reranker.csv"


def get_device(cfg: SearchConfig) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{cfg.device_id}")
    return torch.device("cpu")


def normalize_spaces(t: str) -> str:
    if t is None:
        return ""
    return re.sub(r"\s+", " ", str(t)).strip()


def compose_title_content(title: str, content: str, use_title: bool = True) -> str:
    title = str(title) if pd.notna(title) else ""
    content = str(content) if pd.notna(content) else ""
    return (title + " " + content).strip() if use_title else content.strip()


# =============================================================================
# 1) 데이터 적재
# =============================================================================
def load_food_sheets(excel_path: str, show_progress: bool = True) -> pd.DataFrame:
    xls = pd.ExcelFile(excel_path)
    # 탭명이 연도라고 가정
    sorted_sheets = sorted(xls.sheet_names, key=lambda x: int(x))
    frames = []
    iterator = tqdm(sorted_sheets, desc="Load sheets", unit="sheet") if show_progress else sorted_sheets
    for sheet in iterator:
        df_tmp = pd.read_excel(excel_path, sheet_name=sheet, usecols=["제목", "내용"])
        frames.append(df_tmp)
    df = pd.concat(frames, ignore_index=True)
    df = df[~df["내용"].isnull()].reset_index(drop=True)
    # 검색 텍스트 생성
    df["검색텍스트"] = df.apply(lambda r: compose_title_content(r["제목"], r["내용"], use_title=True), axis=1)
    return df


# =============================================================================
# 2) 리트리버 로딩 (BM25 / LaBSE+FAISS / ko-reranker / LLM)
# =============================================================================
class RetrieverBundle:
    def __init__(self):
        self.kiwi_retriever = None
        self.EMBED_MODEL: Optional[SentenceTransformer] = None
        self.FAISS_INDEX = None
        self.EMBEDDINGS = None
        self.reranker_tokenizer = None
        self.reranker_model = None
        self.GEN = None  # pipeline-like 호출체

    def load_bm25(self, bm25_pickle: str):
        try:
            with open(bm25_pickle, "rb") as f:
                self.kiwi_retriever = pickle.load(f)
            print("[INFO] KiwiBM25 로드 완료")
        except Exception as e:
            self.kiwi_retriever = None
            print(f"[WARN] KiwiBM25 로드 실패 → BM25 비활성화: {e}")

    def load_embeddings_and_faiss(self, embeddings_pickle: str, device: torch.device, use_faiss_gpu: bool):
        with open(embeddings_pickle, "rb") as f:
            self.EMBEDDINGS = pickle.load(f).astype(np.float32)  # (N, D)
        dim = self.EMBEDDINGS.shape[1]
        faiss.normalize_L2(self.EMBEDDINGS)

        # CPU 인덱스
        index_cpu = faiss.IndexFlatIP(dim)
        index_cpu.add(self.EMBEDDINGS)

        # 가능하면 GPU로
        if "cuda" in str(device) and use_faiss_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.FAISS_INDEX = faiss.index_cpu_to_gpu(res, device.index or 0, index_cpu)
                print("[INFO] FAISS GPU index 준비 완료")
            except Exception as e:
                print(f"[WARN] FAISS GPU 전환 실패 → CPU 사용: {e}")
                self.FAISS_INDEX = index_cpu
        else:
            self.FAISS_INDEX = index_cpu
            print("[INFO] FAISS CPU index 사용")

        # SentenceTransformer
        self.EMBED_MODEL = SentenceTransformer("sentence-transformers/LaBSE", device=str(device))
        print("[INFO] FAISS + LaBSE 준비 완료 (device:", device, ")")

    def load_reranker(self, device: torch.device):
        try:
            model_name = "Dongjin-kr/ko-reranker"
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
            print("[INFO] ko-reranker 로드 완료 (device:", device, ")")
        except Exception as e:
            self.reranker_tokenizer, self.reranker_model = None, None
            print(f"[WARN] ko-reranker 로드 실패 → 리랭크 비활성화: {e}")

    def load_llm(self, device: torch.device):
        try:
            from transformers import pipeline
            model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
            tokenizer_exaone = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, timeout=60)

            # 멀티 GPU 분산 로딩
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True
            )
            exaone_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",            # GPU  자동 분산
                quantization_config=bnb_config,
                trust_remote_code=True
            )

            self.GEN = pipeline(
                "text-generation",
                model=exaone_model,
                tokenizer=tokenizer_exaone,
                max_new_tokens=300,
            )
            print("[INFO] EXAONE 멀티-GPU 생성 파이프라인 준비 완료")

        except Exception as e:
            self.GEN = None
            print(f"[WARN] EXAONE LLM 로드 실패 → LLM 비활성화: {e}")



# =============================================================================
# 3) 검색기
# =============================================================================
def retrieve_bm25(bundle: RetrieverBundle, query: str, k: int) -> List[Tuple[str, float]]:
    if bundle.kiwi_retriever is None:
        return []
    bundle.kiwi_retriever.k = k
    docs = bundle.kiwi_retriever.get_relevant_documents(query)
    # 점수 정보 없으면 순위 역가중치
    return [(doc.page_content, (k - rank)) for rank, doc in enumerate(docs, start=1)]


def retrieve_embedding(bundle: RetrieverBundle, data_texts: List[str], query: str, k: int) -> List[Tuple[str, float]]:
    q = bundle.EMBED_MODEL.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = bundle.FAISS_INDEX.search(q, k)  # Inner Product ~ cosine
    idxs = indices[0].tolist()
    sims = distances[0].tolist()
    return [(data_texts[i], sims[r]) for r, i in enumerate(idxs)]


def merge_scores_sum(
    bm25: List[Tuple[str, float]], emb: List[Tuple[str, float]], w_bm25: float, w_emb: float, k: int
) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for doc, s in bm25:
        scores[doc] = scores.get(doc, 0.0) + w_bm25 * s
    for doc, s in emb:
        scores[doc] = scores.get(doc, 0.0) + w_emb * s
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


def merge_scores_max_weighted(
    bm25: List[Tuple[str, float]],
    emb: List[Tuple[str, float]],
    w_bm25: float,
    w_emb: float,
    k: int,
) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for doc, s in bm25:
        scores[doc] = max(scores.get(doc, float("-inf")), w_bm25 * s)
    for doc, s in emb:
        scores[doc] = max(scores.get(doc, float("-inf")), w_emb * s)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


def rerank_with_ko_reranker(
    bundle: RetrieverBundle, query: str, docs: List[Tuple[str, float]], top_k: Optional[int]
) -> List[Tuple[str, float]]:
    if bundle.reranker_model is None or bundle.reranker_tokenizer is None:
        return docs
    texts = [d[0] for d in docs]
    pairs = [[query, t] for t in texts]
    inputs = bundle.reranker_tokenizer(
        pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
    ).to(bundle.reranker_model.device)

    with torch.no_grad():
        outputs = bundle.reranker_model(**inputs)
        r_scores = outputs.logits.view(-1).detach().cpu().numpy().tolist()

    reranked = list(zip(texts, r_scores))
    reranked.sort(key=lambda x: x[1], reverse=True)
    if top_k is None:
        top_k = len(reranked)
    return reranked[:top_k]


def retrieve_docs(
    bundle: RetrieverBundle, data_texts: List[str], query: str, cfg: SearchConfig
) -> List[Tuple[str, float]]:
    mode = cfg.mode.lower()
    if mode == "embedding":
        return retrieve_embedding(bundle, data_texts, query, cfg.k)
    elif mode == "bm25":
        return retrieve_bm25(bundle, query, cfg.k)
    elif mode == "hybrid_sum":
        bm25 = retrieve_bm25(bundle, query, cfg.k_bm25)
        emb = retrieve_embedding(bundle, data_texts, query, cfg.k_embed)
        return merge_scores_sum(bm25, emb, cfg.bm25_weight, cfg.embed_weight, cfg.k)
    elif mode == "hybrid_max":
        bm25 = retrieve_bm25(bundle, query, cfg.k_bm25)
        emb = retrieve_embedding(bundle, data_texts, query, cfg.k_embed)
        return merge_scores_max_weighted(bm25, emb, cfg.bm25_weight, cfg.embed_weight, cfg.k)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


# =============================================================================
# 4) 생성 (LLM off 시 근거만)
# =============================================================================
def make_answer(
    bundle: RetrieverBundle, query: str, docs_scored: List[Tuple[str, float]], cfg: SearchConfig
) -> Tuple[str, List[Tuple[str, float]]]:
    if cfg.use_reranker and len(docs_scored) > 0:
        docs_scored = rerank_with_ko_reranker(bundle, query, docs_scored, top_k=cfg.reranker_top_k or cfg.k)

    top_texts = [d for d, _ in docs_scored[:cfg.k]]
    context = "\n".join(top_texts)

    if not cfg.enable_llm or bundle.GEN is None or prompting_answer is None:
        return "(LLM 비활성화: 근거 문서만 제공)", docs_scored[:cfg.k]

    # RAG 프롬프트 어셈블
    messages = prompting_answer(query, context, rag=True)
    prompt = "\n\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages]) + "\n\nAssistant:"

    # torch.cuda.empty_cache()
    # resp = bundle.GEN(prompt, num_return_sequences=1,max_new_tokens=256)[0]["generated_text"]
    try:
        resp = bundle.GEN(prompt, num_return_sequences=1,max_new_tokens=256)[0]["generated_text"]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("[OOM SKIP] Too large prompt or generation failed")
        resp = "생성 실패 (OOM)"

    answer = resp.split("Assistant:")[-1].strip()
    answer = re.sub(rf"^{re.escape(query)}[\s:：]*", "", answer).strip()
    if not answer:
        answer = "답변을 생성할 수 없습니다."
    return answer, docs_scored[:cfg.k]


# =============================================================================
# 5) 타깃 매칭 (제목+내용 동일 기준)
# =============================================================================
def find_target_in_used(target_text: str, used_docs: List[Tuple[str, float]]) -> Tuple[bool, Optional[int], Optional[float]]:
    norm_target = normalize_spaces(target_text)
    short_probe = norm_target[:200] if len(norm_target) > 200 else norm_target

    for idx, (doc, sc) in enumerate(used_docs):
        norm_doc = normalize_spaces(doc)
        if norm_doc == norm_target:
            return True, idx + 1, sc
        if short_probe and (short_probe in norm_doc or norm_doc[:200] in norm_target):
            return True, idx + 1, sc
    return False, None, None




# =============================================================================
# 6) 실행 루틴 (JSON → CSV)
# =============================================================================
def run_and_save_from_json(input_json_path: str, cfg: SearchConfig, out_path: str,
                           data_texts: List[str], bundle: RetrieverBundle,
                           start_idx: int = 0, end_idx: Optional[int] = None):
    with open(input_json_path, "r", encoding="utf-8") as f:
        qa_items = json.load(f)

    qa_items = qa_items[start_idx:end_idx] if end_idx else qa_items[start_idx:]

    print(f"[INFO] QA 항목 수: {len(qa_items)} (from {start_idx} to {end_idx or 'end'})")
    results = []

    iterator = tqdm(
        qa_items, desc=cfg.progress_desc, unit="q", dynamic_ncols=True, mininterval=0.2
    ) if cfg.show_progress else qa_items

    for item in iterator:
        qid = item.get("id")
        title = item.get("title", "")
        gold = item.get("answer", "")
        query = item["query"]
        target_content = item.get("content", "")

        target_full = compose_title_content(title, target_content, use_title=True)

        docs_scored = retrieve_docs(bundle, data_texts, query, cfg)
        pred, used = make_answer(bundle, query, docs_scored, cfg)

        target_found, target_rank, target_score = find_target_in_used(target_full, used)

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

        if cfg.show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                {"found": int(target_found), "rank": (target_rank if target_rank is not None else "-")},
                refresh=False,
            )

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[DONE] JSON 입력 결과가 '{out_path}'로 저장되었습니다.")


# =============================================================================
# 7) main
# =============================================================================
def main(start_idx: int = 0, end_idx: Optional[int] = None):
    # ---- 설정 ---
    CFG = SearchConfig(
        mode="bm25",
        k=10, k_bm25=10, k_embed=10,
        bm25_weight=0.6, embed_weight=0.4,
        use_reranker=False,
        reranker_top_k=9,
        enable_llm=True,
        show_progress=True,
        progress_desc="Evaluating",
        device_id=0,
        use_faiss_gpu=False,
        excel_path="/home/food/people/subin/data/식품안전정보DB-url 추가(2014~2024).xls",
        bm25_pickle="/home/food/people/minju/embedding/bm25/bm25_all.pkl",
        embeddings_pickle="/home/food/people/minju/embedding/labse2/embeddings_all.pkl",
        input_json="/home/food/people/minju/make_data/merged.json",
        output_csv="/home/food/people/minju/final/top5/bm25_10_answer_1000_1.csv",
    )

    # ---- 시드/디바이스 ---
    set_seed(42)
    device = get_device(CFG)
    print(f"[INFO] Using device: {device}")

    # ---- 데이터 ---
    df = load_food_sheets(CFG.excel_path, show_progress=CFG.show_progress)
    DATA = df["검색텍스트"].tolist()
    print(f"[INFO] 데이터 총 개수: {len(DATA)}")

    # ---- 번들(모델들) 로딩 ---
    bundle = RetrieverBundle()
    bundle.load_bm25(CFG.bm25_pickle)
    bundle.load_embeddings_and_faiss(CFG.embeddings_pickle, device, CFG.use_faiss_gpu)
    if CFG.use_reranker:
        bundle.load_reranker(device)
    if CFG.enable_llm:
        bundle.load_llm(device)

    # ---- 실행 ---
    run_and_save_from_json(CFG.input_json, CFG, CFG.output_csv, DATA, bundle,
                           start_idx=start_idx, end_idx=end_idx)


if __name__ == "__main__":
    print(os.environ.get('CUDA_VISIBLE_DEVICES'))
    import sys
    
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else None
    print(f"[LAUNCH] Processing QA from index {start} to {end}")
    main(start_idx=start, end_idx=end)


# # =============================================================================
# # 6) 실행 루틴 (JSON → CSV)
# # =============================================================================
# def run_and_save_from_json(input_json_path: str, cfg: SearchConfig, out_path: str,
#                            data_texts: List[str], bundle: RetrieverBundle):
#     with open(input_json_path, "r", encoding="utf-8") as f:
#         qa_items = json.load(f)

#     print(f"[INFO] QA 항목 수: {len(qa_items)}")
#     results = []

#     iterator = tqdm(
#         qa_items, desc=cfg.progress_desc, unit="q", dynamic_ncols=True, mininterval=0.2
#     ) if cfg.show_progress else qa_items

#     for item in iterator:
#         qid = item.get("id")
#         title = item.get("title", "")
#         gold = item.get("answer", "")
#         query = item["query"]
#         target_content = item.get("content", "")

#         target_full = compose_title_content(title, target_content, use_title=True)

#         docs_scored = retrieve_docs(bundle, data_texts, query, cfg)
#         pred, used = make_answer(bundle, query, docs_scored, cfg)

#         target_found, target_rank, target_score = find_target_in_used(target_full, used)

#         row = {
#             "id": qid,
#             "제목": title,
#             "질문": query,
#             "예상답변": pred,
#             "정답": gold,
#             "target_found": bool(target_found),
#             "target_rank": target_rank if target_rank is not None else "",
#             "target_score": target_score if target_score is not None else "",
#         }

#         for i in range(cfg.k):
#             if i < len(used):
#                 doc_i, sc_i = used[i]
#                 snippet = doc_i[:300].replace("\n", " ") + "..."
#                 row[f"문서{i+1}"] = f"[{i+1}] {snippet}"
#                 row[f"점수{i+1}"] = sc_i
#             else:
#                 row[f"문서{i+1}"] = ""
#                 row[f"점수{i+1}"] = ""

#         results.append(row)

#         if cfg.show_progress and hasattr(iterator, "set_postfix"):
#             iterator.set_postfix(
#                 {"found": int(target_found), "rank": (target_rank if target_rank is not None else "-")},
#                 refresh=False,
#             )

#     df_out = pd.DataFrame(results)
#     df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
#     print(f"[DONE] JSON 입력 결과가 '{out_path}'로 저장되었습니다.")


# # =============================================================================
# # 7) main
# # =============================================================================
# def main():
#     # ---- 설정 ---
#     CFG = SearchConfig(
#         mode="bm25",        # "embedding" / "bm25" / "hybrid_sum" / "hybrid_max"
#         k=20, k_bm25=10, k_embed=10,
#         bm25_weight=0.6, embed_weight=0.4,
#         use_reranker=True,        # 리랭커 on/off
#         reranker_top_k=10,
#         enable_llm=True,          # LLM on/off
#         show_progress=True,
#         progress_desc="Evaluating",
#         device_id=0,
#         use_faiss_gpu=False,
#         excel_path="/home/food/people/subin/data/식품안전정보DB-url 추가(2014~2024).xls",
#         bm25_pickle="/home/food/people/minju/embedding/bm25/bm25_all.pkl",
#         embeddings_pickle="/home/food/people/minju/embedding/labse2/embeddings_all.pkl",
#         input_json="/home/food/people/minju/make_data/merged.json",
#         output_csv="/home/food/people/minju/final/top5/bm25_20_answer_reranker.csv",
#     )

#     # ---- 시드/디바이스 ---
#     set_seed(42)
#     device = get_device(CFG)
#     print(f"[INFO] Using device: {device}")

#     # ---- 데이터 ---
#     df = load_food_sheets(CFG.excel_path, show_progress=CFG.show_progress)
#     DATA = df["검색텍스트"].tolist()
#     print(f"[INFO] 데이터 총 개수: {len(DATA)}")

#     # ---- 번들(모델들) 로딩 ---
#     bundle = RetrieverBundle()
#     bundle.load_bm25(CFG.bm25_pickle)
#     bundle.load_embeddings_and_faiss(CFG.embeddings_pickle, device, CFG.use_faiss_gpu)
#     if CFG.use_reranker:
#         bundle.load_reranker(device)
#     if CFG.enable_llm:
#         bundle.load_llm(device)

#     # ---- 실행 ---
#     run_and_save_from_json(CFG.input_json, CFG, CFG.output_csv, DATA, bundle)


# if __name__ == "__main__":
#     print(os.environ.get('CUDA_VISIBLE_DEVICES'))
#     main()
