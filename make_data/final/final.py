import os
import random
import pickle
import numpy as np
import faiss
import torch
import pandas as pd
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from langchain_teddynote.retrievers import KiwiBM25Retriever
from llm import set_seed, set_llm, prompting_answer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
# 
# ko-reranker 모델 로드
# reranker_model_name = "Dongjin-kr/ko-reranker"
# reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
# reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
# reranker_model.eval()


# 시드 고정
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

# 데이터 로드
file_path = "/home/food/people/subin/data/식품안전정보DB-url 추가(2014~2024).xls"
df = pd.read_excel(file_path, sheet_name="2023", usecols=["제목", "내용"])
df["제목_내용"] = df["제목"] + " " + df["내용"]
data = df["제목_내용"].to_list()

# KiwiBM25 로드
with open("/home/food/people/minju/embedding/bm25/bm25_2023.pkl", "rb") as f:   #./식품안전/kiwi_index.pkl
    kiwi_retriever = pickle.load(f)
kiwi_retriever.k = 5

# FAISS 로드
with open("/home/food/people/subin/data/embeddings/LaBSE/embeddings_2023.pkl", "rb") as f:
    embeddings = pickle.load(f)

index = faiss.IndexFlatIP(embeddings.shape[1])
faiss.normalize_L2(embeddings)
index.add(embeddings)

embedding_model = SentenceTransformer("sentence-transformers/LaBSE")

# EXAONE 모델 로드
# exaone_model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
# tokenizer_exaone = AutoTokenizer.from_pretrained(exaone_model_name, trust_remote_code = True, timeout=60)
# bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
#                                 bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, llm_int8_enable_fp32_cpu_offload=True)
# exaone_model = AutoModelForCausalLM.from_pretrained(
#     exaone_model_name, device_map="auto", quantization_config=bnb_config, trust_remote_code = True
# )
# generator = pipeline("text-generation", model=exaone_model, tokenizer=tokenizer_exaone, max_new_tokens=300)

def rerank_with_ko_reranker(query, docs, top_k=None):
    # query-doc 쌍 만들기
    pairs = [[query, doc] for doc in docs]

    # 토크나이즈
    inputs = reranker_tokenizer(pairs, padding=True, truncation=True,
                                return_tensors='pt', max_length=512)

    with torch.no_grad():
        outputs = reranker_model(**inputs)
        scores = outputs.logits.view(-1).numpy()

    # softmax 정규화 (선택사항)
    # scores = np.exp(scores - np.max(scores))
    # scores = scores / scores.sum()

    # 스코어로 정렬
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    
    if top_k:
        reranked = reranked[:top_k]
    
    return reranked


def hybrid_search_with_answer(query, data, k=10, bm25_weight=0.6, faiss_weight=0.4):
    bm25_docs = kiwi_retriever.get_relevant_documents(query)
    bm25_result = {doc.page_content: bm25_weight * (len(bm25_docs) - rank)
                for rank, doc in enumerate(bm25_docs)}

    query_emb = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(query_emb, 5)
    faiss_result = {data[idx]: faiss_weight * (len(indices[0]) - rank)
                    for rank, idx in enumerate(indices[0])}

    merged_result = bm25_result.copy()
    for doc, score in faiss_result.items():
        merged_result[doc] = max(merged_result.get(doc, 0), score)

    sorted_docs = sorted(merged_result.items(), key=lambda x: x[1], reverse=True)
    top_ranked_docs = sorted_docs[:k]
    top_docs = [doc for doc, _ in top_ranked_docs]
    
    reranked_docs = top_docs
    # reranked_docs = rerank_with_ko_reranker(query, top_docs_raw, top_k=k)
    
   
   # context_ori = "\n".join(top_docs)
    context =  "\n".join(top_docs)
    #context = "\n".join([doc for doc, _ in reranked_docs])

    # 프롬프트 생성 (이 부분만 바뀜)
    messages = prompting_answer(query, context, rag=True)
    # 메시지를 단일 prompt로 변환 (exaone이 Chat 모델이 아니므로 텍스트 통합 필요)
    prompt = "\n\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
    prompt += "\n\nAssistant:"

    # 응답 생성
    response = generator(prompt, num_return_sequences=1)[0]["generated_text"]

    # 출력 정제
    answer = response.split("Assistant:")[-1].strip()
    answer = re.sub(rf"^{re.escape(query)}[\s:：]*", "", answer).strip()
    if answer == "":
        answer = "답변을 생성할 수 없습니다."

    return answer, reranked_docs
def hybrid_search_retriever_only(query, data, k=10, bm25_weight=0.6, faiss_weight=0.4):
    # BM25 검색
    bm25_docs = kiwi_retriever.get_relevant_documents(query)
    bm25_result = {doc.page_content: bm25_weight * (len(bm25_docs) - rank)
                   for rank, doc in enumerate(bm25_docs)}

    # FAISS 검색
    query_emb = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(query_emb, 5)
    faiss_result = {data[idx]: faiss_weight * (len(indices[0]) - rank)
                    for rank, idx in enumerate(indices[0])}

    # Max 방식으로 점수 병합
    merged_result = bm25_result.copy()
    for doc, score in faiss_result.items():
        merged_result[doc] = max(merged_result.get(doc, 0), score)

    # 정렬 및 반환
    sorted_docs = sorted(merged_result.items(), key=lambda x: x[1], reverse=True)
    top_docs_scored = sorted_docs[:k]  # (doc, score)

    return top_docs_scored


# 실행 예시
query_list = ["Waitrose 8 Red Onion Bhajis with a Date and Tamarind Dip 제품이 왜 회수되었나?", 
"Hassui Kamaboko Co.,Ltd.에서 회수한 제품의 회수 사유는?",
"프랑스 Ducourau사의 굴이 회수된 이유는?",
"말레이시아 트렝가누주에서 발생한 식중독 사망 원인은?", 
"Arauco 올리브유가 판매금지된 이유는?",
"벨기에에서 Isla Délice 식육가공품이 회수된 이유는?",
"영국 Country Kitchen에서 회수한 머핀의 제품명은?",
"Le Duo des Gors 치즈가 회수된 이유는?",
"뉴질랜드에서 Value brand 탄산음료가 회수된 이유는?",
"미국에서 TGD Cuts, LLC가 회수한 과일의 오염 가능성이 있는 병원균은?",
"FishMeatz LLP가 벌금형을 받은 이유는 무엇인가?",
"Rude Health Organic Coconut Drink가 회수된 이유는 무엇인가?",
"New Roots Herbal의 아슈와간다 제품이 회수된 이유는 무엇인가?",
"벨기에에서 BERGERONNETTE Pérail du Fédou 치즈가 회수된 이유는 무엇인가?",
"루샤오자 식품 유한공사의 오향 닭날개가 부적합 판정을 받은 이유는 무엇인가?",
"GDE Grocery Delivery E-Services Canada Inc.가 회수한 닭고기 제품의 유통지역은 어디인가?",
"이탈리아에서 수출한 신선 여름 송로버섯의 카드뮴 함량은 얼마인가?",
"Mrs Kirkham 치즈가 회수된 이유는 무엇인가?",
"코스타리카 내 리스테리아증 감염 사례는 주로 어떤 식품과 관련이 있는가?",
"쓰촨 촨라오라오 식품 과학기술유한공사의 식용 식물 혼합유에서 검출된 부적합 물질은 무엇인가?",
"베트남 하노이 시장관리국은 어떤 불법 행위를 적발했는가?",
"일본 농림수산성이 칠레산 가금육 등의 수입중지 조치를 해제한 이유는 무엇인가?",
"최근 세계동물보건기구(WOAH)가 보고한 고병원성 조류인플루엔자(H5N1) 발생 현황은 어떠한가?",
"칠레에서 보고된 조류인플루엔자 A(H5) 인체 감염 사례의 주요 내용은 무엇인가?",
"스위스 연방평의회는 화학물질 및 폐기물 협약 강화와 관련하여 어떤 조치를 취하고 있는가?",
"벨기에 연방보건부는 아스파탐의 1일허용섭취량(ADI)을 변경하지 않은 이유는 무엇인가?",
"미국 연구진의 연구에 따르면, 카드뮴 식이 노출이 가장 높은 연령대와 주요 노출 식품은 무엇인가?",
"일본 유한회사 밸런스가 마들렌 제품을 회수한 이유와 해당 제품의 판매 정보는 무엇인가?",
"중국 해관총서와 농업농촌부는 터키에서 발생한 가성우역의 유입을 방지하기 위해 어떤 조치를 시행했는가?",
"대만 신베이시에서 적발된 가짜 양고기 판매 사건의 주요 내용은 무엇인가?",
"Springbank Cheese Co.와 Le Grand Fromage에서 회수된 치즈 제품은 무엇이며, 회수 사유는 무엇인가요?",
"스페인 식품안전영양청(AESAN)은 아일랜드산 자숙 냉동게에 대한 경고를 왜 철회했나요?",
"영국 환경식품농촌부(Defra)가 안내한 국경 목표운영모델(Border Target Operating Model)은 무엇인가?",
"나이지리아산 히비스커스 꽃에서 검출된 미승인 물질은 무엇인가?",
"일본에서 회수된 '팔도 꼬꼬면'의 회수 사유는 무엇인가?",
"대만 식약서가 일본산 수입식품의 방사능 검사를 중단한 품목은 무엇인가?",
"네팔에서 보고된 H5N2 고병원성 조류 인플루엔자의 발생 규모는 어떻게 되는가?",
"미국 식품의약품청이 'PrimeZen Black 6000' 제품에 대해 경고한 이유는 무엇인가?",
"독일 CVUA가 서양 송로버섯을 포함한 제품에서 집중적으로 모니터링하는 이유는 무엇인가?",
"미국 식품안전검사국이 공중보건경보를 발령한 냉동 닭고기 제품의 제조사는 어디인가?"]
# for query in query_list:
#     answer, docs = hybrid_search_with_answer(query, data, k=10)
#     print("\n질문:", query)
#     print("답변:", answer)
#     print("검색된 문서:")
#     for i, (doc,score) in enumerate(docs, 1):
#         print(f"[{i}] (점수: {score:.3f})\n{doc[:300]}...\n")
##############################################################################################


results = []

for query in query_list:
    docs = hybrid_search_retriever_only(query, data, k=10)

    doc_snippets = []
    doc_scores = []
    for i, (doc, score) in enumerate(docs, 1):
        snippet = doc[:300].replace("\n", " ")  # 줄바꿈 제거
        doc_snippets.append(f"[{i}] {snippet}...")
        doc_scores.append(score)

    result_row = {
        "질문": query,
        "답변": "LLM 비활성화 - 리트리버 결과만 출력"
    }
    for i in range(10):
        result_row[f"문서{i+1}"] = doc_snippets[i] if i < len(doc_snippets) else ""
        result_row[f"점수{i+1}"] = doc_scores[i] if i < len(doc_scores) else ""

    results.append(result_row)

# 결과 저장
df_result = pd.DataFrame(results)
output_path = "./retriever_only_result.csv"
df_result.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n[완료] 리트리버 결과가 '{output_path}'로 저장되었습니다.")


# results = []

# for query in query_list:
    
#     answer, docs = hybrid_search_with_answer(query, data, k=10)

#     doc_snippets = []
#     doc_scores = []
#     for i, doc in enumerate(docs, 1):
#     # for i, (doc, score) in enumerate(docs, 1):
#         snippet = doc[:300].replace("\n", " ")  # 줄바꿈 제거
#         doc_snippets.append(f"[{i}] {snippet}...")
#         # doc_scores.append(score)

#     result_row = {
#         "질문": query,
#         "답변": answer
#     }
#     for i in range(10):
#         result_row[f"문서{i+1}"] = doc_snippets[i] if i < len(doc_snippets) else ""
#         result_row[f"점수{i+1}"] = doc_scores[i] if i < len(doc_scores) else ""

#     results.append(result_row)

# # DataFrame으로 변환
# df_result = pd.DataFrame(results)

# # 저장
# output_path = "./qa_result_10.csv"
# df_result.to_csv(output_path, index=False, encoding="utf-8-sig")

# print(f"\n 결과가 '{output_path}'로 저장되었습니다.")
