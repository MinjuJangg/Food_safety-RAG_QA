# -*- coding: utf-8 -*-
"""
EXAONE 생성 능력 평가 (Retriever 없이 gold 문서만 사용)
- JSON QA 파일에서 query, title, content, answer 읽기
- gold 문서(제목+내용)를 context로 넣어 EXAONE이 답변 생성
- 결과를 CSV로 저장
"""

import os
import re
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

# ---- (선택) 프롬프트 유틸 불러오기 ----
try:
    from llm import prompting_answer
except Exception:
    prompting_answer = None

# ---- tqdm (진행률 표시) ----
try:
    from tqdm.auto import tqdm
except Exception:  # fallback
    def tqdm(x, *args, **kwargs):
        return x


# ==============================
# 1) 제목+내용 결합 유틸
# ==============================
def compose_title_content(title: str, content: str, use_title: bool = True) -> str:
    title = str(title) if title else ""
    content = str(content) if content else ""
    return (title + " " + content).strip() if use_title else content.strip()


# ==============================
# 2) EXAONE 로드
# ==============================
def load_exaone(device: torch.device):
    model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    tokenizer_exaone = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, timeout=60)

    if "cuda" in str(device):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True
        )
        exaone_model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True
        )
    else:
        exaone_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)

    gen = pipeline(
        "text-generation",
        model=exaone_model,
        tokenizer=tokenizer_exaone,
        max_new_tokens=300,
    )
    return gen


# ==============================
# 3) 답변 생성 (gold 문서만 사용)
# ==============================
def make_answer_from_gold(gen, query: str, gold_doc: str):
    if prompting_answer is None:
        # 단순 프롬프트
        prompt = f"질문: {query}\n\n문서:\n{gold_doc}\n\n답변:"
        
    else:
        # 기존 RAG 프롬프트 유틸 활용
        messages = prompting_answer(query, gold_doc, rag=True)
        prompt = "\n\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages]) + "\n\nAssistant:"
    # print(prompt)
    torch.cuda.empty_cache()
    resp = gen(prompt, num_return_sequences=1, max_new_tokens=256)[0]["generated_text"]

    answer = resp.split("Assistant:")[-1].strip()
    answer = re.sub(rf"^{re.escape(query)}[\s:：]*", "", answer).strip()
    if not answer:
        answer = "답변을 생성할 수 없습니다."
    return answer


# ==============================
# 4) 실행 루틴
# ==============================
def run_gold_eval(input_json: str, output_csv: str, device: torch.device):
    with open(input_json, "r", encoding="utf-8") as f:
        qa_items = json.load(f)

    gen = load_exaone(device)

    results = []
    for item in tqdm(qa_items, desc="Evaluating", unit="q"):
        qid = item.get("id")
        query = item["query"]
        gold = item.get("answer", "")
        title = item.get("title", "")
        content = item.get("content", "")

        # 제목+내용 결합
        gold_doc = compose_title_content(title, content, use_title=True)

        pred = make_answer_from_gold(gen, query, gold_doc)

        results.append({
            "id": qid,
            "질문": query,
            "예상답변": pred,
            "정답": gold,
            "제목": title,
            "gold문서": gold_doc[:300] + "..."
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] 결과가 '{output_csv}'로 저장되었습니다.")


# ==============================
# 5) main
# ==============================
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    input_json = "/home/food/people/minju/make_data/merged.json"
    output_csv = "/home/food/people/minju/final/exaone_gold_title_content.csv"

    run_gold_eval(input_json, output_csv, device)


if __name__ == "__main__":
    main()
