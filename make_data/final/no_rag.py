# -*- coding: utf-8 -*-
"""
EXAONE 생성 능력 평가 (Retriever, gold 문서 없이 질문만 사용)
- JSON QA 파일에서 query, answer 읽기
- query만 넣어 EXAONE이 답변 생성
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
# 1) EXAONE 로드
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
# 2) 답변 생성 (질문만 사용)
# ==============================
def make_answer_from_query_only(gen, query: str):
    if prompting_answer is None:
        # 질문만 프롬프트에 넣기
        prompt = f"질문: {query}\n\n답변:"
    else:
        # prompting_answer 유틸이 있다면 gold_doc을 공백으로
        messages = prompting_answer(query, "", rag=False)
        prompt = "\n\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages]) + "\n\nAssistant:"

    torch.cuda.empty_cache()
    resp = gen(prompt, num_return_sequences=1, max_new_tokens=256)[0]["generated_text"]

    answer = resp.split("Assistant:")[-1].strip()
    if not answer:
        answer = "답변을 생성할 수 없습니다."
    return answer


# ==============================
# 3) 실행 루틴 (질문만)
# ==============================
def run_query_only_eval(input_json: str, output_csv: str, device: torch.device):
    with open(input_json, "r", encoding="utf-8") as f:
        qa_items = json.load(f)

    gen = load_exaone(device)

    results = []
    for item in tqdm(qa_items, desc="Evaluating", unit="q"):
        qid = item.get("id")
        query = item["query"]
        gold = item.get("answer", "")

        # gold 문서 없이 질문만 사용
        pred = make_answer_from_query_only(gen, query)

        results.append({
            "id": qid,
            "질문": query,
            "예상답변": pred,
            "정답": gold,
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] 결과가 '{output_csv}'로 저장되었습니다.")


# ==============================
# 4) main
# ==============================
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    input_json = "/home/food/people/minju/make_data/merged.json"
    output_csv = "/home/food/people/minju/final/exaone_query_only.csv"

    run_query_only_eval(input_json, output_csv, device)


if __name__ == "__main__":
    main()
