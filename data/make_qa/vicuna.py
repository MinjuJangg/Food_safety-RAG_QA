from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
import pandas as pd
import json
import time
import re
import torch

# ✅ Vicuna 모델 로드
model_name = "lmsys/vicuna-7b-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

gen_config = GenerationConfig(
    temperature=0.25,
    top_p=0.7,
    top_k=40,
    max_new_tokens=1024,
    do_sample=True
)

vicuna_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    generation_config=gen_config
)

# 텍스트 정리
def clean_text(text):
    return str(text).replace('"', "'").strip()

# ✅ Vicuna 스타일 프롬프트 + Gemma 출력 규칙 유지
def make_prompt(title, content):
    return f"""### Instruction:
너는 문서 기반 QA 데이터를 생성하는 도우미야.  
주어진 문서에서 질문(Query)과 질문에 대한 답변(Answer) 쌍을 추출해야 해.  
특히 **가장 핵심적이거나 중요한 질문 1개만** 생성해야 하며, 이는 **사람이 이 문서를 봤을 때 가장 먼저 궁금해할 질문**이어야 해.  
**모든 출력은 반드시 한국어로 작성하시오.**
**질문에는 '이번', '해당', '이 제품'과 같은 모호한 지시어 대신, 질문의 대상이 명확히 드러나도록 하세요.  **

출력 생성 시 아래 규칙을 모두 반드시 따르시오:
1. 답변은 반드시 해당 문서에서 직접 확인 가능한 정보만 포함해야 합니다. 문서에 없는 내용을 추측하거나 외삽하지 마세요.  
2. 문서에서 답이 명확히 주어지지 않은 경우, 답변은 "해당 문서에 없음"으로 작성하세요.  
3. 질문에는 '이번', '해당', '이 제품'과 같은 모호한 지시어 대신, 질문의 대상이 명확히 드러나도록 하세요. 


출력은 반드시 다음 JSON 형식을 따라 단 한개의 질문 및 답변을 생성하세요:  
[
  {{"query": "질문", "answer": "답변"}}
]

### Input:
제목: {clean_text(title)}
내용: {clean_text(content)}
"""

# JSON 배열만 추출
def extract_json_from_response(response_text):
    try:
        match = re.search(r"\[\s*{.*?}\s*]", response_text, re.DOTALL)
        if match:
            json_str = match.group().replace("'", '"')
            return json.loads(json_str)
        else:
            raise ValueError("JSON 형식의 QA 쌍을 찾을 수 없습니다.")
    except Exception as e:
        print("JSON 파싱 실패:", e)
        return []

# QA 생성 함수
def get_qa_from_document(title, content):
    prompt = make_prompt(title, content)
    print("====prompt")
    print(prompt)
    try:
        result = vicuna_pipeline(prompt)[0]["generated_text"]
        response_text = result.strip()
        print("=== 원본 응답 ===")
        print(response_text)

        qa_list = extract_json_from_response(response_text)
        print("=== 추출된 QA ===")
        print(qa_list)

        return qa_list, response_text

    except Exception as e:
        print("QA 생성 실패:", e)
        return [], ""

# CSV 불러오기
file_path = "/home/food/people/minju/data/make_qa/real_final.csv"
df = pd.read_csv(file_path)

# QA 컬럼 추가
df["vicuna_query"] = ""
df["vicuna_answer"] = ""

# 루프 QA 생성
for i in range(len(df)):
    print(f"\n\\ [{i+1}/{len(df)}] QA 생성 중...")

    title = df.loc[i, "title"] if "title" in df.columns else ""
    text = df.get("document", df.get("extracted_text", ""))[i]

    if pd.isna(text) or len(str(text).strip()) < 5:
        print("문서가 비어 있음. 스킵합니다.")
        continue

    qa_pairs, raw_response = get_qa_from_document(title, text)

    if qa_pairs:
        df.at[i, "vicuna_query"] = qa_pairs[0]["query"]
        df.at[i, "vicuna_answer"] = qa_pairs[0]["answer"]
    else:
        # QA 추출 실패 시 원본 응답 저장
        df.at[i, "vicuna_answer"] = raw_response

    time.sleep(1)

# 결과 저장
df.to_csv("real_final_with_vicuna_QA.csv", index=False)
print("\n저장 완료: real_final_with_vicuna_QA.csv")
