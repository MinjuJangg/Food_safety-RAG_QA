from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
import json
import time
import re
import torch

# ✅ 로컬 모델 경로
model_path = '/SSL_NAS/concrete/models/models--meta-llama--Meta-Llama-3-8B-Instruct/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e5e23bbe8e749ef0efcf16cad411a7d23bd23298'

# 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map=0)

# 파이프라인 구성 (응답만 반환하도록 설정)
llama_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

# 문자열 안전 처리 함수
def clean_text(text):
    return str(text).replace('"', "'").strip()

# 프롬프트 생성 함수 (LLaMA3 chat format 기반)
def make_chat_prompt(title, content):
    system_msg = (
        "너는 식품 위해 정보 문서 기반 QA 데이터를 생성하는 도우미야. "
        "주어진 문서에서 질문(Query)과 질문에 대한 답변(Answer) 쌍을 추출해야 해. "
        "특히 **가장 핵심적이거나 중요한 질문 1개만** 생성해야 하며, 이는 **사람이 이 문서를 봤을 때 가장 먼저 궁금해할 질문**이어야 해. "
        "모든 출력은 반드시 한국어로 작성하시오."
    )

    user_msg = f"""제목: {clean_text(title)}
내용: {clean_text(content)}

### 출력 형식
[
  {{"query": "질문", "answer": "문서 기반 답변"}}
]

### 생성 규칙
1. 질문은 반드시 주어진 문서 내용을 바탕으로 사람이 실제로 궁금해할 만한 자연스러운 문장으로 작성하세요.
2. 답변은 반드시 해당 문서에서 직접 확인 가능한 정보만 포함해야 합니다.
3. 문서에서 답이 명확히 주어지지 않은 경우, 답변은 "해당 문서에 없음"으로 작성하세요.
4. 질문은 완전한 문장으로 작성하세요. 단문이나 명령문 형태는 피하세요.
5. 질문에는 '이', '이번', '해당', '이 제품'과 같은 모호한 지시어 대신, 질문의 대상이 명확히 드러나도록 하세요. 다만, 질문이 과도하게 길어지지 않도록 제품명이나 기관명 등은 간결하게 필요한 만큼만 포함하세요.
6. 질문(query)에 답변(answer)의 내용이 포함되지 않도록 합니다.
"""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# 🔍 모델 응답에서 JSON 배열만 추출
def extract_json_from_response(response_text):
    try:
        match = re.search(r"\[\s*{.*?}\s*]", response_text, re.DOTALL)
        if match:
            json_str = match.group()
            return json.loads(json_str)
        else:
            raise ValueError("JSON 형식의 QA 쌍을 찾을 수 없습니다.")
    except Exception as e:
        print("JSON 파싱 실패:", e)
        return []

# 🧠 LLaMA를 이용한 QA 추출
def get_qa_from_document(title, content):
    prompt = make_chat_prompt(title, content)

    try:
        result = llama_pipeline(prompt, max_new_tokens=512, do_sample=True, temperature=0.5)[0]["generated_text"]
        response_text = result.strip()
        print("=== 원본 응답 ===")
        print(response_text)

        qa_list = extract_json_from_response(response_text)
        print("=== 추출된 QA ===")
        print(qa_list)

        return qa_list

    except Exception as e:
        print("QA 생성 실패:", e)
        print("원본 응답:", result if 'result' in locals() else "없음")
        return []

# CSV 파일 불러오기
file_path = "/home/food/people/minju/data/make_qa/real_final.csv"
df = pd.read_csv(file_path)

# QA 결과 컬럼 초기화
df["llama_query"] = ""
df["llama_answer"] = ""

# 루프 돌며 QA 생성
for i in range(len(df)):
    print(f"\n\\ [{i+1}/{len(df)}] QA 생성 중...")

    title = df.loc[i, "title"] if "title" in df.columns else ""
    if "document" in df.columns:
        text = df.loc[i, "document"]
    elif "extracted_text" in df.columns:
        text = df.loc[i, "extracted_text"]
    else:
        print("문서 내용이 없습니다.")
        continue

    if pd.isna(text) or len(str(text).strip()) < 5:
        print("문서가 비어 있음. 스킵합니다.")
        continue

    qa_pairs = get_qa_from_document(title, text)

    if qa_pairs:
        queries = [pair["query"] for pair in qa_pairs]
        answers = [pair["answer"] for pair in qa_pairs]
        df.at[i, "llama_query"] = "\n".join(queries)
        df.at[i, "llama_answer"] = "\n".join(answers)

    time.sleep(1)

# 결과 저장
df.to_csv("real_final_with_llama_QA.csv", index=False)
print("\n저장 완료: real_final_with_llama_QA.csv")
