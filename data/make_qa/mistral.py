import re
import json
from huggingface_hub import login
import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import GenerationConfig

# Hugging Face 로그인
login(token="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")

# 모델명
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=2
)

# 파이프라인 정의
gen_config = GenerationConfig(
    temperature=0.4,
    top_p=0.7,
    top_k=40,
    max_new_tokens=1024,
)

mistral_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    generation_config=gen_config
)

# 텍스트 전처리 함수
def clean_text(text):
    return str(text).replace("\n", " ").strip()

# Chat 형식 프롬프트 생성
def make_chat_prompt(title, content):
    system_msg = (
        "너는 QA 데이터를 생성하는 도우미야. "
        "주어진 문서에서 한국어 질문(Query)과 답변(Answer) 쌍을 추출해야 해. "
        "특히 **가장 핵심적이거나 중요한 질문 1개만** 생성해야 하며, "
        "질문에 '이 문서에서'등과 같은 단어 사용하지 말고, 질문의 대상을 명확히 표현해."
    )

    user_msg = f"""제목: {clean_text(title)}
            내용: {clean_text(content)}

### 출력 형식
[
  {{"query": "가장 궁금한 핵심 질문 1개", "answer": "해당 질문에 대한 답변"}}
]

### 생성 규칙
1. 질문은 반드시 주어진 문서 내용을 바탕으로 사람이 실제로 궁금해할 만한 자연스러운 문장으로 작성하세요. 
2. 답변은 반드시 해당 문서에서 직접 확인 가능한 정보만 포함해야 합니다. 문서에 없는 내용을 추측하거나 외삽하지 마세요.
3. 문서에서 답이 명확히 주어지지 않은 경우, 답변은 "해당 문서에 없음"으로 작성하세요.
4. 한 문서당 **하나의** 한국어 질문-답변 쌍만 생성하세요.
5. **질문(query)에는 '이번', '해당', '이 제품', '이', '이 문서에서'와 같은 모호한 지시어를 사용하지 말고, 질문의 대상을 명확히 표현하세요. **
6. 하나의 질문에 너무 많은 요소를 담지 마세요.
7. 질문은 완전한 문장으로 작성하세요.
8. 질문 문장은 유사하거나 반복되지 않도록 하며, 사람스러운 문장 흐름을 갖추세요.
9. 질문은 서로 다른 정보에 초점을 맞춰 다양하게 구성하세요. 
10. 질문에 '이 문서에서'라는 단어 사용하지마세요.
"""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ✅ 응답에서 JSON 추출
def extract_json_from_response(response_text):
    try:
        cleaned = response_text.strip()
        cleaned = re.sub(r"\[/?INST\]", "", cleaned)
        cleaned = re.sub(r"<\|.*?\|>", "", cleaned)
        cleaned = re.sub(r"```json|```", "", cleaned).strip()

        match = re.search(r"\[\s*{.*?}\s*]", cleaned, re.DOTALL)
        if match:
            json_str = match.group()
            return json.loads(json_str)
        else:
            raise ValueError("JSON 배열을 찾을 수 없습니다.")
    except Exception as e:
        print("❌ JSON 파싱 실패:", e)
        print("⛔ 원본 응답:", repr(response_text[:300]))
        return []

# QA 생성 함수
def get_qa_from_document(title, content):
    try:
        chat_prompt = make_chat_prompt(title, content)
        response = mistral_pipeline(chat_prompt, max_new_tokens=512, do_sample=True, temperature=0.7)[0]["generated_text"]

        qa_list = extract_json_from_response(response)
        print(qa_list)
        return qa_list
    except Exception as e:
        print("QA 생성 실패:", e)
        return []

# CSV 파일 불러오기
file_path = "/home/food/people/minju/data/make_qa/real_final.csv"
df = pd.read_csv(file_path)

# QA 컬럼 초기화
df["mistral_query"] = ""
df["mistral_answer"] = ""

# QA 생성 루프
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
        df.at[i, "mistral_query"] = "\n".join(queries)
        df.at[i, "mistral_answer"] = "\n".join(answers)

    time.sleep(5)

# 결과 저장
df.to_csv("real_final_with_mistral_QA.csv", index=False)
print("\n저장 완료: real_final_with_mistral_QA.csv")

# import re
# import json
# import time
# import torch
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from huggingface_hub import login


# # ✅ 모델 및 토크나이저 로드
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

# # ✅ 파이프라인 설정
# mistral_pipeline = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     return_full_text=False
# )

# # ✅ 텍스트 전처리
# def clean_text(text):
#     return str(text).replace('"', "'").replace("\n", " ").strip()

# # ✅ 프롬프트 생성 (Instruction 방식)
# def make_instruction_prompt(title, content):
#     prompt = f"""너는 문서 기반 QA 데이터를 생성하는 도우미야. 주어진 문서에서 질문(Query)과 답변(Answer) 쌍을 추출해야 해. 특히 **가장 핵심적이거나 중요한 질문 1개만** 생성해야 하며, 이는 **사람이 이 문서를 봤을 때 가장 먼저 궁금해할 질문**이어야 해.

# **문서:**
# 제목: {clean_text(title)}
# 내용: {clean_text(content)}

# ### 출력 형식
# [
#   {{"query": "가장 궁금한 핵심 질문 1개", "answer": "해당 질문에 답변"}}
# ]

# ### 생성 규칙
# 1. 질문은 반드시 주어진 문서 내용을 바탕으로 사람이 실제로 궁금해할 만한 자연스러운 문장으로 작성하세요. 인위적인 패턴을 반복하지 마세요.
# 2. 답변은 반드시 해당 문서에서 직접 확인 가능한 정보만 포함해야 합니다. 문서에 없는 내용을 추측하거나 외삽하지 마세요.
# 3. 문서에서 답이 명확히 주어지지 않은 경우, 답변은 "해당 문서에 없음"으로 작성하세요.
# 4. 한 문서당 한개의 질문-답변 쌍만 생성하세요.
# 5. 질문은 서로 다른 정보에 초점을 맞춰 다양하게 구성하세요. 
#    - 예: 제품명, 제조사, 유통 경로, 일자, 회수 사유, 법 위반 항목, 대표자 등
# 6. 하나의 질문에 너무 많은 요소를 담지 마세요.
# 7. 질문은 완전한 문장으로 작성하세요.
# 8. 질문 문장은 유사하거나 반복되지 않도록 하며, 사람스러운 문장 흐름을 갖추세요.
# 9. 질문에는 반드시 질문 대상을 직접 명시하세요. '이번', '이 제품', '해당 제품', '해당 문서', '문서' 처럼 모호한 표현을 쓰면 안 됩니다.
# """
#     return prompt

# # ✅ JSON 응답 추출 함수
# def extract_json_from_response(response_text):
#     try:
#         cleaned = response_text.strip()
#         cleaned = re.sub(r"```json|```", "", cleaned)
#         match = re.search(r"\[\s*{.*?}\s*]", cleaned, re.DOTALL)
#         if match:
#             return json.loads(match.group())
#         else:
#             raise ValueError("JSON 배열을 찾을 수 없습니다.")
#     except Exception as e:
#         print("❌ JSON 파싱 실패:", e)
#         print("⛔ 원본 응답:", repr(response_text[:300]))
#         return []

# # ✅ QA 생성 함수
# def get_qa_from_document(title, content):
#     try:
#         prompt = make_instruction_prompt(title, content)
#         response = mistral_pipeline(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)[0]["generated_text"]
#         return extract_json_from_response(response)
#     except Exception as e:
#         print("QA 생성 실패:", e)
#         return []

# # ✅ 데이터 불러오기
# file_path = "/home/food/people/minju/data/make_qa/real_final.csv"
# df = pd.read_csv(file_path)

# # ✅ 빈 컬럼 생성
# df["mistral_query"] = ""
# df["mistral_answer"] = ""

# # ✅ 문서별 QA 생성
# for i in range(len(df)):
#     print(f"\n[{i+1}/{len(df)}] QA 생성 중...")

#     title = df.loc[i, "title"] if "title" in df.columns else ""
#     if "document" in df.columns:
#         text = df.loc[i, "document"]
#     elif "extracted_text" in df.columns:
#         text = df.loc[i, "extracted_text"]
#     else:
#         print("문서 내용 없음. 스킵.")
#         continue

#     if pd.isna(text) or len(str(text).strip()) < 5:
#         print("문서가 비어 있음. 스킵.")
#         continue

#     qa_pairs = get_qa_from_document(title, text)
#     print(qa_pairs)

#     if qa_pairs:
#         queries = [pair["query"] for pair in qa_pairs]
#         answers = [pair["answer"] for pair in qa_pairs]
#         df.at[i, "mistral_query"] = "\n".join(queries)
#         df.at[i, "mistral_answer"] = "\n".join(answers)

#     time.sleep(3)  # 모델 속도 조절

# # ✅ 결과 저장
# df.to_csv("real_final_with_mistral_QA.csv", index=False)
# print("\n✅ 저장 완료: real_final_with_mistral_QA.csv")

