from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
import pandas as pd
import json
import time
import re
import torch

# 모델 로드
model_name = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
gen_config = GenerationConfig(
    temperature=0.5,
    top_p=0.7,
    top_k=40,
    max_new_tokens=1024,
    do_sample=True
)

gemma_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    generation_config=gen_config
)

# 텍스트 정리
def clean_text(text):
    return str(text).replace('"', "'").strip()

# 💬 Gemini 스타일 단일 프롬프트 생성
def make_prompt(title, content):
    return f"""너는 문서 기반 QA 데이터를 생성하는 도우미야. 주어진 문서에서 질문(Query)과 질문에 대한 답변(Answer) 쌍을 추출해야 해. 특히 **가장 핵심적이거나 중요한 질문 1개만** 생성해야 하며, 이는 **사람이 이 문서를 봤을 때 가장 먼저 궁금해할 질문**이어야 해. 모든 출력은 반드시 한국어로 작성하시오.

제목: {clean_text(title)}
내용: {clean_text(content)}

### 출력 형식
[
 { {"query": "실제 문서 기반 질문", "answer": "문서에서 확인된 답변"}}
]

### 생성 규칙
1. 질문은 반드시 주어진 문서 내용을 바탕으로 사람이 실제로 궁금해할 만한 자연스러운 문장으로 작성하세요. 인위적인 패턴을 반복하지 마세요.
2. 답변은 반드시 해당 문서에서 직접 확인 가능한 정보만 포함해야 합니다. 문서에 없는 내용을 추측하거나 외삽하지 마세요.
3. 문서에서 답이 명확히 주어지지 않은 경우, 답변은 "해당 문서에 없음"으로 작성하세요.
4. 한 문서당 하나의 질문-답변 쌍만 생성하세요.
5. 질문은 서로 다른 정보에 초점을 맞춰 다양하게 구성하세요. 
   - 예: 제품명, 제조사, 유통 경로, 일자, 회수 사유, 법 위반 항목, 대표자 등
6. 하나의 질문에 너무 많은 요소를 담지 마세요. 질문은 간결하고 명확하게 하나의 정보만 물어보도록 하세요.
7. 질문은 완전한 문장으로 작성하세요. 단문이나 명령문 형태는 피하세요.
8. 질문 문장은 유사하거나 반복되지 않도록 하며, 사람스러운 문장 흐름을 갖추세요.
9. '회수 사유는 무엇인가요?' 형식의 질문을 반복하지 마세요. 표현을 다양화하여 사람다운 말투로 바꾸세요.
10. 질문에는 '이번', '해당', '이 제품'과 같은 모호한 지시어 대신, 질문의 대상이 명확히 드러나도록 하세요. 다만, 질문이 과도하게 길어지지 않도록 제품명이나 기관명 등은 간결하게 필요한 만큼만 포함하세요.
11. 주어진 제목, 내용 외 예시를 생성하지 마시오.
12. 출력 형식 외 다른 내용은 출력하지 마시오.
"""

# 🔍 모델 응답에서 JSON 배열만 추출
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

# 🧠 Gemma를 이용한 QA 추출
def get_qa_from_document(title, content):
    prompt = make_prompt(title, content)
    print("====prompt")
    print(prompt)
    try:
        result = gemma_pipeline(prompt)[0]["generated_text"]
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

# QA 컬럼 추가
df["gemma_query"] = ""
df["gemma_answer"] = ""

# 루프 QA 생성
for i in range(len(df)):
    print(f"\n\\ [{i+1}/{len(df)}] QA 생성 중...")

    title = df.loc[i, "title"] if "title" in df.columns else ""
    text = df.get("document", df.get("extracted_text", ""))[i]

    if pd.isna(text) or len(str(text).strip()) < 5:
        print("문서가 비어 있음. 스킵합니다.")
        continue

    qa_pairs = get_qa_from_document(title, text)

    if qa_pairs:
        queries = [pair["query"] for pair in qa_pairs]
        answers = [pair["answer"] for pair in qa_pairs]
        df.at[i, "gemma_query"] = "\n".join(queries)
        df.at[i, "gemma_answer"] = "\n".join(answers)

    time.sleep(1)

# 결과 저장
df.to_csv("real_final_with_gemma_QA.csv", index=False)
print("\n저장 완료: real_final_with_gemma_QA.csv")
