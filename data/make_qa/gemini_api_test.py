from textstat import flesch_reading_ease, flesch_kincaid_grade, gunning_fog
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import language_tool_python
import google.generativeai as genai
import pandas as pd
import ast
import time
import re
import json

# 문법 검사 도구
tool = language_tool_python.LanguageTool('en-US')

# Gemini API 설정
API_KEY = "AIzaSyAgML3TgOUqjkEympxIV_ROqqIaHqeHMIU"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config={
        "temperature": 0.5,
        "top_p": 0.7,
        "top_k": 40,
        "max_output_tokens": 1024,
    }
)

# 안전한 문자열 처리 함수
def clean_text(text):
    return str(text).replace('"', "'").strip()

# Gemini를 이용한 QA 추출 함수
def get_qa_from_document(title, content):
    prompt = f"""  너는 문서 기반 QA 데이터를 생성하는 도우미야. 주어진 문서에서 질문(Query)과 답변(Answer) 쌍을 추출해야 해. 특히 **가장 핵심적이거나 중요한 질문 1개만** 생성해야 하며, 이는 **사람이 이 문서를 봤을 때 가장 먼저 궁금해할 질문**이어야 해.
**문서:**
제목: {clean_text(title)}
내용: {clean_text(content)}

### 출력 형식
[
  {{"query": "질문1", "answer": "문서 기반 답변1"}}
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
"""
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # print("=== Gemini 응답 ===")
        # print(response_text)
        # print("===================")

        # JSON 마크다운 블록 제거
        cleaned = re.sub(r"```json|```", "", response_text).strip()

        # JSON 파싱
        qa_list = json.loads(cleaned)
        print(qa_list)
        return qa_list

    except Exception as e:
        print("QA 생성 실패:", e)
        return []

# 데이터 불러오기
file_path = "/home/food/people/minju/data/make_qa/real_final.csv"
df = pd.read_csv(file_path)

# 빈 컬럼 추가
df["gemini_query"] = ""
df["gemini_answer"] = ""

# QA 생성 루프
for i in range(len(df)):
    print(f"\n\\ [{i+1}/{len(df)}] QA 생성 중...")

    title = df.loc[i, "title"] if "title" in df.columns else ""
    if "document" in df.columns:
        text = df.loc[i, "document"]
    elif "extracted_text" in df.columns:
        text = df.loc[i, "extracted_text"]
    else:
        print(" 문서 내용이 없습니다.")
        continue

    if pd.isna(text) or len(str(text).strip()) < 5:
        print("문서가 비어 있음. 스킵합니다.")
        continue

    qa_pairs = get_qa_from_document(title, text)

    if qa_pairs:
        queries = [pair["query"] for pair in qa_pairs]
        answers = [pair["answer"] for pair in qa_pairs]
        df.at[i, "gemini_query"] = "\n".join(queries)
        df.at[i, "gemini_answer"] = "\n".join(answers)

    time.sleep(15)

# 결과 저장
df.to_csv("real_final_with_gemini_QA.csv", index=False)
print("\n저장 완료: real_final_with_gemini_QA.csv")
