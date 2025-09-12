# from sentence_transformers import SentenceTransformer
# import pandas as pd
# import pickle
# import os
# from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# # LaBSE 모델 로드
# model_path = 'sentence-transformers/LaBSE'
# model = SentenceTransformer(model_path)
# model.max_seq_length = 512

# # 데이터 경로 및 저장 경로
# excel_path = "/home/food/people/subin/data/식품안전정보DB-url 추가(2014~2024).xls"
# save_dir = '/home/food/people/minju/embedding/labse2'
# os.makedirs(save_dir, exist_ok=True)

# # 연도 범위 지정
# years = range(2014, 2025)

# # 전체 임베딩을 모을 리스트 (선택적)
# all_embeddings = []

# for year in years:
#     print(f"\n===== {year}년 데이터 처리 중 =====")

#     try:
#         # 엑셀 읽기
#         df = pd.read_excel(excel_path, sheet_name=str(year), usecols=["제목", "내용"])
#         df = df[~df['내용'].isnull()].reset_index(drop=True)
#         df["제목_내용"] = df["제목"].astype(str) + " " + df["내용"].astype(str)
#         sentences = df['제목_내용'].to_list()

#         print(f"[INFO] {year}년 데이터 개수: {len(sentences)}")

#         if len(sentences) == 0:
#             print(f"[WARN] {year}년 데이터 없음, 스킵")
#             continue

#         # 임베딩 생성
#         embeddings = model.encode(
#             sentences,
#             max_length=512,
#             truncation=True,
#             convert_to_numpy=True,
#             normalize_embeddings=True,  # FAISS 검색 최적화
#             show_progress_bar=True
#         )

#         # 연도별로 개별 저장
#         year_path = os.path.join(save_dir, f"embeddings_{year}.pkl")
#         with open(year_path, "wb") as f:
#             pickle.dump(embeddings, f)

#         print(f"[INFO] {year}년 임베딩 저장 완료 → {year_path}")

#         # 통합 리스트에 추가
#         all_embeddings.append(embeddings)

#     except Exception as e:
#         print(f"[ERROR] {year}년 처리 실패: {e}")
#         continue

# # --- 모든 연도 통합 파일 생성 ---
# if all_embeddings:
#     merged_embeddings = np.vstack(all_embeddings)
#     merged_path = os.path.join(save_dir, "embeddings_all.pkl")
#     with open(merged_path, "wb") as f:
#         pickle.dump(merged_embeddings, f)
#     print(f"\n✅ 전체 임베딩 통합 저장 완료 → {merged_path}")
# else:
#     print("\n❌ 통합 임베딩 생성 실패: 유효한 연도 데이터 없음")
import os
import pickle
import numpy as np

# 연도별 임베딩 경로
save_dir = '/home/food/people/minju/embedding/labse2'
years = range(2014, 2025)

all_embeddings = []

# 연도별 .pkl 파일 로드
for year in years:
    path = os.path.join(save_dir, f"embeddings_{year}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            emb = pickle.load(f)
            all_embeddings.append(emb)
            print(f"[INFO] {year}년 임베딩 로드 완료, shape={emb.shape}")
    else:
        print(f"[WARN] {year}년 파일 없음 → 스킵")

# 전체 합치기
if all_embeddings:
    merged_embeddings = np.vstack(all_embeddings)
    merged_path = os.path.join(save_dir, "embeddings_all.pkl")
    with open(merged_path, "wb") as f:
        pickle.dump(merged_embeddings, f)
    print(f"\n✅ 전체 임베딩 통합 저장 완료 → {merged_path}")
    print(f"최종 shape = {merged_embeddings.shape}")
else:
    print("\n❌ 통합 실패: 불러온 임베딩 없음")
