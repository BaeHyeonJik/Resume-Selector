import os
import numpy as np
from src.preprocess import preprocess_data
from src.embedder import embed_texts
from src.score import calculate_scores
from src.result import print_result

def main():

  # 데이터 전처리
  df = preprocess_data("data/UpdatedResumeDataSet.csv")
  print('데이터 전처리 완료!')

  # 2. 임베딩 생성
  embeddings_path = "data/embeddings.npy"
  if os.path.exists(embeddings_path):
      embeddings = np.load(embeddings_path)
      print("임베딩 로드 완료!")
  else:
      embeddings = embed_texts(df['clean_resume'].tolist())
      np.save(embeddings_path, embeddings)
      print("임베딩 완료 및 저장!")

  # 3. 점수 매기기
  scores = calculate_scores(embeddings, df['Category'].tolist())
  print("점수 배정 완료!")

  # 4. 결과 출력
  df['score'] = scores
  print_result(df)

  



if __name__ == "__main__":
  main()



