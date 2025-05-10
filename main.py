from src.preprocess import preprocess_data
from src.embedder import embed_texts

def main():
  # 1. 데이터 전처리
  df = preprocess_data("data/UpdatedResumeDataSet.csv")

  # 2. 임베딩 생성
  embeddings = embed_texts(df)
  print(embeddings)

  # 3. 클러스터링
  
  # 4. 점수 매기기

  # 결과 출력

if __name__ == "__main__":
  main()




