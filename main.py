from src.preprocess import preprocess_data


def main():
  # 1. 데이터 전처리
  df = preprocess_data("data/UpdatedResumeDataSet.csv")
  print(df)


if __name__ == "__main__":
  main()




