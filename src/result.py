import numpy as np


def print_result(df: np.ndarray):

  # 분야 별로 가장 점수가 높은 3개의 자소서를 추출 
  top3_by_category = (
    df.sort_values(by='score', ascending=False)
      .groupby('Category')
      .head(3)
  )

  for category in top3_by_category['Category'].unique():
      print(f"\n--- {category} ---")
      sub_df = top3_by_category[top3_by_category['Category'] == category]
      for idx, row in sub_df.iterrows():
          print(f"Resume Number: {idx + 2}, Score: {row['score']:.4f}")