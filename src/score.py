import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_category_center(embeddings: np.ndarray , category_list: list[str]) -> dict[str, np.ndarray]:

  category_centers = {}

  unique_categories = np.unique(category_list)
  for category in unique_categories:
    category_indices = [i for i, c in enumerate(category_list) if category == c]
    category_embeddings = embeddings[category_indices]
    category_centers[category] = np.mean(category_embeddings, axis=0)

  return category_centers


def calculate_scores(embeddings: np.ndarray , category_list: list[str]) -> np.ndarray:
  
  category_centers = calculate_category_center(embeddings, category_list)

  scores = []

  for emb, category in zip(embeddings, category_list):
    category_center = category_centers[category]
    similarity = cosine_similarity([emb], [category_center])[0][0]

    score = similarity * 10
    scores.append(score)

  return np.array(scores)




  