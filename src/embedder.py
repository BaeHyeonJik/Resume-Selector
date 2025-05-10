from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from tqdm import tqdm

# 사전 학습된 BERT 모델과 토크나이저 로드
# 'bert-base-uncased'는 소문자로만 구성된 BERT 사전학습 모델을 의미
# 문장을 토큰으로 바꾸는 토크나이저 로드
# 사전 학습된 BERT 모델 로드
# 모델을 추론 모드로 설정 (이미 학습된 모델을 사용하기 때문에)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

def embed_text(text: str) -> np.ndarray:

  # 입력 텍스트를 BERT 입력 형식으로 변환
  # return_tensors='pt': 파이토치 텐서로 반환
  # truncation=True: 길면 자름 (최대 512 토큰)
  # padding='max_length': 부족한 길이는 512까지 0으로 채움
  input = tokenizer(text, return_tensors='pt', truncation= True, max_length=512, padding='max_length')

  # 추론 시에는 torch.no_grad()로 연산 그래프 생성 방지 (메모리 절약, 속도 향상)
  with torch.no_grad():
    outputs = model(**input)
    
    # 마지막 히든 스테이트에서 [CLS] 토큰 벡터 추출 (문장의 대표 의미)
    # BERT 출력은 (batch_size, sequence_length, hidden_size) 형태
    # 첫 번째 토큰([CLS])만 선택
    cls_embedding = outputs.last_hidden_state[:, 0, :]

  return cls_embedding.squeeze().numpy()




def embed_texts(texts: list[str]) -> np.ndarray:

  embeddings = []
  for text in tqdm(texts, desc="Embedding"):
    vec = embed_text(text)
    embeddings.append(vec)
  
  return np.array(embeddings)