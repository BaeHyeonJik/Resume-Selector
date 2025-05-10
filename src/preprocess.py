import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK 리소스 다운로드 (한 번만 실행)
def download_nltk_resources():
    try:
        stopwords.words('english')
    except:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')


# 텍스트 전처리 함수
def preprocess_text(text):
    # 소문자화 및 불필요한 문자 제거
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # 불용어 제거
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]

    # 단어 원형화 (Lemmatization)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)


# 데이터 불러오기
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    download_nltk_resources()
    df['clean_resume'] = df['Resume'].apply(preprocess_text)
    return df[['Category', 'clean_resume']]
