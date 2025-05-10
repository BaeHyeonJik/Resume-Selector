# Resume Selector

## 📌 프로젝트 목적
이 프로젝트는 다양한 이력서 데이터를 전처리하고, 임베딩 및 군집화/스코어링 등의 기법을 통해 특정 직무에 적합한 이력서를 자동으로 선별하는 데 목적이 있습니다.  
채용 담당자의 수작업 부담을 줄이고, 효율적인 이력서 필터링을 지원합니다.

## 📂 사용 데이터셋
- [Updated Resume Dataset on Kaggle](https://www.kaggle.com/datasets/arihantjain/updated-resume-dataset)


## 🔧 실행 방법
```bash
# 가상환경 활성화
python -m venv venv
venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 실행
python main.py