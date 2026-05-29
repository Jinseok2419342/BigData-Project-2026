# 약물 리뷰 부작용 위험 탐지 Streamlit 프로젝트

UCI ML Drug Review dataset을 이용해 환자 리뷰에서 심각한 부작용(ADE) 위험군을 탐지하는 수업용 프로젝트입니다.

## 실행 방법

```powershell
cd drug_review_dashboard
pip install -r requirements.txt
streamlit run app.py
```

## 데이터 준비

앱은 아래 순서로 데이터를 찾습니다.

1. `data/` 폴더의 로컬 CSV
2. KaggleHub `jessicali9530/kuc-hackathon-winter-2018`
3. 데모 샘플 데이터

KaggleHub 다운로드 캐시는 프로젝트 내부 `.kagglehub_cache/`에 저장됩니다. 작업 폴더 밖에 데이터를 쓰지 않기 위한 설정이며, 이 폴더는 `.gitignore`에 포함되어 있습니다.

로컬 CSV를 쓰는 경우 아래 파일명 중 하나로 `data/` 폴더에 넣으면 자동 인식됩니다.

- `drugsComTrain_raw.csv`
- `drugsComTest_raw.csv`
- `drug_reviews.csv`
- `uci_drug_reviews.csv`
- `kuc_drug_reviews.csv`

원본 데이터의 주요 컬럼은 `drugName`, `condition`, `review`, `rating`, `date`, `usefulCount`입니다.

## 구현 범위

- EDA 페이지: 요약, 결측치, 평점/리뷰 길이/위험군 분포
- 시각화 페이지: 약물별 위험군 비율, 평점 대비 위험군 산점도, IQR 평점 이상치, 키워드 비교
- 모델 서비스 페이지: 약물명과 증상 텍스트 입력 후 RandomForest 위험도 예측
- 이미지 입력: 약통 이미지 업로드와 파일명 기반 약물 후보 매칭
- 리포트: 규칙 기반 AI 상담 리포트와 선택적 Ollama 호출
- 데이터 조회: 검색, 필터링, CSV 다운로드

## 주의

원본 데이터에는 공식적인 "심각한 ADE" 라벨이 없습니다. 이 프로젝트는 심각 증상 키워드와 낮은 평점을 결합한 약한 라벨을 target으로 만들어 수업용 분류 모델을 구성합니다.
예측 결과는 의학적 진단이 아니며, 실제 긴급 증상은 의료기관 상담이 우선입니다.

## 문제 해결

앱이 계속 데모 샘플을 보여주면 Streamlit 캐시가 남아 있을 수 있습니다.

```powershell
# 실행 중인 Streamlit을 끈 뒤 다시 실행
streamlit run app.py
```

그래도 동일하면 앱 화면 우측 상단 메뉴에서 **Clear cache**를 누른 뒤 재실행하세요.
