# 프로젝트 템플릿 (복사해서 시작하세요)

5주차 영화 대시보드와 **같은 멀티페이지 구조**입니다. 새 기술 없음.

## 시작하기
```bash
# 1) 이 폴더를 복사해 내 프로젝트 폴더로 (예: my_project)
# 2) 의존성 설치
pip install -r requirements.txt
# 3) 실행
streamlit run app.py
```

## 폴더 구조
```
app.py                # 진입점 (st.navigation) — 제목/이름만 바꾸면 됨
pages/
  1_EDA.py            # 1차 작업: 요약·결측·분포
  2_시각화.py          # 2차 작업: plotly 그래프 + 해석
  3_모델_서비스.py     # 3차 작업: 입력→예측→결과 (경로 A~D 중 택1)
src/
  data_loader.py      # 내 데이터 연결 (여기 한 줄만 바꾸면 됨)
  features.py         # 정제·특성 (과제로 채움)
data/                 # 데이터 파일 (sample.csv + 이미지 샘플 data/images/)
이미지_데이터_쓸_때/   # 이미지 프로젝트용 변형본 (적재·EDA 2파일 교체)
보고서_템플릿.md
requirements.txt
```

> **표(CSV) 데이터**면 아래 순서대로. **이미지 데이터**면 `이미지_데이터_쓸_때/README.md` 를 먼저 보세요.

## 작업 순서 (권장)
1. `src/data_loader.py`의 `load_data()`를 **내 데이터에 연결** → 앱 실행되는지 확인
2. `pages/1_EDA.py` 확인 (대부분 그대로 동작)
3. `pages/2_시각화.py`에서 컬럼/해석 채우기
4. `pages/3_모델_서비스.py`에서 **경로 A~D 중 하나만** 남기고 완성
5. `src/features.py`로 특성 다듬기 → 모델 성능 개선
6. `보고서_템플릿.md` 작성

> 막히면 가이드의 FAQ와 7·10·11·12주 실습 코드를 참고하세요.
