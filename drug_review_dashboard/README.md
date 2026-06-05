# 약물 리뷰 부작용 위험 탐지 Streamlit 프로젝트

## 👨‍🏫 교수님 피드백 및 문제 해결 요약 (Professor's Feedback & Resolution)

| 구분 | 교수님 피드백 | 해결 내용 |
|---|---|---|
| **1. 타깃 라벨 및 노이즈 문제** | "정답 라벨이 직접 들어 있지 않으므로 라벨 노이즈가 매우 커질 수 있습니다... 규칙 기반 약한 라벨(weak labeling)로 시작하는 편이 훨씬 현실적입니다." | 교수님의 권고대로 평점(rating ≤ 3)과 부작용 텍스트 키워드를 조합한 '데이터 기반 약한 라벨링(Weak Labeling)' 전략으로 타깃을 전면 재정의하여 현실적이고 견고한 모델 기반을 마련함. |
| **2. 타깃 누수(Target Leakage) 결함 발견 및 수정** | (모델의 신뢰성과 정확도 우려 관련) | 구현 과정에서 기존 코드의 정확도가 99.6%로 비정상적으로 높았던 '타깃 누수(Target Leakage)' 결함을 자체 탐지함. 라벨 생성에 쓰인 변수들을 학습 데이터에서 완벽히 격리(`LABEL_DEFINING_FEATURES` 제거)하고 순수 텍스트(TF-IDF) 중심으로 모델을 전면 개조하여, 학술적으로 당당히 방어 가능한 정직한 성능(90~97%)으로 고도화 완료함. |
| **3. 프로젝트 범위 조정 (이미지 인식 기능)** | "수업 프로젝트에서는 핵심을 텍스트 기반 부작용 위험 신호 탐지에 두는 것이 가장 좋고, 이미지 입력은 과감히 빼는 편이 안전합니다." | 교수님의 의견을 적극 수용하여, 텍스트 기반 위험 탐지 파이프라인 및 AI 챗봇을 프로젝트의 메인 주인공으로 설정함. 우려하셨던 약통 이미지 OCR 기능은 메인 시스템을 방해하지 않도록 별도의 '보너스 멀티모달 탭'으로 완전히 격리(Decoupled)하고, 로컬 비전 모델 미구동 시에도 대시보드가 터지지 않도록 예외 처리(Fallback) 안전장치를 완벽히 구축함. |
| **4. 데이터셋 라이선스 명확화** | "UCI 원본 라이선스 설명은 일부 부정확할 가능성이 있습니다... 정확한 라이선스 문구를 정리하는 것이 좋습니다." | UCI Druglib.com 원본 데이터셋의 라이선스가 'CC BY 4.0' 규격을 따름을 명확히 확인하고, Kaggle 재배포본과의 대조 결과를 문서에 공식 명시하여 법적/학술적 리스크를 완전히 제거함. |

## 🏗️ 시스템 아키텍처 (R&D → Production)

> **모든 전처리, EDA, 특성 엔지니어링 실험 및 검증은 `notebooks/` 파이프라인(주피터 랩 환경)에서 선행 연구 및 자산화되었습니다.**
>
> **Streamlit 대시보드는 주피터 랩에서 완벽히 검증된 공통 핵심 데이터 모듈(`src/`)을 그대로 재사용(Reflect)하여 실시간 서비스 배포를 수행하는 구조로, R&D와 Production이 유기적으로 연결된 정석적 아키텍처를 충족합니다.**

- **공통 핵심 모듈 (Single Source of Truth)**: `src/data_loader.py`(`load_drug_reviews`) · `src/features.py`(`add_features`) · `src/data_cache.py`(`prepare_data`)
- **R&D / Lab**: `notebooks/EDA_and_Preprocessing_Analysis.ipynb` 에서 전처리·특성 로직을 정의·검증·증명
- **Production**: Streamlit 앱(`pages/`)이 동일한 `src/` 모듈을 import 하여 서비스. `get_prepared_data()`는 `prepare_data()`의 캐시(@st.cache_data) 래퍼일 뿐 핵심 로직은 동일
- **동일성 보장**: 노트북 로직(`prepare_data`)과 앱 로직(`get_prepared_data`)의 데이터 형상·컬럼·특성 출력이 **완전히 동일**함을 확인(동일 shape·dtype·특성 해시)

```
src/ (공통 핵심 모듈: load_drug_reviews → add_features → prepare_data)
        │                              │
   notebooks/ (R&D·검증·증명)     pages/ (Production·실시간 서비스, get_prepared_data 캐시)
```

---

UCI ML Drug Review dataset을 이용해 환자 리뷰에서 심각한 부작용(ADE) 위험군을 탐지하는 수업용 프로젝트입니다.

## 실행 방법

### 1) 가상환경(venv) 생성 및 활성화

가상환경을 사용하면 시스템 파이썬과 의존성이 섞이지 않습니다. 프로젝트 폴더에서 실행하세요.

**Windows (PowerShell)**
```powershell
cd drug_review_dashboard
python -m venv venv
# 실행 정책 때문에 활성화가 막히면(최초 1회): Set-ExecutionPolicy -Scope Process RemoteSigned
.\venv\Scripts\Activate.ps1
```

**Windows (cmd)**
```bat
cd drug_review_dashboard
python -m venv venv
venv\Scripts\activate.bat
```

**macOS / Linux (bash/zsh)**
```bash
cd drug_review_dashboard
python3 -m venv venv
source venv/bin/activate
```

활성화되면 프롬프트 앞에 `(venv)`가 표시됩니다. 종료는 `deactivate`.

### 2) 의존성 설치 및 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

> 참고: 이 저장소에는 이미 만들어진 `venv/`가 상위 폴더에 포함될 수 있습니다. 새 환경을 만들 때는 위 절차를 따르세요. `venv/`는 보통 버전 관리에서 제외합니다.

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
- 시각화 페이지: 약물별 위험군 비율, 평점 대비 위험군 산점도, IQR 평점 이상치, 키워드 비교, 월별 추이
- 모델 서비스 페이지: 약물명과 증상 텍스트 입력 후 RandomForest 위험도 예측 + 모델 비교표(RF/HGB/규칙 베이스라인)
- 이미지 입력: 약통 이미지 업로드 → Ollama 비전 모델 멀티모달 약물명 인식(실패 시 파일명 기반 후보로 대체)
- 리포트: 규칙 기반 AI 상담 리포트와 선택적 Ollama 호출
- 데이터 조회: 검색, 필터링, CSV 다운로드
- AI 상담 챗봇: 약물별 모니터링 요약 + 부작용/주의사항 질의응답(리뷰 데이터 근거, Ollama 또는 규칙 기반 답변)
- 별도 EDA 스크립트(`eda_report.py`): 전체 데이터 통계 요약 + 그림 저장(`eda_outputs/`)

## LLM 백엔드 (OpenAI / Ollama / 오프라인)

모델 서비스 페이지와 AI 상담 챗봇은 사이드바에서 답변 엔진을 고를 수 있습니다.

- **OpenAI API**: `OPENAI_API_KEY`가 있으면 `gpt-4o-mini`(기본) 등으로 호출. 키가 없거나 호출 실패 시 자동으로 Ollama → 규칙 기반 순으로 대체합니다.
- **로컬 Ollama**: 로컬 `gemma3` 등으로 호출. 실패 시 규칙 기반으로 대체합니다.
- **오프라인(규칙 기반)**: LLM 없이 리뷰 데이터 통계 기반 답변. 항상 동작(시연 안전).

우선순위/폴백 로직은 `src/llm_helper.py`의 `route_chat()`에 구현되어 있습니다.

API 키 설정:

```powershell
# .env.example 을 .env 로 복사하고 실제 키를 입력 (.env 는 git에 커밋되지 않음)
Copy-Item .env.example .env
# .env 안의 OPENAI_API_KEY=your_api_key_here 를 실제 키로 교체
```

키는 streamlit secrets(`.streamlit/secrets.toml`의 `OPENAI_API_KEY`) → `.env`/환경변수 순으로 탐색합니다.

## Jupyter Notebook 실행 방법

EDA·전처리·특성 엔지니어링 전 과정을 정리한 노트북이 `notebooks/EDA_and_Preprocessing_Analysis.ipynb` 에 있습니다. JupyterLab에서 셀을 위에서부터 실행하면 215k행 적재 → 정제 → 6개 그림 → 누수 before/after까지 재현됩니다.

```bash
# (가상환경 활성화 후) 의존성 설치 — jupyterlab 포함
pip install -r requirements.txt

# 프로젝트 폴더(drug_review_dashboard)에서 JupyterLab 실행
jupyter lab
```

JupyterLab이 브라우저에서 열리면 왼쪽 파일 탐색기에서 `notebooks/EDA_and_Preprocessing_Analysis.ipynb` 를 더블클릭해 엽니다. 상단 메뉴 **Run → Run All Cells** 로 전체를 실행할 수 있습니다(전체 데이터 적재·모델 비교 포함 약 1~2분 소요).

> 특정 노트북만 바로 열려면: `jupyter lab notebooks/EDA_and_Preprocessing_Analysis.ipynb`
> 커널은 이 프로젝트의 가상환경(venv) Python을 사용해야 `src/` 모듈이 import 됩니다.

## 테스트

LLM 라우팅/폴백 로직 단위 테스트:

```powershell
pip install pytest
python -m pytest tests/ -q
```

## 데이터 누수(leakage) 처리

약한 라벨은 심각/증상 키워드 수와 낮은 평점으로 정의되므로, 이 컬럼들을 그대로 학습에 넣으면 모델이 라벨 규칙을 복원해 비현실적 성능(Acc~99%)이 나온다. 따라서 라벨 정의 컬럼(`severe_keyword_count`, `symptom_keyword_count`, `low_rating_flag`)을 학습 특성에서 제외(`MODEL_FEATURES`)하고 리뷰 원문 TF-IDF + 비누수 특성만으로 학습해 **정직한 성능(F1≈0.90)** 을 보고한다. 자세한 내용은 `보고서.md` 4·6장 참고.

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
