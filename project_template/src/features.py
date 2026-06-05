# src/features.py — 정제 & 특성 만들기 (2차 작업~과제에서 채움)
import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """결측치·이상치·타입을 정리한 DataFrame 반환.

    TODO 예시 (필요한 것만 골라 쓰기):
      - 결측 제거:        df = df.dropna(subset=["중요컬럼"])
      - 결측 채우기:      df["A"] = df["A"].fillna(df["A"].median())
      - 타입 변환:        df["날짜"] = pd.to_datetime(df["날짜"])
      - 중복 제거:        df = df.drop_duplicates()
    """
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """모델에 넣을 새 특성(파생 변수)을 추가한 DataFrame 반환.

    "이 데이터에서 새로운 특성을 찾아낸다"가 프로젝트의 핵심입니다.
    TODO 예시:
      - df["길이"] = df["텍스트"].str.len()
      - df["요일"] = df["날짜"].dt.dayofweek
      - 범주형 인코딩:  pd.get_dummies(df, columns=["범주컬럼"])
    """
    return df
