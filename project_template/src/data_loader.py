# src/data_loader.py — 데이터 읽기 한 곳에 모으기
# 모든 페이지가 이 함수 하나를 import 해서 같은 데이터를 본다.
import pandas as pd
import streamlit as st


@st.cache_data  # ★ 데이터는 한 번만 읽고 캐시에 보관 (새로고침해도 다시 안 읽음)
def load_data():
    """프로젝트 데이터를 DataFrame으로 반환.

    TODO: 본인 데이터에 맞게 아래 한 줄만 바꾸면 됩니다.
      - CSV:   pd.read_csv("data/파일.csv")
      - 한글 깨지면:  encoding="utf-8"  또는  encoding="cp949"
      - 너무 크면 개발 중엔:  nrows=50000  로 일부만
      - Open API/크롤링이면 받은 결과를 DataFrame으로 만들어 반환
    """
    df = pd.read_csv("data/sample.csv")  # TODO: 내 파일 경로로 변경
    return df
