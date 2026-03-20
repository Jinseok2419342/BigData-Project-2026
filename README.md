# BigData-Project-2026
- 빅데이터 분석 프로젝트
- [20241499 장진석]

---

### venv 가상환경 활성화
```powershell
venv\Scripts\Activate.ps1
```
오류 나는 경우 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

- 가상환경 내 파일 실행 예시.
`python env_test.py`

- 폴더 밑이면
`python`까지 치고, 파일 경로 복사한거 붙여넣으면 됨 ""붙이면 더 좋고

- Streamlit은 `streamlit run my_profile.py` 이런식


### GitHub 자격 증명 삭제
```
cmdkey /delete:git:https://github.com
```

### GitHub 몇몇 복붙용
```git
git add . 
git commit -m "변경 내용 설명" 
git push 
```