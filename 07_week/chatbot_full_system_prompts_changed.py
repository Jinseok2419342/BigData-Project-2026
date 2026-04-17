import streamlit as st
import ollama

# ── 페이지 설정 ──
st.set_page_config(page_title="AI 챗봇", page_icon="🤖", layout="wide")

# ── 사이드바: 설정 ──
with st.sidebar:
    st.header("챗봇 설정")

    # 시스템 프롬프트 선택
    system_prompts = {
        "GitHub 커밋 메시지 전문가" : "당신은 Commits 전문가입니다. 사용자가 변경 내용을 말하면, 적절한 커밋 메시지를 작성해주세요",
        "AWS 클라우드 아키텍트" : "당신은 AWS Solutions Architect입니다. 상황에 맞는 AWS 아키텍처를 설계하고 추천해주세요",
        "Git 전문가" : "당신은 Git과 브랜치 전략 전문가입니다. Git 관련 질문에 실무 팁을 알려주세요",
        "SwiftUI 전문가" : "당신은 SwiftUI 전문가입니다. 사용자가 UI 요구사항을 말하면 SwiftUI 코드로 구현 방법을 알려주고, 최적화 팁도 추가해주세요",
        "전기차 하드웨어 연구원" : "당신은 자동차 기업의 전기차 하드웨어 선행연구 연구원입니다. 사용자의 질문에 대해 정확한 기술 용어들로 설명하며 답변해야 합니다",
        "Swift 코드 리뷰어" : "당신은 8년차 iOS 시니어 개발자입니다. Swift 코드를 리뷰하고, 개선점과 Best Practice를 알려주세요.",
        "Next.js 전문가" : "당신은 Next.js 전문가입니다. App Router, Server Components, SSR/SSG, API Routes를 활용한 구현 방법을 알려주세요."
    }

    selected_role = st.selectbox("AI 역할 선택", list(system_prompts.keys()))

    # 직접 입력 옵션
    custom_prompt = st.text_area(
        "또는 직접 입력:",
        placeholder="AI에게 부여할 역할을 입력하세요...",
        height=100
    )

    # Temperature 조절 (Ollama의 options 파라미터로 전달됨)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

    st.divider()

    # 대화 초기화 버튼
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # 대화 통계
    if "messages" in st.session_state:
        user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
        st.caption(f"대화 수: {user_msgs}턴")

# ── 시스템 프롬프트 결정 ──
system_prompt = custom_prompt.strip() if custom_prompt.strip() else system_prompts[selected_role]

# ── 메인 화면 ──
st.title("AI 챗봇")
st.caption(f"현재 역할: {selected_role} | Temperature: {temperature}")

# ── 대화 기록 초기화 ──
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── 기존 대화 표시 ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── 사용자 입력 ──
user_input = st.chat_input("메시지를 입력하세요")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Ollama에 전달할 메시지 (시스템 프롬프트 + 대화 기록)
    ollama_messages = [
        {"role": "system", "content": system_prompt}
    ] + st.session_state.messages

    with st.chat_message("assistant"):
        stream = ollama.chat(
            model="gemma3:4b",
            messages=ollama_messages,
            stream=True,
            options={"temperature": temperature}  # options: Ollama 모델 동작 제어 (temperature, top_k, top_p 등을 딕셔너리로 전달)
        )

        def stream_generator():
            for chunk in stream:
                yield chunk["message"]["content"]

        full_response = st.write_stream(stream_generator())

    st.session_state.messages.append({"role": "assistant", "content": full_response})
