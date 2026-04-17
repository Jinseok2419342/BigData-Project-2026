import ollama

reviews = [
    "치킨이 바삭하고 정말 맛있었어요! 양도 푸짐해요.",
    "피자가 너무 기름지고 맛이 없어요. 다시는 시키지 않을 것 같아요.",
    "보통이었어요. 가격 대비 특별한 점은 없네요.",
    "떡볶이가 매콤하고 맛있는데, 배달이 너무 늦어서 아쉬웠어요."
]

for review in reviews:
    response = ollama.chat(
        model = "gemma3:12b",
        messages = [
            {
                "role" : "system",
                "content" : """당신은 음식 배달 리뷰 분석 전문가입니다.
주어진 리뷰를 분석해서 아래 항목만 명확하게 답변하세요.

- 감정: 긍정 / 부정 / 중립
- 주요 이유: 한 문장으로 간단히 설명
- 추천 여부: 추천 / 비추천 / 보통

답변은 자연스럽게 한국어로 작성해주세요."""
            },
            {
                "role" : "user",
                "content" : review
            }
        ]
    )

    print(f"리뷰: {review[:30]}...")
    print(f"분석: {response['message']['content']}")
    print("-" * 60)