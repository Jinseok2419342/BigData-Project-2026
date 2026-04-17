import ollama
import json

reviews = [
    "아메리카노가 진하고 맛있어요. 매일 아침 여기서 마시고 싶을 정도예요!",
    "라떼가 너무 달고 시럽 맛이 강해요. 다음엔 덜 달게 해달라고 해야겠어요.",
    "가격 대비 양이 적고 맛도 평범해요. 특별히 추천하고 싶진 않네요."
]

results = []

for review in reviews:
    response = ollama.chat(
        model = "gemma3:12b",
        messages=[
            {
                "role": "system",
                "content": """당신은 카페 음료 리뷰 분석 전문가입니다.
주어진 리뷰를 분석하여 반드시 아래 JSON 형식으로만 응답하세요.
다른 설명이나 텍스트는 절대 포함하지 마세요.

{"sentiment": "긍정/부정/중립", 
 "confidence": 0.0~1.0 사이의 숫자, 
 "keywords": ["키워드1", "키워드2", "키워드3"],
 "reason": "한 줄로 분석 이유 간단히 설명"}
"""
            },
            {
                "role": "user",
                "content": review
            }
        ]
    )

    raw = response["message"]["content"]

    # JSON 파싱 시도
    try:
        # LLM이 ```json ... ``` 으로 감싸서 응답할 수 있음
        clean = raw.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()

        data = json.loads(clean)
        results.append(data)
        print(f"리뷰: {review[:25]}...")
        print(f"   감성: {data['sentiment']}, 확신도: {data['confidence']}")
        print(f"   키워드: {data['keywords']}")
    except json.JSONDecodeError:
        print(f"JSON 파싱 실패: {raw[:100]}")

    print("-" * 50)

print(f"\n총 {len(results)}개 리뷰 분석 완료")
