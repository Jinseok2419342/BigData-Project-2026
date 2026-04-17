import ollama

article = """
최근 한국에서 홈카페 문화가 빠르게 확산되고 있다. 많은 사람들이 집에서 직접 커피를 내리고, 
예쁜 라떼 아트를 만들며 여유로운 시간을 즐기고 있다. 특히 2030 젊은 층 사이에서는 
커피 머신과 다양한 원두를 수집하는 것이 새로운 취미로 자리 잡았다. 
한 설문조사에 따르면, 지난 1년 동안 홈카페 관련 용품 판매가 42% 증가했다고 한다. 
이러한 트렌드는 외식 비용을 절감하면서도 자신만의 여유로운 공간을 만드는 데 도움이 되고 있다. 
다만 좋은 원두를 고르는 안목과 머신 관리에 대한 지식이 필요해, 초보자들이 어려움을 느끼는 경우도 많다.
"""

# 키워드 추출
print("=== 키워드 추출 ===")
response = ollama.chat(
    model = "gemma3:12b",
    messages = [
        {
            "role": "system",
            "content": "주어진 텍스트에서 가장 중요한 핵심 키워드 5개를 추출하세요. "
                       "의미가 중복되지 않게 선택하고, 한국어로만 작성하세요. "
                       "키워드만 쉼표(,)로 구분하여 한 줄로 출력하세요."
        },
        {
            "role": "user",
            "content": article
        }
    ]
)
print(response["message"]["content"])

# 요약
print("\n=== 3줄 요약 ===")
response = ollama.chat(
    model="gemma3:12b",
    messages=[
        {
            "role": "system",
            "content": "주어진 기사를 정확히 3줄로 요약해주세요. "
                       "각 줄은 한 문장으로 구성하고, 전체 내용을 균형 있게 담아주세요."
        },
        {
            "role": "user",
            "content": article
        }
    ]
)
print(response["message"]["content"])
