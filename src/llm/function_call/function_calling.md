##  OpenAI Function Call (순수 API)
```python
# 함수 스키마 직접 정의
{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "날씨 조회",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }
}

# 복잡한 메시지 관리
messages.append(response_message)
messages.append({"tool_call_id": tool_call.id, "role": "tool", ...})
```

## LangChain Tool Call (추상화)
```python
python# 간단한 데코레이터
@tool
def get_weather(city: str) -> str:
    """날씨를 조회합니다."""
    return "맑음, 22°C"

# 자동 관리
agent_executor.invoke({"input": "서울 날씨 어때?"})
```

## 실제 동작 예시
* "서울 날씨 어때?" 질문 시
#### OpenAI Function Call 흐름:
```
1. 사용자 메시지 → LLM
2. LLM → "get_weather 함수 호출 필요!"
3. 개발자 코드 → get_weather("서울") 실행
4. 결과 → LLM에게 다시 전달
5. LLM → "서울은 맑음, 22°C입니다" 응답
```

#### LangChain Tool Call 흐름:
```
1. 사용자 메시지 → AgentExecutor
2. AgentExecutor → 자동으로 적절한 도구 선택 & 실행
3. 결과 → 자동으로 최종 응답 생성
```
