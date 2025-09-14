# 🤖 스마트 개인 비서 챗봇

사용자가 자연어로 질문하면 LLM이 상황을 판단해서 적절한 API를 호출하고 답변하는 멀티 기능 챗봇

## 🚀 핵심 기능

### 📰 뉴스 검색
- 카테고리별 최신 뉴스 검색
- 키워드 기반 뉴스 검색
- 뉴스 감성 분석

### 🌤️ 날씨 정보
- 도시별 현재 날씨 조회
- 날씨 예보 (5일)
- 온도 기반 옷차림 추천

### 💱 환율 정보
- 실시간 환율 조회
- 다양한 통화 간 환율 계산
- 통화 변환 계산기

### 📈 주식 정보
- 한국/미국 주식 가격 조회
- 관심 종목 관리
- 종목별 현재가 모니터링

### 📅 일정 관리
- 일정 추가/수정/삭제
- 주간 일정 조회
- 할일 목록 관리

## 🛠️ 설치 및 실행

### 실행 방법

#### CLI 버전
```bash
python main.py
```

#### Streamlit 웹 UI 버전
```bash
streamlit run src/simple_assistant/app.py
```

브라우저에서 http://localhost:8501 접속

### 4. 데모 데이터 생성 (선택사항)
```bash
python demo_data.py
```

## 💬 사용 예시

### 단일 기능 질문
- "삼성전자 뉴스 찾아줘"
- "서울 날씨 어때?"
- "달러 환율 알려줘"
- "삼성전자 주가 확인해줘"
- "내일 회의 일정 추가해줘"

### 복합 질문
- "삼성전자 주가 확인하고, 관련 뉴스도 찾아줘. 그리고 달러 환율도 알려줘"
- "내일 서울 날씨 확인하고 비 온다면 우산 챙기라고 일정 추가해줘"

## 🎯 주요 특징

### 🔧 개별 에이전트 시스템
- **NewsAgent**: 뉴스 검색 및 감성 분석
- **WeatherAgent**: 날씨 정보 및 옷차림 추천
- **FinanceAgent**: 환율 및 주식 정보
- **ScheduleAgent**: 일정 및 할일 관리

### 🌐 Streamlit Web UI
- 직관적인 웹 인터페이스
- 탭별 기능 구분
- 실시간 데이터 업데이트
- 통합 대시보드

### 🤖 스마트 컨트롤러
- 자연어 의도 분석
- 복합 쿼리 처리
- 엔티티 추출
- 컨텍스트 기반 응답

## 🔧 API 키 설정

### OpenAI API (선택사항 - LangChain 사용시에만 필요)
1. [OpenAI Platform](https://platform.openai.com/)에서 API 키 생성
2. `.env` 파일에 `OPENAI_API_KEY` 설정

### 뉴스 API (선택)
1. [NewsAPI](https://newsapi.org/)에서 API 키 생성
2. `.env` 파일에 `NEWS_API_KEY` 설정

### 날씨 API (선택)
1. [OpenWeatherMap](https://openweathermap.org/api)에서 API 키 생성
2. `.env` 파일에 `WEATHER_API_KEY` 설정

*모든 API 키는 선택사항이며, 없으면 더미 데이터로 동작합니다.*

## 📁 프로젝트 구조
```
src/simple_assistant/
├── main.py              # CLI 실행 파일
├── app.py               # Streamlit 웹 UI
├── run_streamlit.py     # Streamlit 실행 스크립트
├── controller.py        # 통합 컨트롤러
├── demo_data.py         # 데모 데이터 생성
├── README.md           # 사용법 가이드
├── agents/             # 개별 에이전트들
│   ├── news_agent.py   # 뉴스 에이전트
│   ├── weather_agent.py # 날씨 에이전트
│   ├── finance_agent.py # 금융 에이전트
│   └── schedule_agent.py # 일정 에이전트
├── utils/
│   ├── api_client.py   # API 클라이언트들
│   └── tools.py        # LangChain Tools
└── workflows/
    └── main_workflow.py # LangChain 워크플로우
```

## 🎮 사용 가이드

1. **데모 데이터 생성**: `python demo_data.py`로 샘플 일정/할일 생성
2. **CLI 버전**: 터미널에서 대화형 챗봇 사용
3. **웹 UI 버전**: 브라우저에서 직관적인 인터페이스 사용
4. **통합 대시보드**: 모든 정보를 한눈에 확인
5. **개별 기능**: 각 탭에서 세부 기능 활용
