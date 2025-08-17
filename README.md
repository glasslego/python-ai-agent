# python-ai-agent
simple python ai agent

# Installation

```bash
# 가상환경 생성
uv venv --python 3.11

# 가상환경 activate
source .venv/bin/activate

uv init
```
### 환경변수 설정
```bash
#.env 파일에 아래 환경변수 추가
OPENAI_API_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXX
CLAUDE_API_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXX
GEMINI_API_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

## 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   News API      │    │  Web Scraper    │    │  Data Storage   │
│   (뉴스 수집)     │───▶│  (콘텐츠 추출)     │───▶│  (Pandas DF)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                     │
┌─────────────────┐    ┌─────────────────┐           │
│  LLM Interface  │    │ Analysis Engine │◀──────────┘
│  (사용자 쿼리)     │◀───│ (NLP 분석 수행)   │
└─────────────────┘    └─────────────────┘
```

## 데이터 파이프라인
### 데이터 수집
- 뉴스 API를 통해 최신 뉴스 데이터를 수집
- 수집된 뉴스는 Pandas DataFrame 형태로 저장

### 데이터 전처리
- 수집된 뉴스 데이터를 정제하고 필요한 정보 추출
- 텍스트 정제, 불용어 제거 등 NLP 전처리 수행

### 데이터 분석
- TF-IDF 등을 이용한 텍스트 분석
- K-Means 클러스터링을 통한 뉴스 분류
- LDA를 이용한 토픽 모델링
- 감성 분석을 통한 뉴스의 긍정/부정 평가
- 문서 유사도 분석을 통한 유사 뉴스 검색
- 뉴스 요약을 통한 핵심 내용 추출

### 사용자 쿼리에 대한 답변 생성
- 사용자 쿼리 → LLM 의도 파악 → 적절한 분석 함수 호출 → 결과 반환

#### 사용자 쿼리 예시
```
"최근 1주일 IT 뉴스 인기 키워드 10개" → TF-IDF 키워드 분석
"삼성전자 관련 뉴스 감정 분석" → 감정 분석 + 통계
"AI 뉴스 5개 그룹으로 클러스터링" → K-means 클러스터링
"오늘 경제 뉴스 요약" → 자동 뉴스 요약
```
