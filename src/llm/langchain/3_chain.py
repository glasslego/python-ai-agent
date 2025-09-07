from dotenv import load_dotenv
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

# 환경 변수 로드
load_dotenv()


# ============================================================
# 3. 체인 (Chain)
# ============================================================


def chain_simple_sequential():
    """
    순차적 체인 예제 - 여러 작업을 순서대로 연결
    """

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
    )

    # 체인 1: 주제를 받아 시놉시스 생성
    synopsis_template = """
    다음 주제로 영화 시놉시스를 2-3문장으로 작성하세요:
    주제: {topic}
    """
    synopsis_prompt = PromptTemplate(
        input_variables=["topic"], template=synopsis_template
    )
    # LLMChain 대신 LCEL 사용
    synopsis_chain = synopsis_prompt | llm | StrOutputParser()

    # 체인 2: 시놉시스를 받아 제목 생성
    title_template = """
    다음 시놉시스에 어울리는 영화 제목을 3개 제안하세요:
    시놉시스: {synopsis}
    """
    title_prompt = PromptTemplate(input_variables=["synopsis"], template=title_template)
    # LLMChain 대신 LCEL 사용
    title_chain = title_prompt | llm | StrOutputParser()

    # 순차적 실행 (SimpleSequentialChain은 유지 - 아직 deprecated 아님)
    def run_sequential(topic):
        print(f"입력 주제: {topic}")

        # 1단계: 시놉시스 생성
        synopsis = synopsis_chain.invoke({"topic": topic})
        print(f"생성된 시놉시스:\n{synopsis}\n")

        # 2단계: 제목 생성
        titles = title_chain.invoke({"synopsis": synopsis})
        print(f"제안된 제목들:\n{titles}")

        return titles

    # 실행 (run() 대신 invoke() 사용)
    result = run_sequential("인공지능과 인간의 우정")
    print(f"최종 결과:\n{result}")

    return synopsis_chain, title_chain


def chain_with_output_parser():
    """
    출력 파서를 포함한 체인 예제
    """

    # 출력 파서 생성
    output_parser = CommaSeparatedListOutputParser()

    # 프롬프트 템플릿
    template = """
    {category} 카테고리의 상위 5개 항목을 나열하세요.
    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["category"],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
    )

    # LLM과 체인 생성
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
    )

    # LLMChain 대신 LCEL 사용
    chain = prompt | llm | output_parser

    # 실행 및 파싱
    categories = ["프로그래밍 언어", "데이터베이스", "웹 프레임워크"]

    for category in categories:
        # run() 대신 invoke() 사용
        parsed = chain.invoke({"category": category})
        print(f"\n{category}:")
        for i, item in enumerate(parsed, 1):
            print(f"  {i}. {item}")

    return chain


def chain_parallel_processing():
    """
    병렬 처리 체인 예제 - 여러 작업을 동시에 수행
    """

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
    )

    # 여러 분석 체인 생성
    chains = {}

    # 감정 분석 체인
    sentiment_prompt = PromptTemplate(
        template="다음 텍스트의 감정을 분석하세요 (긍정/부정/중립): {text}",
        input_variables=["text"],
    )
    # LLMChain 대신 LCEL 사용
    chains["sentiment"] = sentiment_prompt | llm | StrOutputParser()

    # 요약 체인
    summary_prompt = PromptTemplate(
        template="다음 텍스트를 한 문장으로 요약하세요: {text}",
        input_variables=["text"],
    )
    # LLMChain 대신 LCEL 사용
    chains["summary"] = summary_prompt | llm | StrOutputParser()

    # 키워드 추출 체인
    keyword_prompt = PromptTemplate(
        template="다음 텍스트에서 주요 키워드 3개를 추출하세요: {text}",
        input_variables=["text"],
    )
    # LLMChain 대신 LCEL 사용
    chains["keywords"] = keyword_prompt | llm | StrOutputParser()

    # 테스트 텍스트
    text = """
    인공지능 기술의 발전은 우리 삶을 크게 변화시키고 있습니다.
    의료, 교육, 금융 등 다양한 분야에서 AI가 활용되며 효율성이 크게 향상되었습니다.
    하지만 일자리 감소와 개인정보 보호 등의 문제도 함께 고려해야 합니다.
    """

    # 병렬로 실행
    results = {}
    for name, chain in chains.items():
        # run() 대신 invoke() 사용
        results[name] = chain.invoke({"text": text})

    # 결과 출력
    print("병렬 분석 결과:")
    for name, result in results.items():
        print(f"\n[{name}]: {result}")

    return chains


if __name__ == "__main__":
    chain_simple_sequential()
    chain_with_output_parser()
    chain_parallel_processing()
