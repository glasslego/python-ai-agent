"""
LangChain 유틸리티 함수 모음
LangChain을 사용한 기본적인 설정과 체인 생성 함수들
"""

import os

from dotenv import load_dotenv
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI

# 환경 변수 로드
load_dotenv()


def setup_environment():
    """환경 설정 및 API 키 확인"""
    if not os.getenv("OPENAI_API_KEY"):
        print("\n경고: OPENAI_API_KEY 설정이 필요합니다!")
        print("설정 방법:")
        print("1. 프로젝트 루트 디렉토리에 .env 파일 생성")
        print("2. .env 파일에 다음과 같이 OPENAI_API_KEY=your-api-key-here 추가")
        return False
    return True


def get_openai_llm(model="gpt-3.5-turbo", temperature=0.7):
    """OpenAI LLM 인스턴스 생성"""
    return ChatOpenAI(model=model, temperature=temperature)


def get_ollama_llm(model="llama2"):
    """Ollama LLM 인스턴스 생성"""
    return Ollama(model=model)


def create_prompt_template(template, input_variables):
    """프롬프트 템플릿 생성"""
    return PromptTemplate(input_variables=input_variables, template=template)


def create_llm_chain(llm, prompt):
    """LLM 체인 생성"""
    return LLMChain(llm=llm, prompt=prompt)


def create_conversation_chain(llm, memory=None, verbose=False):
    """대화 체인 생성"""
    if memory is None:
        memory = ConversationBufferMemory()
    return ConversationChain(llm=llm, memory=memory, verbose=verbose)


def create_text_splitter(chunk_size=100, chunk_overlap=20, separator="\n"):
    """텍스트 분할기 생성"""
    return CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator
    )
