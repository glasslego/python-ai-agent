# https://newsapi.org/sources
# https://newsapi.org/docs/client-libraries/python
# flake8: noqa: E501

import time

import pandas as pd
import requests

from src.news_analysis.parse_url import parse_article_content

NEWS_API_KEY = "9268c1b56a644f2fa03cc5979fbfcb75"


def get_latest_it_news_json(num_articles=10):
    """
    NewsAPI를 사용하여 최신 IT 뉴스 기사를 JSON 형태로 가져옵니다.

    Args:
        num_articles (int): 가져올 기사 수 (기본값: 10)

    Returns:
        dict: 기사 제목과 URL이 담긴 JSON 객체.
              오류 발생 시 빈 딕셔너리를 반환합니다.
    """
    api_key = NEWS_API_KEY
    if not api_key:
        print("❌ 에러: NewsAPI 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return []

    url = "https://newsapi.org/v2/top-headlines"

    # API 요청에 필요한 파라미터 설정
    params = {
        "apiKey": api_key,
        "category": "technology",  # IT(기술) 카테고리
        "country": "us",  # 주요 IT 뉴스가 많은 미국으로 설정 (한국 'kr'도 가능)
    }

    try:
        response = requests.get(url, params=params)
        # HTTP 상태 코드가 에러를 나타내면 자동으로 예외 발생 시키는 Response 객체 메소드
        response.raise_for_status()

        # 응답 데이터를 JSON 형태로 파싱
        data = response.json()

        # for debugging
        print(data)

        # 기사 목록 추출
        articles = data.get("articles")

        if not articles:
            print("💡 새로운 IT 뉴스를 찾을 수 없습니다.")
            return []

        return articles

    except requests.exceptions.RequestException as e:
        print(f"❌ API 요청 중 에러가 발생했습니다: {e}")
        return []
    except Exception as e:
        print(f"❌ 알 수 없는 에러가 발생했습니다: {e}")
        return []


def get_latest_it_news_formatted(num_articles=10):
    """
    NewsAPI를 사용하여 최신 IT 뉴스 기사를 가져옵니다.

    Args:
        num_articles (int): 가져올 기사 수 (기본값: 10)

    Returns:
        list: 기사 제목과 URL이 담긴 문자열 리스트.
              오류 발생 시 빈 리스트를 반환합니다.
    """
    articles = get_latest_it_news_json(num_articles)

    formatted_news = []
    for i, article in enumerate(articles):
        title = article.get("title")
        url = article.get("url")
        formatted_news.append(f"{i + 1}. {title}\n   - 링크: {url}")

    return formatted_news


def get_latest_it_news_dataframe(num_articles=10) -> pd.DataFrame:
    articles = get_latest_it_news_json(num_articles)
    sources = []
    authors = []
    titles = []
    descriptions = []
    urls = []
    published_ats = []
    contents = []

    for article in articles:
        sources.append(article["source"]["name"])
        authors.append(article["author"])
        titles.append(article["title"])
        descriptions.append(article["description"])
        urls.append(article["url"])
        published_ats.append(article["publishedAt"])
        contents.append(article["content"])

    new_article_data = {
        "Source": sources,
        "Author": authors,
        "Title": titles,
        "Description": descriptions,
        "URL": urls,
        "Published At": published_ats,
        "Content": contents,
    }

    # pandas DataFrame으로 변환
    df = pd.DataFrame(new_article_data)

    print("\n DataFrame 변환 완료!")
    print(f"DataFrame 크기: {df.shape} (행: {df.shape[0]}, 열: {df.shape[1]})")

    print("\n DataFrame 정보:")
    print(df.info())

    print("\n DataFrame 미리보기 (처음 3개 행):")
    print(df.head(3))

    # url 파싱 결과를 DataFrame에 추가
    df_with_content = add_parsed_content_to_dataframe(df)
    print("\n DataFrame 미리보기 (처음 3개 행):")
    print(df_with_content.head(3))

    return df_with_content


def get_news_content(url: str):
    """
    주어진 URL에서 뉴스 기사의 내용을 가져옵니다.

    Args:
        url (str): 뉴스 기사 URL

    Returns:
        str: 뉴스 기사의 파싱된 내용과 메타 정보가 포함된 딕셔너리.
    """

    try:
        parsed_data = parse_article_content(url)
        print(parsed_data)

        res = {
            "Parsed Content": parsed_data.get("content", ""),
            "Parsed Author": parsed_data.get("author", ""),
            "Meta Description": parsed_data.get("meta_description", ""),
            "Parsed Publish Date": parsed_data.get("publish_date", ""),
            "Tags": ", ".join(parsed_data.get("tags", [])),
            "Word Count": parsed_data.get("word_count", 0),
            "Reading Time (min)": parsed_data.get("reading_time", 0),
            "Parse Status": parsed_data.get("status", "not_attempted"),
        }

        return res

    except Exception as e:
        print(f"❌ 뉴스 기사 내용을 가져오는 중 에러 발생: {e}")
        return {
            "Parsed Content": "",
            "Parsed Author": "",
            "Meta Description": "",
            "Parsed Publish Date": "",
            "Tags": "",
            "Word Count": 0,
            "Reading Time (min)": 0,
            "Parse Status": "error",
        }


def add_parsed_content_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """기존 DataFrame에 URL 파싱 결과 추가"""

    df_copy = df.copy()

    # 새 컬럼들 초기화
    df_copy["Parsed Content"] = ""
    df_copy["Parsed Author"] = ""
    df_copy["Meta Description"] = ""
    df_copy["Parsed Publish Date"] = ""
    df_copy["Tags"] = ""
    df_copy["Word Count"] = 0
    df_copy["Reading Time (min)"] = 0
    df_copy["Parse Status"] = "pending"

    # iterrows로 각 URL 파싱
    for index, row in df_copy.iterrows():
        print(f"🔍 [{index + 1}/{len(df_copy)}] {row['Title'][:40]}...")

        url = row["URL"]
        parsed_data = get_news_content(url)  # 기존 함수 사용

        # 파싱 결과를 DataFrame에 직접 할당
        df_copy.at[index, "Parsed Content"] = parsed_data["Parsed Content"]
        df_copy.at[index, "Parsed Author"] = parsed_data["Parsed Author"]
        df_copy.at[index, "Meta Description"] = parsed_data["Meta Description"]
        df_copy.at[index, "Parsed Publish Date"] = parsed_data["Parsed Publish Date"]
        df_copy.at[index, "Tags"] = parsed_data["Tags"]
        df_copy.at[index, "Word Count"] = parsed_data["Word Count"]
        df_copy.at[index, "Reading Time (min)"] = parsed_data["Reading Time (min)"]
        df_copy.at[index, "Parse Status"] = parsed_data["Parse Status"]

        # 서버 부하를 위해서 1초 대기 (optional)
        time.sleep(1)

    return df_copy


if __name__ == "__main__":
    print("🚀 최신 IT 뉴스를 가져옵니다...")
    latest_it_news = get_latest_it_news_json(100)
    print(latest_it_news)

    formatted_news = get_latest_it_news_formatted(100)
    for news in formatted_news:
        print(news)

    df = get_latest_it_news_dataframe(100)
    df.to_csv("it_news_articles.csv", index=False)

    # res = get_news_content("https://nintendoeverything.com/nintendo-announces-my-mario-series-of-products-including-switch-app/")

    # latest_news = get_latest_it_news(10)
    #
    # if latest_news:
    #     print("\n--- 📰 최신 IT 뉴스 Top 10---")
    #     for news_item in latest_news:
    #         print(news_item)
    #     print("--------------------------\n")
