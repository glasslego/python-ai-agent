# https://newsapi.org/sources
# flake8: noqa: E501
import re

import requests
from bs4 import BeautifulSoup


def parse_article_content(url, timeout=10):
    """
    BeautifulSoup을 사용하여 뉴스 기사 URL에서 전체 내용을 파싱합니다.

    Args:
        url (str): 파싱할 기사 URL
        timeout (int): 요청 타임아웃 (초)

    Returns:
        dict: 파싱된 기사 내용 (제목, 본문, 메타데이터 등)
    """

    print(f"🔍 URL 파싱 중: {url}")

    # HTTP 헤더 설정 (봇 차단 방지)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        # BeautifulSoup으로 HTML 파싱
        soup = BeautifulSoup(response.content, "html.parser")

        # 파싱 결과 저장
        parsed_data = {
            "url": url,
            "status": "success",
            "title": extract_title(soup),
            "content": extract_content(soup),
            "meta_description": extract_meta_description(soup),
            "publish_date": extract_publish_date(soup),
            "author": extract_author(soup),
            "tags": extract_tags(soup),
            "word_count": 0,
            "reading_time": 0,
        }

        # 단어 수 및 읽기 시간 계산
        if parsed_data["content"]:
            word_count = len(parsed_data["content"].split())
            parsed_data["word_count"] = word_count
            parsed_data["reading_time"] = max(1, word_count // 200)  # 분당 200단어 기준

        print(
            f"파싱 성공: {parsed_data['word_count']}단어, 예상 읽기시간 {parsed_data['reading_time']}분"
        )

        return parsed_data

    except requests.exceptions.Timeout:
        print(f" 타임아웃: {url}")
        return {"url": url, "status": "timeout", "error": "Request timeout"}

    except requests.exceptions.RequestException as e:
        print(f"네트워크 오류: {e}")
        return {"url": url, "status": "network_error", "error": str(e)}

    except Exception as e:
        print(f"파싱 오류: {e}")
        return {"url": url, "status": "parse_error", "error": str(e)}


def extract_title(soup):
    """HTML에서 제목 추출"""

    # 다양한 제목 선택자 시도
    title_selectors = [
        "h1.entry-title",
        "h1.article-title",
        "h1.post-title",
        "h1.headline",
        "h1",
        "title",
        '[property="og:title"]',
        '[name="twitter:title"]',
    ]

    for selector in title_selectors:
        element = soup.select_one(selector)
        if element:
            title = (
                element.get_text(strip=True)
                if element.name != "meta"
                else element.get("content", "")
            )
            if title and len(title) > 5:  # 너무 짧은 제목 제외
                return clean_text(title)

    return "제목을 찾을 수 없음"


def extract_content(soup):
    """HTML에서 본문 내용 추출"""

    # 불필요한 태그들 제거
    for tag in soup(
        [
            "script",
            "style",
            "nav",
            "header",
            "footer",
            "aside",
            "advertisement",
            "ads",
            "social-share",
            "comments",
        ]
    ):
        tag.decompose()

    # 본문 추출을 위한 선택자들 (사이트별로 다름)
    content_selectors = [
        "article .entry-content",
        "article .post-content",
        "article .article-body",
        "article .content",
        ".story-body",
        ".article-content",
        ".post-body",
        ".entry-content",
        "article",
        ".main-content article",
        '[role="main"] article',
    ]

    content = ""

    for selector in content_selectors:
        elements = soup.select(selector)
        if elements:
            # 가장 긴 텍스트를 가진 요소 선택
            for element in elements:
                text = element.get_text(separator="\n", strip=True)
                if len(text) > len(content):
                    content = text
            if content:
                break

    # 선택자로 못 찾으면 p 태그들로 시도
    if len(content) < 100:
        paragraphs = soup.find_all("p")
        paragraph_texts = []
        for p in paragraphs:
            text = p.get_text(strip=True)
            if len(text) > 30:  # 너무 짧은 문단 제외
                paragraph_texts.append(text)

        content = "\n\n".join(paragraph_texts)

    return clean_text(content) if content else "본문을 찾을 수 없음"


def extract_meta_description(soup):
    """메타 설명 추출"""

    meta_selectors = [
        'meta[name="description"]',
        'meta[property="og:description"]',
        'meta[name="twitter:description"]',
    ]

    for selector in meta_selectors:
        meta = soup.select_one(selector)
        if meta and meta.get("content"):
            return clean_text(meta.get("content"))

    return ""


def extract_publish_date(soup):
    """발행일 추출"""

    date_selectors = [
        "time[datetime]",
        '[property="article:published_time"]',
        '[name="publish_date"]',
        ".publish-date",
        ".article-date",
        ".post-date",
    ]

    for selector in date_selectors:
        element = soup.select_one(selector)
        if element:
            date_text = (
                element.get("datetime")
                or element.get("content")
                or element.get_text(strip=True)
            )
            if date_text:
                return date_text

    return ""


def extract_author(soup):
    """저자 추출"""

    author_selectors = [
        '[rel="author"]',
        ".author-name",
        ".byline-author",
        '[property="article:author"]',
        ".article-author",
    ]

    for selector in author_selectors:
        element = soup.select_one(selector)
        if element:
            author = element.get("content") or element.get_text(strip=True)
            if author:
                return clean_text(author)

    return ""


def extract_tags(soup):
    """태그/키워드 추출"""

    tags = []

    # 메타 키워드
    keywords_meta = soup.select_one('meta[name="keywords"]')
    if keywords_meta and keywords_meta.get("content"):
        tags.extend([tag.strip() for tag in keywords_meta.get("content").split(",")])

    # 기사 태그 섹션
    tag_selectors = [".tags a", ".article-tags a", ".post-tags a", '[rel="tag"]']

    for selector in tag_selectors:
        tag_elements = soup.select(selector)
        for tag_el in tag_elements:
            tag_text = tag_el.get_text(strip=True)
            if tag_text and tag_text not in tags:
                tags.append(tag_text)

    return tags[:10]  # 최대 10개만 반환


def clean_text(text):
    """텍스트 정리 (공백, 특수문자 등)"""
    if not text:
        return ""

    # 연속된 공백, 탭, 줄바꿈 정리
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text)

    # 앞뒤 공백 제거
    text = text.strip()

    return text


if __name__ == "__main__":
    url = "https://nintendoeverything.com/nintendo-announces-my-mario-series-of-products-including-switch-app/"
    parsed_data = parse_article_content(url)
    print(parsed_data)
