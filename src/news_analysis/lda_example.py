from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == "__main__":
    # 1. 문서 준비
    documents = [
        "스마트폰 새로운 기능 출시",
        "주식 시장 상승 추세",
        "AI 기술 발전 놀라워",
    ]

    # 2. 단어 벡터화
    vectorizer = CountVectorizer(max_features=100)
    doc_word_matrix = vectorizer.fit_transform(documents)

    # 3. LDA 모델 학습
    lda = LatentDirichletAllocation(n_components=2, random_state=42)
    lda.fit(doc_word_matrix)

    # 4. 토픽별 주요 단어 확인
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-5:][::-1]]
        print(f"토픽 {topic_idx + 1}: {', '.join(top_words)}")
