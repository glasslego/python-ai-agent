from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == "__main__":
    # 문서 예시
    documents = [
        "고양이는 귀여운 동물이다",
        "강아지는 충성스러운 동물이다",
        "고양이와 강아지는 모두 좋은 동물이다",
    ]

    # TF-IDF 벡터라이저 생성
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # 특성 이름 (단어들) 확인
    feature_names = vectorizer.get_feature_names_out()
    print("단어들:", feature_names)

    # 첫 번째 문서의 TF-IDF 점수
    print("\n문서 1의 TF-IDF 점수:")
    for i, word in enumerate(feature_names):
        score = tfidf_matrix[0, i]
        if score > 0:
            print(f"{word}: {score:.4f}")

    print("\n문서 2의 TF-IDF 점수:")
    for i, word in enumerate(feature_names):
        score = tfidf_matrix[1, i]
        if score > 0:
            print(f"{word}: {score:.4f}")

    print("\n문서 3의 TF-IDF 점수:")
    for i, word in enumerate(feature_names):
        score = tfidf_matrix[2, i]
        if score > 0:
            print(f"{word}: {score:.4f}")
