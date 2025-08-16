# flake8: noqa: E501
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFAnalyzer:
    """TF-IDF 기반 텍스트 분석"""

    # TODO
    def __init__(self, max_features=1000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range, min_df=2, max_df=0.95
        )
        self.tfidf_matrix = None
        self.feature_names = None

    def fit_transform(self, texts):
        """TF-IDF 벡터화 수행"""
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        print("✅ TF-IDF 벡터화 완료")
        print(f"   문서 수: {self.tfidf_matrix.shape[0]}")
        print(f"   특성 수: {self.tfidf_matrix.shape[1]}")

        return self.tfidf_matrix

    def print_analyzer_result(self):
        tfidf_matrix = self.tfidf_matrix
        # 1. 기본 정보
        print(f"📊 매트릭스 크기: {tfidf_matrix.shape}")
        print(f"   • 문서 수: {tfidf_matrix.shape[0]}")
        print(f"   • 특성(단어) 수: {tfidf_matrix.shape[1]}")
        print(f"   • 매트릭스 타입: {type(tfidf_matrix)}")
        print(f"   • 데이터 타입: {tfidf_matrix.dtype}")

        # 2. 희소성 정보 (Sparsity)
        total_elements = tfidf_matrix.shape[0] * tfidf_matrix.shape[1]
        non_zero_elements = tfidf_matrix.nnz
        sparsity = (1 - non_zero_elements / total_elements) * 100
        print("\n🕳️  희소성 정보:")
        print(f"   • 전체 원소 수: {total_elements:,}")
        print(f"   • 0이 아닌 원소 수: {non_zero_elements:,}")
        print(f"   • 희소성: {sparsity:.2f}% (0인 원소 비율)")

        # 3. 특성(단어) 정보
        feature_names = self.feature_names
        print("\n📝 특성(단어) 정보:")
        print(f"   • 첫 10개 단어: {list(feature_names[:10])}")
        print(f"   • 마지막 10개 단어: {list(feature_names[-10:])}")

        # 4. 문서별 통계
        print("\n📄 문서별 통계:")
        doc_word_counts = np.array(tfidf_matrix.sum(axis=1)).flatten()
        print(f"   • 문서당 평균 TF-IDF 합: {doc_word_counts.mean():.3f}")
        print(f"   • 최대 TF-IDF 합: {doc_word_counts.max():.3f}")
        print(f"   • 최소 TF-IDF 합: {doc_word_counts.min():.3f}")

        # 5. 단어별 통계
        print("\n🔤 단어별 통계:")
        word_frequencies = np.array(tfidf_matrix.sum(axis=0)).flatten()
        print(f"   • 단어당 평균 TF-IDF 합: {word_frequencies.mean():.3f}")
        print(
            f"   • 최대 TF-IDF 합을 가진 단어: {feature_names[word_frequencies.argmax()]}"
        )
        print(
            f"   • 최소 TF-IDF 합을 가진 단어: {feature_names[word_frequencies.argmin()]}"
        )

        # 6. 상위 중요 단어들
        top_word_indices = word_frequencies.argsort()[-10:][::-1]
        print("\n🏆 상위 10개 중요 단어 (TF-IDF 합 기준):")
        for i, idx in enumerate(top_word_indices, 1):
            word = feature_names[idx]
            score = word_frequencies[idx]
            print(f"   {i:2d}. {word:<15} (점수: {score:.3f})")

    def get_top_keywords(self, n_keywords=20):
        """전체 코퍼스에서 상위 키워드 추출"""
        # 각 특성의 평균 TF-IDF 점수 계산
        mean_scores = np.array(self.tfidf_matrix.mean(axis=0)).flatten()

        # 상위 키워드 추출
        top_indices = mean_scores.argsort()[-n_keywords:][::-1]
        top_keywords = [(self.feature_names[i], mean_scores[i]) for i in top_indices]

        return top_keywords

    def get_document_keywords(self, doc_index, n_keywords=10):
        """특정 문서의 상위 키워드 추출"""
        doc_scores = self.tfidf_matrix[doc_index].toarray().flatten()
        top_indices = doc_scores.argsort()[-n_keywords:][::-1]

        doc_keywords = [
            (self.feature_names[i], doc_scores[i])
            for i in top_indices
            if doc_scores[i] > 0
        ]

        return doc_keywords

    def analyze_keywords(self, df):
        """키워드 분석 수행"""
        print("\n🔍 TF-IDF 키워드 분석")
        print("-" * 40)

        # 전체 상위 키워드
        top_keywords = self.get_top_keywords(20)
        print("📈 전체 상위 키워드:")
        for i, (keyword, score) in enumerate(top_keywords, 1):
            print(f"   {i:2d}. {keyword:<20} ({score:.4f})")

        # 각 문서별 키워드
        print("\n📰 문서별 상위 키워드:")
        for i in range(min(3, len(df))):
            title = df.iloc[i]["Title"][:40]
            doc_keywords = self.get_document_keywords(i, 5)
            print(f"\n   📄 {title}...")
            for keyword, score in doc_keywords:
                print(f"      • {keyword} ({score:.3f})")

    def export_keywords_to_csv(self, df, output_file="tfidf_keywords.csv"):
        """키워드 분석 결과를 CSV로 저장"""
        results = []

        for i in range(len(df)):
            doc_keywords = self.get_document_keywords(i, 10)
            keywords = [kw[0] for kw in doc_keywords]
            scores = [kw[1] for kw in doc_keywords]

            result = {
                "Document_Index": i,
                "Title": df.iloc[i]["Title"],
                "Top_Keywords": ", ".join(keywords[:5]),
                "Top_Keyword_Scores": ", ".join(
                    [f"{score:.3f}" for score in scores[:5]]
                ),
            }
            results.append(result)

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"💾 키워드 분석 결과가 {output_file}에 저장되었습니다.")

        return results_df


def main():
    print("🚀 TF-IDF 텍스트 분석 시작")
    print("=" * 50)

    try:
        input_file = "it_news_articles_processed.csv"
        df = pd.read_csv(input_file)
        print(f"데이터 로드 완료: {len(df)}개 문서")

        # 빈 텍스트 제거
        df = df[df["processed_content"].str.len() > 0]
        print(f"유효한 문서: {len(df)}개")

        # TF-IDF 분석기 초기화
        analyzer = TFIDFAnalyzer(
            max_features=1500,  # 특성 수 증가
            ngram_range=(1, 3),  # 3-gram까지 포함
        )

        # TF-IDF 벡터화 수행
        print("\n🔄 TF-IDF 벡터화 진행 중...")
        analyzer.fit_transform(df["processed_content"])

        # 키워드 분석 수행
        analyzer.analyze_keywords(df)

        analyzer.print_analyzer_result()

        # 전체 상위 키워드를 DataFrame으로 저장
        print("\n💾 결과 저장 중...")
        top_keywords = analyzer.get_top_keywords(50)
        keywords_df = pd.DataFrame(top_keywords, columns=["Keyword", "TF-IDF_Score"])
        keywords_df.to_csv("top_keywords.csv", index=False)
        print("✅ 상위 키워드가 'top_keywords.csv'에 저장되었습니다.")

        # 문서별 키워드 분석 결과 저장
        analyzer.export_keywords_to_csv(df, "document_keywords.csv")

    except Exception as e:
        print(f"❌ 예상치 못한 오류 발생: {e}")


if __name__ == "__main__":
    main()
