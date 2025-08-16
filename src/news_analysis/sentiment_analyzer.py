# flake8: noqa: E501
import ssl
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer

# 한글 폰트 설정
plt.rcParams["font.family"] = [
    "DejaVu Sans",
    "AppleGothic",
    "Malgun Gothic",
    "gulim",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

# 폰트 경고 무시
warnings.filterwarnings(
    "ignore", category=UserWarning, module="matplotlib.font_manager"
)

# SSL 인증서 문제 해결
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


class SentimentAnalyzer:
    """감정 분석 클래스"""

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text):
        """개별 텍스트 감정 분석"""
        scores = self.sia.polarity_scores(text)

        # 감정 분류
        if scores["compound"] >= 0.05:
            sentiment = "positive"
        elif scores["compound"] <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
            "compound": scores["compound"],
        }

    def analyze_dataframe(self, df, text_column="Parsed Content"):
        """DataFrame 전체 감정 분석"""
        print("\n😊 감정 분석 수행")
        print("-" * 40)

        df_copy = df.copy()

        # 각 기사의 감정 분석
        sentiment_results = []
        for _, row in df_copy.iterrows():
            text = row[text_column]
            if isinstance(text, str) and len(text) > 0:
                result = self.analyze_sentiment(text)
                sentiment_results.append(result)
            else:
                sentiment_results.append(
                    {
                        "sentiment": "neutral",
                        "positive": 0,
                        "negative": 0,
                        "neutral": 1,
                        "compound": 0,
                    }
                )

        # 결과를 DataFrame에 추가
        sentiment_df = pd.DataFrame(sentiment_results)
        df_result = pd.concat([df_copy, sentiment_df], axis=1)

        # ========== 감정 개수 상세 분석 추가 ==========
        self.print_sentiment_counts(df_result)

        return df_result

    def print_sentiment_counts(self, df_result):
        """감정 개수 상세 분석 출력"""
        sentiment_counts = df_result["sentiment"].value_counts()
        total_docs = len(df_result)

        print("📊 감정 분포 상세:")
        print(f"   📄 총 문서 수: {total_docs}개")
        print("-" * 30)

        # 각 감정별 상세 정보
        for sentiment in ["positive", "negative", "neutral"]:
            if sentiment in sentiment_counts:
                count = sentiment_counts[sentiment]
                percentage = (count / total_docs) * 100

                # 이모지 매핑
                emoji_map = {"positive": "😊", "negative": "😞", "neutral": "😐"}

                print(
                    f"   {emoji_map[sentiment]} {sentiment.capitalize()}: {count}개 ({percentage:.1f}%)"
                )

                # 해당 감정의 compound 점수 통계
                sentiment_data = df_result[df_result["sentiment"] == sentiment]
                if len(sentiment_data) > 0:
                    avg_compound = sentiment_data["compound"].mean()
                    max_compound = sentiment_data["compound"].max()
                    min_compound = sentiment_data["compound"].min()
                    print(f"      ↳ 평균 점수: {avg_compound:.3f}")
                    print(f"      ↳ 최고 점수: {max_compound:.3f}")
                    print(f"      ↳ 최저 점수: {min_compound:.3f}")
            else:
                print(f"   😐 {sentiment.capitalize()}: 0개 (0.0%)")

        print("-" * 30)

        # 추가 통계
        print("📈 전체 통계:")
        print(
            f"   • 긍정 비율: {sentiment_counts.get('positive', 0) / total_docs * 100:.1f}%"
        )
        print(
            f"   • 부정 비율: {sentiment_counts.get('negative', 0) / total_docs * 100:.1f}%"
        )
        print(
            f"   • 중성 비율: {sentiment_counts.get('neutral', 0) / total_docs * 100:.1f}%"
        )

        # 감정 성향 분석
        positive_ratio = sentiment_counts.get("positive", 0) / total_docs
        negative_ratio = sentiment_counts.get("negative", 0) / total_docs

        if positive_ratio > negative_ratio * 1.5:
            trend = "📈 전반적으로 긍정적"
        elif negative_ratio > positive_ratio * 1.5:
            trend = "📉 전반적으로 부정적"
        else:
            trend = "⚖️  균형잡힌 감정 분포"

        print(f"   • 전체 성향: {trend}")

        # 감정별 샘플 기사
        print("\n📰 감정별 샘플 기사:")
        for sentiment in ["positive", "negative", "neutral"]:
            sample = df_result[df_result["sentiment"] == sentiment].head(1)
            if not sample.empty:
                title = sample.iloc[0]["Title"][:50]
                score = sample.iloc[0]["compound"]
                print(f"   {sentiment.capitalize()}: {title}... (점수: {score:.2f})")

    def get_detailed_sentiment_stats(self, df_result):
        """상세한 감정 통계 반환"""
        stats = {}

        for sentiment in ["positive", "negative", "neutral"]:
            sentiment_data = df_result[df_result["sentiment"] == sentiment]

            stats[sentiment] = {
                "count": len(sentiment_data),
                "percentage": len(sentiment_data) / len(df_result) * 100,
                "avg_compound": (
                    sentiment_data["compound"].mean() if len(sentiment_data) > 0 else 0
                ),
                "max_compound": (
                    sentiment_data["compound"].max() if len(sentiment_data) > 0 else 0
                ),
                "min_compound": (
                    sentiment_data["compound"].min() if len(sentiment_data) > 0 else 0
                ),
                "std_compound": (
                    sentiment_data["compound"].std() if len(sentiment_data) > 0 else 0
                ),
            }

        return stats

    def print_sentiment_ranking(self, df_result, top_n=5):
        """감정별 상위/하위 문서 랭킹"""
        print(f"\n🏆 감정 점수 랭킹 (상위 {top_n}개)")
        print("=" * 50)

        # 최고 긍정 문서들
        top_positive = df_result.nlargest(top_n, "compound")
        print("😍 가장 긍정적인 문서들:")
        for i, (_, row) in enumerate(top_positive.iterrows(), 1):
            title = row["Title"][:45]
            score = row["compound"]
            print(f"   {i}. {title}... (점수: {score:.3f})")

        # 최고 부정 문서들
        top_negative = df_result.nsmallest(top_n, "compound")
        print("\n😞 가장 부정적인 문서들:")
        for i, (_, row) in enumerate(top_negative.iterrows(), 1):
            title = row["Title"][:45]
            score = row["compound"]
            print(f"   {i}. {title}... (점수: {score:.3f})")

    def get_extreme_sentiments(self, df_result, n_samples=3):
        """가장 긍정적/부정적인 기사 찾기"""
        print("\n🎯 극단적 감정 기사 분석")
        print("-" * 40)

        # 가장 긍정적인 기사들
        most_positive = df_result.nlargest(n_samples, "compound")
        print("😍 가장 긍정적인 기사들:")
        for i, (_, row) in enumerate(most_positive.iterrows(), 1):
            title = row["Title"][:60]
            score = row["compound"]
            print(f"   {i}. {title}... (점수: {score:.3f})")

        # 가장 부정적인 기사들
        most_negative = df_result.nsmallest(n_samples, "compound")
        print("\n😞 가장 부정적인 기사들:")
        for i, (_, row) in enumerate(most_negative.iterrows(), 1):
            title = row["Title"][:60]
            score = row["compound"]
            print(f"   {i}. {title}... (점수: {score:.3f})")

    def create_sentiment_visualization(self, df_result, save_plot=True):
        """감정 분석 결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("뉴스 기사 감정 분석 결과", fontsize=16, fontweight="bold")

        # 1. 감정 분포 파이 차트
        sentiment_counts = df_result["sentiment"].value_counts()
        colors = ["#2ecc71", "#e74c3c", "#95a5a6"]  # green, red, gray
        axes[0, 0].pie(
            sentiment_counts.values,
            labels=[
                f"{label}\n({count}개)" for label, count in sentiment_counts.items()
            ],
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        axes[0, 0].set_title("감정 분포")

        # 2. Compound 점수 히스토그램
        axes[0, 1].hist(
            df_result["compound"],
            bins=30,
            color="skyblue",
            alpha=0.7,
            edgecolor="black",
        )
        axes[0, 1].axvline(x=0, color="red", linestyle="--", alpha=0.7)
        axes[0, 1].set_title("Compound 점수 분포")
        axes[0, 1].set_xlabel("Compound Score")
        axes[0, 1].set_ylabel("기사 수")

        # 3. 감정별 Compound 점수 박스플롯
        sentiment_order = ["negative", "neutral", "positive"]
        sns.boxplot(
            data=df_result,
            x="sentiment",
            y="compound",
            order=sentiment_order,
            ax=axes[1, 0],
        )
        axes[1, 0].set_title("감정별 Compound 점수 분포")
        axes[1, 0].set_xlabel("감정")
        axes[1, 0].set_ylabel("Compound Score")

        # 4. 감정 점수 상관관계
        corr_data = df_result[["positive", "negative", "neutral", "compound"]].corr()
        sns.heatmap(corr_data, annot=True, cmap="coolwarm", center=0, ax=axes[1, 1])
        axes[1, 1].set_title("감정 점수 상관관계")

        plt.tight_layout()

        if save_plot:
            plt.savefig("sentiment_analysis_results.png", dpi=300, bbox_inches="tight")
            print("📊 시각화 결과가 'sentiment_analysis_results.png'에 저장되었습니다.")

        plt.show()

    def export_sentiment_summary(self, df_result, output_file="sentiment_summary.csv"):
        """감정 분석 요약 저장"""
        # 기본 통계
        sentiment_counts = df_result["sentiment"].value_counts()

        summary_stats = {
            "Total_Articles": len(df_result),
            "Positive_Count": sentiment_counts.get("positive", 0),
            "Negative_Count": sentiment_counts.get("negative", 0),
            "Neutral_Count": sentiment_counts.get("neutral", 0),
            "Positive_Percentage": sentiment_counts.get("positive", 0)
            / len(df_result)
            * 100,
            "Negative_Percentage": sentiment_counts.get("negative", 0)
            / len(df_result)
            * 100,
            "Neutral_Percentage": sentiment_counts.get("neutral", 0)
            / len(df_result)
            * 100,
            "Average_Compound": df_result["compound"].mean(),
            "Max_Compound": df_result["compound"].max(),
            "Min_Compound": df_result["compound"].min(),
            "Std_Compound": df_result["compound"].std(),
        }

        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv(output_file, index=False)
        print(f"📊 감정 분석 요약이 '{output_file}'에 저장되었습니다.")

        return summary_stats


def main():
    print("🚀 뉴스 기사 감정 분석 시작")
    print("=" * 50)

    try:
        # 데이터 로드
        input_file = "it_news_articles_processed.csv"  # 또는 원본 데이터 파일
        df = pd.read_csv(input_file)
        print(f"📄 데이터 로드 완료: {len(df)}개 기사")

        # 사용할 텍스트 컬럼 결정
        text_column = "processed_content"
        print(f"📝 분석 대상 컬럼: '{text_column}'")

        # 빈 텍스트 제거
        initial_count = len(df)
        df = df.dropna(subset=[text_column])
        df = df[df[text_column].astype(str).str.len() > 0]
        print(f"📄 유효한 기사: {len(df)}개 (제거됨: {initial_count - len(df)}개)")

        # 감정 분석기 초기화
        analyzer = SentimentAnalyzer()

        # 감정 분석 수행
        print("\n🔄 감정 분석 진행 중...")
        df_result = analyzer.analyze_dataframe(df, text_column)

        # ========== 추가된 상세 분석 ==========

        # 상세 감정 통계
        detailed_stats = analyzer.get_detailed_sentiment_stats(df_result)
        print("\n📊 상세 감정 통계:")
        for sentiment, stats in detailed_stats.items():
            print(f"   {sentiment.capitalize()}:")
            print(f"      • 문서 수: {stats['count']}개 ({stats['percentage']:.1f}%)")
            print(f"      • 평균 점수: {stats['avg_compound']:.3f}")
            if stats["count"] > 0:
                print(f"      • 표준편차: {stats['std_compound']:.3f}")

        # 감정 점수 랭킹
        analyzer.print_sentiment_ranking(df_result, top_n=3)

        # 극단적 감정 기사 분석
        analyzer.get_extreme_sentiments(df_result, n_samples=3)

        # 전체 요약 통계
        print("\n📈 전체 요약:")
        print(f"   📄 총 분석 문서: {len(df_result)}개")
        print(
            f"   😊 긍정 문서: {len(df_result[df_result['sentiment'] == 'positive'])}개"
        )
        print(
            f"   😞 부정 문서: {len(df_result[df_result['sentiment'] == 'negative'])}개"
        )
        print(
            f"   😐 중성 문서: {len(df_result[df_result['sentiment'] == 'neutral'])}개"
        )
        print(f"   📊 평균 Compound 점수: {df_result['compound'].mean():.3f}")
        print(f"   📈 최고 점수: {df_result['compound'].max():.3f}")
        print(f"   📉 최저 점수: {df_result['compound'].min():.3f}")

        # 결과 저장
        print("\n💾 결과 저장 중...")
        output_file = "news_with_sentiment.csv"
        df_result.to_csv(output_file, index=False)
        print(f"✅ 감정 분석 결과가 '{output_file}'에 저장되었습니다.")

        # 요약 통계 저장
        analyzer.export_sentiment_summary(df_result)

        # 시각화 생성 (선택사항)
        try:
            create_viz = (
                input("\n📊 시각화를 생성하시겠습니까? (y/n): ").lower().strip()
            )
            if create_viz == "y":
                analyzer.create_sentiment_visualization(df_result)
        except Exception as e:
            print("error creating visualization:", e)

        # 감정별 키워드 분석 (보너스)
        if "processed_content" in df_result.columns:
            print("\n🔍 감정별 특징 분석:")
            for sentiment in ["positive", "negative"]:
                sentiment_texts = df_result[df_result["sentiment"] == sentiment][
                    "processed_content"
                ]
                if len(sentiment_texts) > 0:
                    # 가장 긴 텍스트의 단어들 (간단한 키워드 추출)
                    longest_text = sentiment_texts.loc[
                        sentiment_texts.str.len().idxmax()
                    ]
                    top_words = longest_text.split()[:5]
                    print(
                        f"   {sentiment.capitalize()} 대표 단어: {', '.join(top_words)}"
                    )

        print("\n🎉 감정 분석 완료!")

        # ========== 최종 요약 카운트 ==========
        final_counts = df_result["sentiment"].value_counts()
        print("\n📋 최종 결과 요약:")
        print(f"   📊 총 {len(df_result)}개 문서 분석 완료")
        print(
            f"   😊 긍정적 문서: {final_counts.get('positive', 0)}개 ({final_counts.get('positive', 0) / len(df_result) * 100:.1f}%)"
        )
        print(
            f"   😞 부정적 문서: {final_counts.get('negative', 0)}개 ({final_counts.get('negative', 0) / len(df_result) * 100:.1f}%)"
        )
        print(
            f"   😐 중성적 문서: {final_counts.get('neutral', 0)}개 ({final_counts.get('neutral', 0) / len(df_result) * 100:.1f}%)"
        )

    except Exception as e:
        print(f"❌ 예상치 못한 오류 발생: {e}")


if __name__ == "__main__":
    main()
