"""
NLTK VADER 기반 감정 분석 모듈
"""

import ssl
import warnings
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from loguru import logger
from nltk.sentiment import SentimentIntensityAnalyzer

# 폰트 설정
plt.rcParams["font.family"] = "AppleGothic"  # MacOS

# 경고 무시
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
    """NLTK VADER 기반 감정 분석 클래스"""

    def __init__(self, download_vader: bool = True):
        """
        Args:
            download_vader: VADER 사전 자동 다운로드 여부
        """
        if download_vader:
            try:
                nltk.download("vader_lexicon", quiet=True)
            except Exception as e:
                logger.info(f"VADER 다운로드 중 에러: {e}")

        self.sia = SentimentIntensityAnalyzer()
        self.sentiment_threshold = {"positive": 0.05, "negative": -0.05}

    def analyze_single_text(self, text: str) -> Dict[str, Any]:
        """
        단일 텍스트 감정 분석

        Args:
            text: 분석할 텍스트

        Returns:
            감정 분석 결과 딕셔너리
        """
        if not isinstance(text, str) or not text.strip():
            return {
                "sentiment": "neutral",
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "compound": 0.0,
                "confidence": 0.0,
            }

        scores = self.sia.polarity_scores(text)

        # 감정 분류
        compound = scores["compound"]
        if compound >= self.sentiment_threshold["positive"]:
            sentiment = "positive"
            confidence = abs(compound)
        elif compound <= self.sentiment_threshold["negative"]:
            sentiment = "negative"
            confidence = abs(compound)
        else:
            sentiment = "neutral"
            confidence = 1 - abs(compound)  # 중성에 가까울수록 높은 신뢰도

        return {
            "sentiment": sentiment,
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
            "compound": compound,
            "confidence": confidence,
        }

    def analyze_dataframe(
        self, df: pd.DataFrame, text_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        DataFrame 전체 감정 분석

        Args:
            df: 분석할 DataFrame
            text_columns: 분석할 텍스트 컬럼 리스트
                         (None인 경우 기본 컬럼들 사용)

        Returns:
            감정 분석 결과가 추가된 DataFrame
        """
        if df.empty:
            return df

        df_copy = df.copy()

        # 기본 텍스트 컬럼 설정
        if text_columns is None:
            text_columns = [
                "Title",
                "Description",
                "Parsed Content",
                "Content",
                "processed_text",
            ]

        # 사용 가능한 컬럼 찾기
        available_columns = [col for col in text_columns if col in df_copy.columns]

        if not available_columns:
            logger.error(" 분석할 텍스트 컬럼을 찾을 수 없습니다.")
            return df_copy

        logger.info("\n😊 감정 분석 수행 중...")
        logger.info(f"   📝 분석 컬럼: {', '.join(available_columns)}")
        logger.info("-" * 50)

        # 각 기사의 감정 분석
        sentiment_results = []

        for idx, row in df_copy.iterrows():
            # 여러 텍스트 컬럼을 결합
            combined_text = ""
            for col in available_columns:
                if pd.notna(row[col]):
                    combined_text += f" {str(row[col])}"

            combined_text = combined_text.strip()
            result = self.analyze_single_text(combined_text)
            sentiment_results.append(result)

            # 진행 상황 표시
            if (idx + 1) % 10 == 0 or idx == len(df_copy) - 1:
                logger.info(
                    f"   📊 진행률: {idx + 1}/{len(df_copy)} ({(idx + 1) / len(df_copy) * 100:.1f}%)"  # noqa: E501
                )

        # 결과를 DataFrame에 추가
        sentiment_df = pd.DataFrame(sentiment_results)
        df_result = pd.concat([df_copy, sentiment_df], axis=1)

        # 감정 분포 출력
        self._print_sentiment_analysis_summary(df_result)

        return df_result

    def _print_sentiment_analysis_summary(self, df_result: pd.DataFrame) -> None:
        """감정 분석 결과 요약 출력"""
        sentiment_counts = df_result["sentiment"].value_counts()
        total_docs = len(df_result)

        logger.info("\n📊 감정 분석 결과 요약")
        logger.info("=" * 40)
        logger.info(f"📄 총 문서 수: {total_docs}개")
        logger.info("-" * 40)

        # 각 감정별 상세 정보
        emoji_map = {"positive": "😊", "negative": "😞", "neutral": "😐"}

        for sentiment in ["positive", "negative", "neutral"]:
            count = sentiment_counts.get(sentiment, 0)
            percentage = (count / total_docs) * 100

            logger.info(
                f"{emoji_map[sentiment]} {sentiment.capitalize()}: {count}개 ({percentage:.1f}%)"  # noqa: E501
            )

            # 해당 감정의 통계
            if count > 0:
                sentiment_data = df_result[df_result["sentiment"] == sentiment]
                avg_compound = sentiment_data["compound"].mean()
                avg_confidence = sentiment_data["confidence"].mean()
                logger.info(f"   ↳ 평균 점수: {avg_compound:.3f}")
                logger.info(f"   ↳ 평균 신뢰도: {avg_confidence:.3f}")

        logger.info("-" * 40)

        # 전체 통계
        avg_compound = df_result["compound"].mean()
        logger.info(f" 전체 평균 감정 점수: {avg_compound:.3f}")

        # 감정 성향 분석
        positive_ratio = sentiment_counts.get("positive", 0) / total_docs
        negative_ratio = sentiment_counts.get("negative", 0) / total_docs

        if positive_ratio > negative_ratio * 1.5:
            trend = "📈 전반적으로 긍정적 성향"
        elif negative_ratio > positive_ratio * 1.5:
            trend = "📉 전반적으로 부정적 성향"
        else:
            trend = "⚖️  균형잡힌 감정 분포"

        logger.info(f" 감정 성향: {trend}")

    def get_sentiment_statistics(self, df_result: pd.DataFrame) -> Dict[str, Any]:
        """
        상세한 감정 통계 반환

        Args:
            df_result: 감정 분석된 DataFrame

        Returns:
            감정 통계 딕셔너리
        """
        if "sentiment" not in df_result.columns:
            return {}

        stats = {
            "total_documents": len(df_result),
            "sentiment_distribution": {},
            "compound_stats": {
                "mean": float(df_result["compound"].mean()),
                "std": float(df_result["compound"].std()),
                "min": float(df_result["compound"].min()),
                "max": float(df_result["compound"].max()),
            },
        }

        # 감정별 통계
        for sentiment in ["positive", "negative", "neutral"]:
            sentiment_data = df_result[df_result["sentiment"] == sentiment]

            stats["sentiment_distribution"][sentiment] = {
                "count": len(sentiment_data),
                "percentage": len(sentiment_data) / len(df_result) * 100,
                "avg_compound": (
                    float(sentiment_data["compound"].mean())
                    if len(sentiment_data) > 0
                    else 0
                ),
                "avg_confidence": (
                    float(sentiment_data["confidence"].mean())
                    if len(sentiment_data) > 0
                    else 0
                ),
                "std_compound": (
                    float(sentiment_data["compound"].std())
                    if len(sentiment_data) > 0
                    else 0
                ),
            }

        return stats

    def get_extreme_sentiments(
        self, df_result: pd.DataFrame, n_samples: int = 3
    ) -> Dict[str, List[Dict]]:
        """
        극단적 감정 기사 추출

        Args:
            df_result: 감정 분석된 DataFrame
            n_samples: 추출할 샘플 수

        Returns:
            극단적 감정 기사 딕셔너리
        """
        if "compound" not in df_result.columns:
            return {}

        # 가장 긍정적인 기사들
        most_positive = df_result.nlargest(n_samples, "compound")
        positive_samples = []
        for _, row in most_positive.iterrows():
            positive_samples.append(
                {
                    "title": row.get("Title", "제목 없음"),
                    "compound": row["compound"],
                    "confidence": row.get("confidence", 0),
                }
            )

        # 가장 부정적인 기사들
        most_negative = df_result.nsmallest(n_samples, "compound")
        negative_samples = []
        for _, row in most_negative.iterrows():
            negative_samples.append(
                {
                    "title": row.get("Title", "제목 없음"),
                    "compound": row["compound"],
                    "confidence": row.get("confidence", 0),
                }
            )

        return {"most_positive": positive_samples, "most_negative": negative_samples}

    def visualize_sentiments(
        self, df_result: pd.DataFrame, save_path: str = "sentiment_analysis_results.png"
    ) -> None:
        """
        감정 분석 결과 시각화

        Args:
            df_result: 감정 분석된 DataFrame
            save_path: 저장할 파일 경로
        """
        if "sentiment" not in df_result.columns:
            logger.error(" 감정 분석 결과가 없습니다.")
            return

        logger.info("\n📊 감정 분석 시각화 생성 중...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("뉴스 기사 감정 분석 결과", fontsize=16, fontweight="bold")

        # 1. 감정 분포 파이 차트
        sentiment_counts = df_result["sentiment"].value_counts()
        colors = ["#2ecc71", "#e74c3c", "#95a5a6"]  # green, red, gray

        axes[0, 0].pie(
            sentiment_counts.values,
            labels=[
                f"{label.capitalize()}\n({count}개)"
                for label, count in sentiment_counts.items()
            ],
            autopct="%1.1f%%",
            colors=colors[: len(sentiment_counts)],
            startangle=90,
        )
        axes[0, 0].set_title("감정 분포", fontsize=14)

        # 2. Compound 점수 히스토그램
        axes[0, 1].hist(
            df_result["compound"],
            bins=25,
            color="skyblue",
            alpha=0.7,
            edgecolor="black",
        )
        axes[0, 1].axvline(x=0, color="red", linestyle="--", alpha=0.7, label="중성선")
        axes[0, 1].axvline(
            x=self.sentiment_threshold["positive"],
            color="green",
            linestyle=":",
            alpha=0.7,
            label="긍정 임계값",
        )
        axes[0, 1].axvline(
            x=self.sentiment_threshold["negative"],
            color="red",
            linestyle=":",
            alpha=0.7,
            label="부정 임계값",
        )
        axes[0, 1].set_title("Compound 점수 분포", fontsize=14)
        axes[0, 1].set_xlabel("Compound Score")
        axes[0, 1].set_ylabel("기사 수")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 감정별 Compound 점수 박스플롯
        sentiment_order = ["negative", "neutral", "positive"]
        available_sentiments = [
            s for s in sentiment_order if s in sentiment_counts.index
        ]

        if available_sentiments:
            sns.boxplot(
                data=df_result,
                x="sentiment",
                y="compound",
                order=available_sentiments,
                ax=axes[1, 0],
            )
        axes[1, 0].set_title("감정별 Compound 점수 분포", fontsize=14)
        axes[1, 0].set_xlabel("감정")
        axes[1, 0].set_ylabel("Compound Score")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 감정 점수 상관관계
        score_columns = ["positive", "negative", "neutral", "compound"]
        available_score_columns = [
            col for col in score_columns if col in df_result.columns
        ]

        if len(available_score_columns) > 1:
            corr_data = df_result[available_score_columns].corr()
            sns.heatmap(
                corr_data,
                annot=True,
                cmap="RdBu_r",
                center=0,
                ax=axes[1, 1],
                square=True,
            )
        axes[1, 1].set_title("감정 점수 상관관계", fontsize=14)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        logger.info(f"   📁 시각화 결과가 '{save_path}'로 저장되었습니다.")

    def export_results(
        self, df_result: pd.DataFrame, output_prefix: str = "sentiment_analysis"
    ) -> Dict[str, str]:
        """
        감정 분석 결과 내보내기

        Args:
            df_result: 감정 분석된 DataFrame
            output_prefix: 출력 파일 접두사

        Returns:
            저장된 파일 경로들
        """
        saved_files = {}

        logger.info("\n💾 감정 분석 결과 저장 중...")

        # 1. 전체 결과 저장
        results_file = f"{output_prefix}_results.csv"
        df_result.to_csv(results_file, index=False, encoding="utf-8-sig")
        saved_files["results"] = results_file

        # 2. 요약 통계 저장
        stats = self.get_sentiment_statistics(df_result)
        summary_file = f"{output_prefix}_summary.csv"

        summary_data = []
        summary_data.append(
            {"metric": "total_documents", "value": stats["total_documents"]}
        )

        for sentiment, data in stats["sentiment_distribution"].items():
            summary_data.extend(
                [
                    {"metric": f"{sentiment}_count", "value": data["count"]},
                    {"metric": f"{sentiment}_percentage", "value": data["percentage"]},
                    {
                        "metric": f"{sentiment}_avg_compound",
                        "value": data["avg_compound"],
                    },
                    {
                        "metric": f"{sentiment}_avg_confidence",
                        "value": data["avg_confidence"],
                    },
                ]
            )

        for metric, value in stats["compound_stats"].items():
            summary_data.append({"metric": f"compound_{metric}", "value": value})

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")
        saved_files["summary"] = summary_file

        # 3. 극단적 감정 기사 저장
        extremes = self.get_extreme_sentiments(df_result, 5)
        if extremes:
            extreme_data = []

            for pos_sample in extremes["most_positive"]:
                extreme_data.append(
                    {
                        "type": "most_positive",
                        "title": pos_sample["title"],
                        "compound": pos_sample["compound"],
                        "confidence": pos_sample["confidence"],
                    }
                )

            for neg_sample in extremes["most_negative"]:
                extreme_data.append(
                    {
                        "type": "most_negative",
                        "title": neg_sample["title"],
                        "compound": neg_sample["compound"],
                        "confidence": neg_sample["confidence"],
                    }
                )

            extreme_file = f"{output_prefix}_extremes.csv"
            extreme_df = pd.DataFrame(extreme_data)
            extreme_df.to_csv(extreme_file, index=False, encoding="utf-8-sig")
            saved_files["extremes"] = extreme_file

        logger.info(f"   📄 전체 결과: {results_file}")
        logger.info(f"   📊 요약 통계: {summary_file}")
        if "extremes" in saved_files:
            logger.info(f"   🎯 극단적 감정: {saved_files['extremes']}")

        return saved_files


if __name__ == "__main__":
    logger.info("😊 감정 분석 모듈 테스트")

    df = pd.read_csv("../processed_news.csv")

    # 감정 분석 수행
    analyzer = SentimentAnalyzer()
    df_result = analyzer.analyze_dataframe(df, text_columns=["Title", "processed_text"])

    # 통계 출력
    stats = analyzer.get_sentiment_statistics(df_result)
    logger.info(f"\n감정 통계: {stats}")

    # 극단적 감정 기사
    extremes = analyzer.get_extreme_sentiments(df_result, 2)
    logger.info(f"\n극단적 감정: {extremes}")
