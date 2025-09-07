# flake8: noqa: E501
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "AppleGothic"


class PageRankCalculator:
    """PageRank 알고리즘을 구현한 클래스"""

    def __init__(self, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
        self.damping_factor = damping_factor  # 감쇠 인수 (보통 0.85)
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def create_adjacency_matrix(
        self, links: Dict[str, List[str]]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        링크 정보로부터 인접 행렬을 생성

        Args:
            links: {페이지: [연결된_페이지들]} 형태의 딕셔너리

        Returns:
            adjacency_matrix: 인접 행렬
            node_names: 노드 이름 리스트
        """
        # 모든 페이지 이름 수집
        all_pages = set(links.keys())
        for page_list in links.values():
            all_pages.update(page_list)

        node_names = sorted(list(all_pages))
        n = len(node_names)
        name_to_idx = {name: idx for idx, name in enumerate(node_names)}

        # 인접 행렬 초기화
        adj_matrix = np.zeros((n, n))

        # 링크 정보를 인접 행렬에 반영
        for from_page, to_pages in links.items():
            from_idx = name_to_idx[from_page]
            for to_page in to_pages:
                to_idx = name_to_idx[to_page]
                adj_matrix[from_idx][to_idx] = 1

        return adj_matrix, node_names

    def create_transition_matrix(self, adj_matrix: np.ndarray) -> np.ndarray:
        """
        인접 행렬로부터 전이 행렬 생성
        각 행의 합이 1이 되도록 정규화
        """
        n = adj_matrix.shape[0]
        transition_matrix = adj_matrix.copy().astype(float)

        # 각 행(페이지)에 대해 나가는 링크 수로 나누기
        for i in range(n):
            outgoing_links = np.sum(transition_matrix[i])
            if outgoing_links > 0:
                transition_matrix[i] = transition_matrix[i] / outgoing_links
            else:
                # 나가는 링크가 없는 경우 모든 페이지로 균등하게 연결
                transition_matrix[i] = np.ones(n) / n

        return transition_matrix

    def calculate_pagerank_manual(self, adj_matrix: np.ndarray) -> np.ndarray:
        """
        PageRank를 수동으로 계산
        PR(A) = (1-d)/N + d * Σ(PR(Ti)/C(Ti))
        """
        n = adj_matrix.shape[0]

        # 전이 행렬 생성
        transition_matrix = self.create_transition_matrix(adj_matrix)

        # 초기 PageRank 값 (균등 분배)
        pagerank = np.ones(n) / n

        # Google 매트릭스 생성
        # G = d * M + (1-d)/n * J (J는 모든 원소가 1인 행렬)
        google_matrix = self.damping_factor * transition_matrix + (
            1 - self.damping_factor
        ) / n * np.ones((n, n))

        # 반복 계산
        for iteration in range(self.max_iterations):
            new_pagerank = google_matrix.T @ pagerank

            # 수렴 확인
            if np.linalg.norm(new_pagerank - pagerank) < self.tolerance:
                print(f"수렴 완료: {iteration + 1}번 반복")
                break

            pagerank = new_pagerank

        return pagerank

    def calculate_pagerank_power_iteration(self, adj_matrix: np.ndarray) -> np.ndarray:
        """
        Power Iteration 방법으로 PageRank 계산
        """
        n = adj_matrix.shape[0]
        transition_matrix = self.create_transition_matrix(adj_matrix)

        # 초기 PageRank 값
        pagerank = np.ones(n) / n

        for iteration in range(self.max_iterations):
            new_pagerank = (1 - self.damping_factor) / n + self.damping_factor * (
                transition_matrix.T @ pagerank
            )

            # 수렴 확인
            if np.linalg.norm(new_pagerank - pagerank) < self.tolerance:
                print(f"Power Iteration 수렴: {iteration + 1}번 반복")
                break

            pagerank = new_pagerank

        return pagerank


def create_sample_web_graph():
    """예시 웹 그래프 생성 - 학교 인기도 시나리오"""
    # 학교 친구들의 추천 관계를 웹페이지 링크로 모델링
    links = {
        "은영": ["친구", "지민"],  # 인기 있는 친구가 은영을 추천
        "친구": ["민수", "지수", "한솔", "태영", "수진"],  # 친구가 많은 사람들을 추천
        "민수": ["친구"],  # 친구를 역추천
        "지수": ["친구"],
        "한솔": ["친구", "은영"],  # 친구와 은영 둘 다 추천
        "태영": ["친구"],
        "수진": ["친구"],
        "지민": ["은영"],  # 은영을 추천
        "철수": [
            "영희",
            "민철",
            "상우",
            "진영",
            "수연",
            "미영",
            "동현",
            "예진",
            "성민",
            "하늘",
        ],  # 10명이 추천
        "영희": [],
        "민철": [],
        "상우": [],
        "진영": [],
        "수연": [],
        "미영": [],
        "동현": [],
        "예진": [],
        "성민": [],
        "하늘": [],
    }
    return links


def create_academic_web_graph():
    """학술 논문 인용 관계 예시"""
    links = {
        "AI_Survey_Paper": ["Deep_Learning", "Machine_Learning", "Neural_Networks"],
        "Deep_Learning": ["CNN_Paper", "RNN_Paper", "Transformer"],
        "Machine_Learning": ["SVM_Paper", "Decision_Tree"],
        "Neural_Networks": ["Perceptron", "Backprop"],
        "Transformer": ["BERT", "GPT", "Attention"],
        "BERT": ["NLP_Applications"],
        "GPT": ["Language_Models"],
        "CNN_Paper": ["Computer_Vision"],
        "RNN_Paper": ["Sequence_Models"],
        "SVM_Paper": [],
        "Decision_Tree": [],
        "Perceptron": [],
        "Backprop": [],
        "Attention": ["BERT", "GPT"],
        "NLP_Applications": [],
        "Language_Models": [],
        "Computer_Vision": [],
        "Sequence_Models": [],
    }
    return links


def visualize_graph_with_pagerank(
    links: Dict[str, List[str]],
    pagerank_scores: np.ndarray,
    node_names: List[str],
    title: str = "PageRank 시각화",
):
    """그래프와 PageRank 점수를 시각화"""

    # NetworkX 그래프 생성
    G = nx.DiGraph()

    # 노드와 엣지 추가
    for from_node, to_nodes in links.items():
        for to_node in to_nodes:
            G.add_edge(from_node, to_node)

    # 모든 노드가 그래프에 포함되도록 보장
    for node in node_names:
        if node not in G.nodes():
            G.add_node(node)

    # 레이아웃 설정
    plt.figure(figsize=(15, 10))

    # 스프링 레이아웃 사용 (노드들이 자연스럽게 배치됨)
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

    # PageRank 점수에 따른 노드 크기 설정
    name_to_score = {name: score for name, score in zip(node_names, pagerank_scores)}
    node_sizes = [name_to_score.get(node, 0) * 10000 for node in G.nodes()]

    # PageRank 점수에 따른 색상 설정
    node_colors = [name_to_score.get(node, 0) for node in G.nodes()]

    # 노드 그리기
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.Reds,
        alpha=0.8,
    )

    # 엣지 그리기
    nx.draw_networkx_edges(
        G, pos, edge_color="gray", arrows=True, arrowsize=20, arrowstyle="->", alpha=0.6
    )

    # 라벨 그리기
    labels = {}
    for node in G.nodes():
        score = name_to_score.get(node, 0)
        labels[node] = f"{node}\n({score:.3f})"

    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold")

    plt.title(title, fontsize=16, fontweight="bold", pad=20)

    # 컬러바 추가
    if nodes:
        cbar = plt.colorbar(nodes, shrink=0.8)
        cbar.set_label("PageRank Score", fontsize=12)

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def create_pagerank_comparison_chart(results: Dict[str, Dict]):
    """다양한 그래프의 PageRank 결과 비교"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("PageRank 결과 비교 분석", fontsize=16, fontweight="bold")

    # 1. 학교 친구 관계 - Top 5
    school_data = results["school"]
    school_df = (
        pd.DataFrame({"Node": school_data["names"], "PageRank": school_data["scores"]})
        .sort_values("PageRank", ascending=False)
        .head(10)
    )

    axes[0, 0].bar(
        range(len(school_df)), school_df["PageRank"], color="skyblue", alpha=0.7
    )
    axes[0, 0].set_title("학교 친구 관계 PageRank Top 10")
    axes[0, 0].set_ylabel("PageRank Score")
    axes[0, 0].set_xticks(range(len(school_df)))
    axes[0, 0].set_xticklabels(school_df["Node"], rotation=45, ha="right")

    # 2. 학술 논문 관계 - Top 10
    academic_data = results["academic"]
    academic_df = (
        pd.DataFrame(
            {"Node": academic_data["names"], "PageRank": academic_data["scores"]}
        )
        .sort_values("PageRank", ascending=False)
        .head(10)
    )

    axes[0, 1].bar(
        range(len(academic_df)), academic_df["PageRank"], color="lightcoral", alpha=0.7
    )
    axes[0, 1].set_title("학술 논문 인용 PageRank Top 10")
    axes[0, 1].set_ylabel("PageRank Score")
    axes[0, 1].set_xticks(range(len(academic_df)))
    axes[0, 1].set_xticklabels(academic_df["Node"], rotation=45, ha="right")

    # 3. 점수 분포 히스토그램
    axes[1, 0].hist(
        school_data["scores"], bins=10, alpha=0.7, label="학교 관계", color="skyblue"
    )
    axes[1, 0].hist(
        academic_data["scores"],
        bins=10,
        alpha=0.7,
        label="학술 관계",
        color="lightcoral",
    )
    axes[1, 0].set_title("PageRank 점수 분포")
    axes[1, 0].set_xlabel("PageRank Score")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend()

    # 4. 상위 노드들의 직접 비교
    top_school = school_df.head(5)
    top_academic = academic_df.head(5)

    x_pos = np.arange(5)
    width = 0.35

    axes[1, 1].bar(
        x_pos - width / 2,
        top_school["PageRank"],
        width,
        label="학교 관계",
        alpha=0.7,
        color="skyblue",
    )
    axes[1, 1].bar(
        x_pos + width / 2,
        top_academic["PageRank"],
        width,
        label="학술 관계",
        alpha=0.7,
        color="lightcoral",
    )

    axes[1, 1].set_title("Top 5 노드 비교")
    axes[1, 1].set_ylabel("PageRank Score")
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(["1위", "2위", "3위", "4위", "5위"])
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


def demonstrate_pagerank_concepts():
    """PageRank의 핵심 개념 시연"""
    print("=" * 70)
    print("PageRank 알고리즘 핵심 개념 시연")
    print("=" * 70)

    print("\n1. 기본 원리:")
    print("   - 많은 링크를 받는 페이지 = 중요한 페이지")
    print("   - 중요한 페이지로부터 받는 링크 = 더 가치 있는 링크")
    print("   - 감쇠 인수(d=0.85): 사용자가 링크를 따라갈 확률")

    print("\n2. 수식 상세 설명:")
    print("   PR(A) = (1-d)/N + d * Σ(PR(Ti)/C(Ti))")
    print("   ")
    print("   각 항목의 의미:")
    print("   - PR(A): 페이지 A의 PageRank 점수")
    print("   - d: 감쇠 인수 (보통 0.85, 링크를 따라갈 확률)")
    print("   - N: 전체 페이지 수")
    print("   - Ti: 페이지 A로 링크하는 페이지들")
    print("   - C(Ti): 페이지 Ti의 나가는 링크 수")
    print("   - (1-d)/N: 랜덤 점프로 인한 기본 점수")
    print("   - d * Σ(...): 실제 링크 구조를 통한 점수 전파")

    print("\n3. 알고리즘 작동 과정:")
    print("   STEP 1: 모든 페이지에 1/N의 초기 점수 할당")
    print("   STEP 2: 각 페이지의 점수를 나가는 링크 수로 나누어 분배")
    print("   STEP 3: 각 페이지가 받은 점수들을 합산")
    print("   STEP 4: 감쇠 인수를 적용하고 랜덤 점프 점수 추가")
    print("   STEP 5: 수렴할 때까지 STEP 2-4 반복")

    print("\n4. 감쇠 인수의 역할:")
    print("   - d=0.85: 85% 확률로 링크 따라감, 15% 확률로 랜덤 점프")
    print("   - 높은 d: 링크 구조가 더 중요 (실제 연결 관계 강조)")
    print("   - 낮은 d: 모든 페이지가 더 균등한 점수 (민주적 분배)")
    print("   - d=0: 모든 페이지 동일 점수 (1/N)")
    print("   - d=1: 순수 링크 구조만 반영 (수렴 문제 발생 가능)")


def test_pagerank_algorithms():
    """PageRank 알고리즘 테스트"""

    demonstrate_pagerank_concepts()

    calc = PageRankCalculator(damping_factor=0.85)
    results = {}

    print("\n" + "=" * 70)
    print("테스트 1: 학교 친구 관계 (은영 vs 철수)")
    print("=" * 70)

    # 학교 친구 관계 테스트
    school_links = create_sample_web_graph()
    school_adj, school_names = calc.create_adjacency_matrix(school_links)
    school_pagerank = calc.calculate_pagerank_manual(school_adj)

    print("\n학교 친구 관계 PageRank 결과:")
    school_results = list(zip(school_names, school_pagerank))
    school_results.sort(key=lambda x: x[1], reverse=True)

    for i, (name, score) in enumerate(school_results[:10]):
        print(f"{i + 1:2d}. {name:15s}: {score:.6f}")

    results["school"] = {"names": school_names, "scores": school_pagerank}

    print("\n핵심 인사이트:")
    eun_young_score = school_pagerank[school_names.index("은영")]
    cheol_su_score = school_pagerank[school_names.index("철수")]
    friend_score = school_pagerank[school_names.index("친구")]

    print(f"  은영: {eun_young_score:.6f} (인기 있는 '친구'로부터 링크)")
    print(f"  철수: {cheol_su_score:.6f} (10명으로부터 링크, 하지만 인기도 낮음)")
    print(f"  친구: {friend_score:.6f} (가장 높은 점수 - 허브 역할)")

    print("\n" + "=" * 70)
    print("테스트 2: 학술 논문 인용 관계")
    print("=" * 70)

    # 학술 논문 관계 테스트
    academic_links = create_academic_web_graph()
    academic_adj, academic_names = calc.create_adjacency_matrix(academic_links)
    academic_pagerank = calc.calculate_pagerank_manual(academic_adj)

    print("\n학술 논문 PageRank 결과:")
    academic_results = list(zip(academic_names, academic_pagerank))
    academic_results.sort(key=lambda x: x[1], reverse=True)

    for i, (name, score) in enumerate(academic_results[:10]):
        print(f"{i + 1:2d}. {name:20s}: {score:.6f}")

    results["academic"] = {"names": academic_names, "scores": academic_pagerank}

    # 시각화
    print("\n그래프 시각화를 생성합니다...")

    try:
        # 학교 관계 시각화
        visualize_graph_with_pagerank(
            school_links, school_pagerank, school_names, "학교 친구 관계 PageRank"
        )

        # 학술 관계 시각화
        visualize_graph_with_pagerank(
            academic_links, academic_pagerank, academic_names, "학술 논문 인용 PageRank"
        )

        # 비교 차트
        create_pagerank_comparison_chart(results)

    except Exception as e:
        print(f"시각화를 건너뜁니다: {e}")

    return results


def analyze_pagerank_properties():
    """PageRank의 특성 분석"""
    print("\n" + "=" * 70)
    print("PageRank 특성 분석")
    print("=" * 70)

    calc = PageRankCalculator()

    # 다양한 감쇠 인수로 테스트
    damping_factors = [0.5, 0.85, 0.95]
    school_links = create_sample_web_graph()
    school_adj, school_names = calc.create_adjacency_matrix(school_links)

    print("\n감쇠 인수(damping factor)에 따른 PageRank 변화:")
    print("감쇠 인수가 높을수록 링크 구조의 영향이 커집니다.")

    for d in damping_factors:
        calc.damping_factor = d
        pagerank = calc.calculate_pagerank_manual(school_adj)

        print(f"\n감쇠 인수 = {d}")
        results = list(zip(school_names, pagerank))
        results.sort(key=lambda x: x[1], reverse=True)

        for name, score in results[:5]:
            print(f"  {name:10s}: {score:.6f}")


def main():
    """메인 실행 함수"""
    print("PageRank 알고리즘 데모 프로그램")
    print("=" * 70)

    # 1. PageRank 알고리즘 테스트
    results = test_pagerank_algorithms()
    print(results)

    # 2. PageRank 특성 분석
    analyze_pagerank_properties()

    # 3. 결론 및 해석
    print("\n" + "=" * 70)
    print("결론 및 해석")
    print("=" * 70)
    print("1. PageRank는 단순히 링크 개수가 아닌 '질적 중요도'를 측정")
    print("2. 허브 페이지(많은 페이지로 연결)의 중요도가 높음")
    print("3. 감쇠 인수는 링크 구조 vs 균등 분포의 균형을 조절")
    print("4. 실제 검색 엔진에서는 200개 이상의 다른 요소들과 함께 사용")


if __name__ == "__main__":
    main()
