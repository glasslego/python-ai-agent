# flake8: noqa: E501
def calculate_popularity(friend_graph, iterations=20, d=0.85):
    """
    친구 관계도를 바탕으로 PageRank 알고리즘을 이용해 인기도 점수를 계산합니다.

    Args:
        friend_graph (dict): key가 친구 이름, value가 해당 친구가 팔로우하는 친구 리스트인 딕셔너리.
        iterations (int): 점수 계산을 반복할 횟수. 많을수록 점수가 정확해집니다.
        d (float): 감쇠 계수(Damping Factor). 0.85는 85%를 의미합니다.
    """

    # --- 1. 초기 설정 ---

    # 네트워크에 있는 전체 친구 수를 계산합니다. (N)
    num_friends = len(friend_graph)

    # 모든 친구에게 공평하게 초기 인기도 점수를 1/N로 할당합니다.
    # 아직 누가 더 인기 있는지 모르기 때문에, 모두 동등한 상태에서 시작합니다.
    pagerank_scores = {friend: 1 / num_friends for friend in friend_graph}
    print("intial pagerank scores:", pagerank_scores)

    # --- 2. 점수 계산 반복 ---

    # 정해진 횟수(iterations)만큼 인기도 점수 업데이트를 반복합니다.
    # 이 과정을 반복하면서 점수가 점차 실제 인기도에 가깝게 수렴합니다.
    for i in range(iterations):
        # 이번 반복에서 계산될 새로운 점수를 저장할 임시 딕셔너리를 생성합니다.
        # 기존 점수(pagerank_scores)에 영향을 주지 않고 계산하기 위함입니다.
        new_scores = {friend: 0 for friend in friend_graph}

        # --- 2-1. 각자의 점수를 팔로워에게 나눠주기 ---

        # 모든 친구(follower)를 순회하며, 각자가 팔로우하는 사람(following_list)에게 점수를 나눠줍니다.
        for follower, following_list in friend_graph.items():
            # 팔로우하는 친구가 있을 경우에만 점수를 나눠줍니다.
            if len(following_list) > 0:
                # 자신의 현재 인기도 점수를 자신이 팔로우하는 친구 수로 나눕니다.
                # 즉, 인기도를 공평하게 1/n로 나눠주는 과정입니다.
                score_to_distribute = pagerank_scores[follower] / len(following_list)

                # 내가 팔로우하는 모든 친구(followed_friend)에게 나눠줄 점수를 더해줍니다.
                for followed_friend in following_list:
                    new_scores[followed_friend] += score_to_distribute
            # (만약 팔로우하는 친구가 없는 'dangling node'가 있다면, 이 사람의 점수는 네트워크 전체에 분배하는 로직을 추가할 수도 있습니다.)

        # --- 2-2. 감쇠 계수(d)를 적용해 최종 점수 보정 ---
        # 모든 친구들의 점수에 대해 최종 보정을 진행합니다.
        for friend in pagerank_scores:
            # (1-d)는 '랜덤 서퍼' 모델로, 친구 추천과 상관없이 새로운 페이지(친구)를 발견할 확률입니다.
            # 이 최소 점수( (1-d)/N )를 모든 친구에게 기본적으로 더해줍니다.
            # 이를 통해 아무에게도 추천받지 못하는 친구도 최소한의 점수를 가지게 되어 '인기 소외' 현상을 방지합니다.
            random_popularity = (1 - d) / num_friends

            # d는 친구의 추천을 따라갈 확률(85%)입니다.
            # 위에서 계산된 추천 점수(new_scores[friend])에 d를 곱해줍니다.
            # 최종 점수 = (랜덤하게 발견될 확률 점수) + (친구 추천을 통해 얻은 점수)
            new_scores[friend] = random_popularity + d * new_scores[friend]

        # 이번 반복에서 계산된 새로운 점수로 기존 점수를 업데이트합니다.
        pagerank_scores = new_scores

    # --- 3. 최종 결과 반환 ---
    # 모든 반복이 끝난 후, 최종적으로 계산된 인기도 점수를 반환합니다.
    return pagerank_scores


def main():
    # --- 친구 관계도 정의 ---
    friend_network = {
        "Alex": ["Bill", "Elena"],
        "Bill": ["Chloe", "Elena"],
        "Chloe": ["Alex", "Elena"],
        "David": ["Alex", "Chloe", "Frank", "Elena"],
        "Elena": ["Alex"],
        "Frank": ["David", "Grace", "Elena"],
        "Grace": ["Elena"],
    }

    # --- 인기도 계산 실행 및 결과 출력 ---
    final_scores = calculate_popularity(friend_network)

    print("\n--- 최종 인기도 점수 ---")
    sorted_scores = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
    for friend, score in sorted_scores:
        print(f"{friend}: {score:.4f}")


if __name__ == "__main__":
    main()
