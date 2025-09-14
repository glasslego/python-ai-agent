"""데모 데이터 생성 스크립트"""

import json
import os
from datetime import date, datetime, timedelta


def create_demo_schedule():
    """데모 일정 데이터 생성"""
    schedules = {}

    today = date.today()

    # 오늘 일정
    schedules[str(today)] = [
        {
            "id": 1,
            "title": "팀 회의",
            "time": "10:00",
            "description": "주간 팀 회의 - 프로젝트 진행상황 공유",
            "priority": "high",
            "completed": False,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        {
            "id": 2,
            "title": "클라이언트 미팅",
            "time": "14:30",
            "description": "신규 프로젝트 제안서 발표",
            "priority": "high",
            "completed": False,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    ]

    # 내일 일정
    tomorrow = today + timedelta(days=1)
    schedules[str(tomorrow)] = [
        {
            "id": 1,
            "title": "의료진 검진",
            "time": "09:00",
            "description": "연례 건강검진",
            "priority": "medium",
            "completed": False,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        {
            "id": 2,
            "title": "점심 약속",
            "time": "12:30",
            "description": "대학동기와 점심식사",
            "priority": "low",
            "completed": False,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    ]

    # 일정 파일 저장
    schedule_file = "tmp/schedules.json"
    with open(schedule_file, "w", encoding="utf-8") as f:
        json.dump(schedules, f, ensure_ascii=False, indent=2)

    print(f"✅ 데모 일정 데이터 생성 완료: {schedule_file}")


def create_demo_todos():
    """데모 할일 데이터 생성"""
    todos = [
        {
            "id": 1,
            "task": "프로젝트 문서 업데이트",
            "priority": "high",
            "due_date": str(date.today()),
            "completed": False,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        {
            "id": 2,
            "task": "이메일 답장하기",
            "priority": "medium",
            "due_date": "",
            "completed": False,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        {
            "id": 3,
            "task": "운동하기",
            "priority": "low",
            "due_date": "",
            "completed": True,
            "created_at": (datetime.now() - timedelta(days=1)).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    ]

    # 할일 파일 저장
    todo_file = "tmp/todos.json"
    with open(todo_file, "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)

    print(f"✅ 데모 할일 데이터 생성 완료: {todo_file}")


def create_demo_watchlist():
    """데모 관심종목 데이터 생성"""
    watchlist = {
        "005930": {
            "name": "삼성전자",
            "added_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "000660": {
            "name": "SK하이닉스",
            "added_date": (datetime.now() - timedelta(days=3)).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        },
        "AAPL": {
            "name": "Apple Inc.",
            "added_date": (datetime.now() - timedelta(days=7)).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        },
        "MSFT": {
            "name": "Microsoft Corporation",
            "added_date": (datetime.now() - timedelta(days=5)).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        },
    }

    # 관심종목 파일 저장
    watchlist_file = "tmp/stock_watchlist.json"
    with open(watchlist_file, "w", encoding="utf-8") as f:
        json.dump(watchlist, f, ensure_ascii=False, indent=2)

    print(f"✅ 데모 관심종목 데이터 생성 완료: {watchlist_file}")


def create_all_demo_data():
    """모든 데모 데이터 생성"""
    print("🔧 데모 데이터 생성 중...")

    # /tmp 디렉토리 확인/생성
    os.makedirs("tmp", exist_ok=True)

    create_demo_schedule()
    create_demo_todos()
    create_demo_watchlist()

    print("\n🎉 모든 데모 데이터 생성이 완료되었습니다!")
    print("\n📋 생성된 파일들:")
    print("- tmp/schedules.json (일정 데이터)")
    print("- tmp/todos.json (할일 데이터)")
    print("- tmp/stock_watchlist.json (관심종목 데이터)")


if __name__ == "__main__":
    create_all_demo_data()
