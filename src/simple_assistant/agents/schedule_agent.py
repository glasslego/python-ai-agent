import json
from datetime import datetime, timedelta


class ScheduleAgent:
    def __init__(self):
        self.schedule_file = "../tmp/schedules.json"
        self.todo_file = "../tmp/todos.json"

    def add_schedule(
        self,
        date: str,
        title: str,
        time: str = "",
        description: str = "",
        priority: str = "medium",
    ):
        """일정 추가"""
        try:
            try:
                with open(self.schedule_file, "r", encoding="utf-8") as f:
                    schedules = json.load(f)
            except FileNotFoundError:
                schedules = {}

            if date not in schedules:
                schedules[date] = []

            new_schedule = {
                "id": len(schedules.get(date, [])) + 1,
                "title": title,
                "time": time,
                "description": description,
                "priority": priority,
                "completed": False,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            schedules[date].append(new_schedule)

            with open(self.schedule_file, "w", encoding="utf-8") as f:
                json.dump(schedules, f, ensure_ascii=False, indent=2)

            return {
                "status": "success",
                "message": f"{date}에 '{title}' 일정이 추가되었습니다",
                "schedule": new_schedule,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_schedule(self, date: str):
        """특정 날짜 일정 조회"""
        try:
            with open(self.schedule_file, "r", encoding="utf-8") as f:
                schedules = json.load(f)

            if date in schedules:
                return {"status": "success", "date": date, "schedules": schedules[date]}
            else:
                return {
                    "status": "success",
                    "date": date,
                    "schedules": [],
                    "message": f"{date}에 등록된 일정이 없습니다",
                }
        except FileNotFoundError:
            return {
                "status": "success",
                "date": date,
                "schedules": [],
                "message": "일정 파일을 찾을 수 없습니다",
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_week_schedule(self, start_date: str = None):
        """주간 일정 조회"""
        if not start_date:
            start_date = datetime.now().strftime("%Y-%m-%d")

        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            week_schedules = {}

            for i in range(7):
                current_date = (start + timedelta(days=i)).strftime("%Y-%m-%d")
                schedule_info = self.get_schedule(current_date)
                week_schedules[current_date] = schedule_info["schedules"]

            return {
                "status": "success",
                "start_date": start_date,
                "week_schedules": week_schedules,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def update_schedule(self, date: str, schedule_id: int, **kwargs):
        """일정 수정"""
        try:
            with open(self.schedule_file, "r", encoding="utf-8") as f:
                schedules = json.load(f)

            if date in schedules:
                for schedule in schedules[date]:
                    if schedule["id"] == schedule_id:
                        for key, value in kwargs.items():
                            if key in schedule:
                                schedule[key] = value
                        schedule["updated_at"] = datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )

                        with open(self.schedule_file, "w", encoding="utf-8") as f:
                            json.dump(schedules, f, ensure_ascii=False, indent=2)

                        return {
                            "status": "success",
                            "message": "일정이 수정되었습니다",
                            "schedule": schedule,
                        }

                return {"status": "error", "message": "해당 일정을 찾을 수 없습니다"}
            else:
                return {"status": "error", "message": f"{date}에 일정이 없습니다"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def delete_schedule(self, date: str, schedule_id: int):
        """일정 삭제"""
        try:
            with open(self.schedule_file, "r", encoding="utf-8") as f:
                schedules = json.load(f)

            if date in schedules:
                schedules[date] = [s for s in schedules[date] if s["id"] != schedule_id]

                with open(self.schedule_file, "w", encoding="utf-8") as f:
                    json.dump(schedules, f, ensure_ascii=False, indent=2)

                return {"status": "success", "message": "일정이 삭제되었습니다"}
            else:
                return {"status": "error", "message": f"{date}에 일정이 없습니다"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def add_todo(self, task: str, priority: str = "medium", due_date: str = ""):
        """할일 추가"""
        try:
            try:
                with open(self.todo_file, "r", encoding="utf-8") as f:
                    todos = json.load(f)
            except FileNotFoundError:
                todos = []

            new_todo = {
                "id": len(todos) + 1,
                "task": task,
                "priority": priority,
                "due_date": due_date,
                "completed": False,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            todos.append(new_todo)

            with open(self.todo_file, "w", encoding="utf-8") as f:
                json.dump(todos, f, ensure_ascii=False, indent=2)

            return {
                "status": "success",
                "message": f"'{task}' 할일이 추가되었습니다",
                "todo": new_todo,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_todos(self, completed: bool = None):
        """할일 목록 조회"""
        try:
            with open(self.todo_file, "r", encoding="utf-8") as f:
                todos = json.load(f)

            if completed is not None:
                todos = [todo for todo in todos if todo["completed"] == completed]

            return {"status": "success", "todos": todos}
        except FileNotFoundError:
            return {"status": "success", "todos": [], "message": "할일 목록이 없습니다"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def complete_todo(self, todo_id: int):
        """할일 완료 표시"""
        try:
            with open(self.todo_file, "r", encoding="utf-8") as f:
                todos = json.load(f)

            for todo in todos:
                if todo["id"] == todo_id:
                    todo["completed"] = True
                    todo["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    with open(self.todo_file, "w", encoding="utf-8") as f:
                        json.dump(todos, f, ensure_ascii=False, indent=2)

                    return {
                        "status": "success",
                        "message": "할일이 완료되었습니다",
                        "todo": todo,
                    }

            return {"status": "error", "message": "해당 할일을 찾을 수 없습니다"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


def main():
    print("🧪 ScheduleAgent 테스트")
    agent = ScheduleAgent()

    # 일정 추가 테스트
    print("\n📅 일정 추가 테스트:")
    add_result1 = agent.add_schedule(
        "2024-03-20", "팀 회의", "10:00", "주간 팀 미팅", "high"
    )
    print(f"일정 추가 1: {add_result1}")

    add_result2 = agent.add_schedule(
        "2024-03-20", "점심 약속", "12:30", "대학동기와 점심", "medium"
    )
    print(f"일정 추가 2: {add_result2}")

    add_result3 = agent.add_schedule(
        "2024-03-21", "병원 방문", "14:00", "정기 검진", "high"
    )
    print(f"일정 추가 3: {add_result3}")

    # 특정 날짜 일정 조회 테스트
    print("\n📋 일정 조회 테스트:")
    schedule_result1 = agent.get_schedule("2024-03-20")
    print(f"2024-03-20 일정: {schedule_result1}")

    schedule_result2 = agent.get_schedule("2024-03-21")
    print(f"2024-03-21 일정: {schedule_result2}")

    schedule_result3 = agent.get_schedule("2024-03-22")
    print(f"2024-03-22 일정 (빈 날): {schedule_result3}")

    # 주간 일정 조회 테스트
    print("\n📆 주간 일정 조회 테스트:")
    week_result = agent.get_week_schedule("2024-03-20")
    print(f"주간 일정: {week_result}")

    # 일정 수정 테스트
    print("\n✏️ 일정 수정 테스트:")
    update_result = agent.update_schedule(
        "2024-03-20", 1, title="팀 회의 (수정됨)", time="10:30"
    )
    print(f"일정 수정: {update_result}")

    # 수정 후 일정 확인
    updated_schedule = agent.get_schedule("2024-03-20")
    print(f"수정 후 일정: {updated_schedule}")

    # 일정 삭제 테스트
    print("\n🗑️ 일정 삭제 테스트:")
    delete_result = agent.delete_schedule("2024-03-20", 2)
    print(f"일정 삭제: {delete_result}")

    # 삭제 후 일정 확인
    after_delete_schedule = agent.get_schedule("2024-03-20")
    print(f"삭제 후 일정: {after_delete_schedule}")

    print("\n" + "=" * 30)
    print("📝 할일 관리 테스트")
    print("=" * 30)

    # 할일 추가 테스트
    print("\n➕ 할일 추가 테스트:")
    todo_result1 = agent.add_todo("프로젝트 문서 작성", "high", "2024-03-22")
    print(f"할일 추가 1: {todo_result1}")

    todo_result2 = agent.add_todo("이메일 답장", "medium")
    print(f"할일 추가 2: {todo_result2}")

    todo_result3 = agent.add_todo("운동하기", "low")
    print(f"할일 추가 3: {todo_result3}")

    # 전체 할일 조회 테스트
    print("\n📋 전체 할일 조회 테스트:")
    all_todos = agent.get_todos()
    print(f"전체 할일: {all_todos}")

    # 미완료 할일 조회 테스트
    print("\n⏳ 미완료 할일 조회 테스트:")
    pending_todos = agent.get_todos(completed=False)
    print(f"미완료 할일: {pending_todos}")

    # 할일 완료 테스트
    print("\n✅ 할일 완료 테스트:")
    complete_result = agent.complete_todo(1)
    print(f"할일 완료: {complete_result}")

    # 완료된 할일 조회 테스트
    print("\n🎉 완료된 할일 조회 테스트:")
    completed_todos = agent.get_todos(completed=True)
    print(f"완료된 할일: {completed_todos}")

    # 완료 후 전체 할일 확인
    print("\n📊 완료 후 전체 할일 상태:")
    final_todos = agent.get_todos()
    print(f"최종 할일 목록: {final_todos}")

    print("\n🎉 모든 테스트 완료!")


if __name__ == "__main__":
    main()
