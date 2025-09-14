# flake8: noqa: E501
from datetime import date

from agents.finance_agent import FinanceAgent
from agents.news_agent import NewsAgent
from agents.schedule_agent import ScheduleAgent
from agents.weather_agent import WeatherAgent

import streamlit as st

# 페이지 설정
st.set_page_config(page_title="🤖 스마트 개인 비서", page_icon="🤖", layout="wide")


# 에이전트 초기화
@st.cache_resource
def init_agents():
    return {
        "news": NewsAgent(),
        "weather": WeatherAgent(),
        "finance": FinanceAgent(),
        "schedule": ScheduleAgent(),
    }


agents = init_agents()

# 사이드바 메뉴
st.sidebar.title("🤖 스마트 개인 비서")
page = st.sidebar.selectbox(
    "기능 선택", ["📰 뉴스", "🌤️ 날씨", "💱 금융", "📅 일정관리", "🔧 통합 대시보드"]
)

# 메인 콘텐츠
if page == "📰 뉴스":
    st.title("📰 뉴스 검색")

    col1, col2 = st.columns([2, 1])

    with col1:
        query = st.text_input(
            "검색어를 입력하세요", placeholder="예: 삼성전자, AI, 경제"
        )

    with col2:
        limit = st.selectbox("결과 개수", [3, 5, 10], index=1)

    if st.button("뉴스 검색", type="primary"):
        if query:
            with st.spinner("뉴스를 검색중..."):
                result = agents["news"].search_news(query, limit=limit)

                if result["status"] == "success":
                    st.success(
                        f"'{query}' 관련 뉴스 {result['total_results']}건을 찾았습니다."
                    )

                    for i, article in enumerate(result["articles"], 1):
                        with st.expander(f"{i}. {article['title']}"):
                            st.write(article["description"])
                            if article["url"]:
                                st.link_button("원문 보기", article["url"])
                else:
                    st.error(f"오류: {result['message']}")
        else:
            st.warning("검색어를 입력해주세요.")

    # 감성 분석 기능
    st.divider()
    st.subheader("📈 뉴스 감성 분석")

    if st.button("최신 헤드라인 감성 분석"):
        with st.spinner("감성 분석 중..."):
            headlines = agents["news"].get_top_headlines()
            if headlines["status"] == "success":
                sentiment_results = agents["news"].analyze_sentiment(
                    headlines["headlines"]
                )

                for result in sentiment_results:
                    sentiment_color = {"긍정적": "🟢", "부정적": "🔴", "중립적": "⚪"}

                    st.write(
                        f"{sentiment_color[result['sentiment']]} **{result['sentiment']}** - {result['title']}"
                    )

elif page == "🌤️ 날씨":
    st.title("🌤️ 날씨 정보")

    col1, col2 = st.columns([2, 1])

    with col1:
        city = st.text_input(
            "도시명을 입력하세요", value="Seoul", placeholder="예: Seoul, Tokyo"
        )

    with col2:
        forecast_days = st.selectbox("예보 일수", [1, 3, 5, 7], index=2)

    if st.button("날씨 조회", type="primary"):
        with st.spinner("날씨 정보를 가져오는 중..."):
            # 현재 날씨
            current_weather = agents["weather"].get_current_weather(city)

            if current_weather["status"] == "success":
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "🌡️ 온도", f"{current_weather['weather']['temperature']}°C"
                    )

                with col2:
                    st.metric("💧 습도", f"{current_weather['weather']['humidity']}%")

                with col3:
                    st.metric("🌤️ 날씨", current_weather["weather"]["description"])

                # 옷차림 추천
                st.subheader("👔 옷차림 추천")
                clothing = current_weather["clothing_recommendation"]

                st.info(f"**{clothing['category']}** - {clothing['advice']}")

                cols = st.columns(len(clothing["items"]))
                for i, item in enumerate(clothing["items"]):
                    with cols[i]:
                        st.write(f"• {item}")

            else:
                st.error(f"오류: {current_weather['message']}")

    # 날씨 예보
    st.divider()
    st.subheader("📅 날씨 예보")

    if st.button("예보 조회"):
        with st.spinner("예보 정보를 가져오는 중..."):
            forecast = agents["weather"].get_weather_forecast(city, forecast_days)

            if forecast["status"] == "success":
                for day_forecast in forecast["forecast"]:
                    with st.expander(
                        f"{day_forecast['date']} - {day_forecast['description']}"
                    ):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"최고: {day_forecast['temperature']['max']}°C")
                        with col2:
                            st.write(f"최저: {day_forecast['temperature']['min']}°C")
                        with col3:
                            st.write(f"습도: {day_forecast['humidity']}%")

elif page == "💱 금융":
    st.title("💱 금융 정보")

    tab1, tab2, tab3 = st.tabs(["환율", "주식", "관심종목"])

    with tab1:
        st.subheader("💱 환율 정보")

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            from_currency = st.selectbox(
                "기준 통화", ["USD", "EUR", "JPY", "KRW"], index=0
            )

        with col2:
            to_currency = st.selectbox(
                "대상 통화", ["KRW", "USD", "EUR", "JPY"], index=0
            )

        with col3:
            amount = st.number_input("금액", min_value=0.01, value=1.0, step=0.01)

        if st.button("환율 조회"):
            result = agents["finance"].calculate_currency_conversion(
                amount, from_currency, to_currency
            )

            if result["status"] == "success":
                st.success(
                    f"{result['original_amount']} {result['original_currency']} = {result['converted_amount']} {result['converted_currency']}"
                )
                st.info(
                    f"환율: 1 {from_currency} = {result['exchange_rate']} {to_currency}"
                )
            else:
                st.error(f"오류: {result['message']}")

        # 다중 환율 조회
        st.divider()
        if st.button("주요 환율 조회"):
            result = agents["finance"].get_multiple_exchange_rates("USD")

            if result["status"] == "success":
                cols = st.columns(len(result["rates"]))

                for i, (currency, rate) in enumerate(result["rates"].items()):
                    with cols[i]:
                        st.metric(f"USD → {currency}", f"{rate}")

    with tab2:
        st.subheader("📈 주식 정보")

        col1, col2 = st.columns([2, 1])

        with col1:
            symbol = st.text_input("종목코드", placeholder="예: 005930, AAPL")

        with col2:
            st.write("")  # 공백
            st.write("")  # 공백
            add_to_watch = st.checkbox("관심종목에 추가")

        if st.button("주가 조회"):
            if symbol:
                result = agents["finance"].get_stock_price(symbol)

                if result["status"] == "success":
                    st.metric(
                        f"{result['name']} ({result['symbol']})",
                        f"{result['price']} {result['currency']}",
                    )

                    if add_to_watch:
                        watch_result = agents["finance"].add_to_watchlist(
                            symbol, result["name"]
                        )
                        if watch_result["status"] == "success":
                            st.success(watch_result["message"])
                else:
                    st.error(f"오류: {result['message']}")
            else:
                st.warning("종목코드를 입력해주세요.")

    with tab3:
        st.subheader("⭐ 관심 종목")

        watchlist_result = agents["finance"].get_watchlist()

        if watchlist_result["status"] == "success" and watchlist_result["watchlist"]:
            for symbol, info in watchlist_result["watchlist"].items():
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.write(f"**{info['name']}** ({symbol})")

                with col2:
                    st.write(f"{info['current_price']} {info['currency']}")

                with col3:
                    if st.button("제거", key=f"remove_{symbol}"):
                        remove_result = agents["finance"].remove_from_watchlist(symbol)
                        if remove_result["status"] == "success":
                            st.success(remove_result["message"])
                            st.rerun()
        else:
            st.info("관심 종목이 없습니다.")

elif page == "📅 일정관리":
    st.title("📅 일정 관리")

    tab1, tab2, tab3 = st.tabs(["일정 추가", "일정 조회", "할일 관리"])

    with tab1:
        st.subheader("➕ 새 일정 추가")

        col1, col2 = st.columns(2)

        with col1:
            schedule_date = st.date_input("날짜", value=date.today())
            title = st.text_input("일정 제목")

        with col2:
            time = st.time_input("시간", value=None)
            priority = st.selectbox("우선순위", ["low", "medium", "high"], index=1)

        description = st.text_area("설명 (선택사항)")

        if st.button("일정 추가"):
            if title:
                time_str = time.strftime("%H:%M") if time else ""
                result = agents["schedule"].add_schedule(
                    str(schedule_date), title, time_str, description, priority
                )

                if result["status"] == "success":
                    st.success(result["message"])
                else:
                    st.error(f"오류: {result['message']}")
            else:
                st.warning("일정 제목을 입력해주세요.")

    with tab2:
        st.subheader("📋 일정 조회")

        col1, col2 = st.columns([2, 1])

        with col1:
            query_date = st.date_input("조회할 날짜", value=date.today())

        with col2:
            st.write("")  # 공백
            st.write("")  # 공백
            show_week = st.checkbox("주간 일정 보기")

        if st.button("일정 조회"):
            if show_week:
                result = agents["schedule"].get_week_schedule(str(query_date))

                if result["status"] == "success":
                    for date_str, schedules in result["week_schedules"].items():
                        if schedules:
                            st.subheader(f"📅 {date_str}")
                            for schedule in schedules:
                                priority_icon = {
                                    "high": "🔴",
                                    "medium": "🟡",
                                    "low": "🟢",
                                }
                                time_str = (
                                    f" ({schedule['time']})" if schedule["time"] else ""
                                )
                                st.write(
                                    f"{priority_icon[schedule['priority']]} **{schedule['title']}**{time_str}"
                                )
                                if schedule["description"]:
                                    st.write(f"   {schedule['description']}")
            else:
                result = agents["schedule"].get_schedule(str(query_date))

                if result["status"] == "success":
                    if result["schedules"]:
                        for schedule in result["schedules"]:
                            priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                            time_str = (
                                f" ({schedule['time']})" if schedule["time"] else ""
                            )

                            with st.expander(
                                f"{priority_icon[schedule['priority']]} {schedule['title']}{time_str}"
                            ):
                                if schedule["description"]:
                                    st.write(schedule["description"])

                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button(
                                        "완료", key=f"complete_{schedule['id']}"
                                    ):
                                        update_result = agents[
                                            "schedule"
                                        ].update_schedule(
                                            str(query_date),
                                            schedule["id"],
                                            completed=True,
                                        )
                                        if update_result["status"] == "success":
                                            st.success("일정이 완료되었습니다")
                                            st.rerun()

                                with col2:
                                    if st.button(
                                        "삭제", key=f"delete_{schedule['id']}"
                                    ):
                                        delete_result = agents[
                                            "schedule"
                                        ].delete_schedule(
                                            str(query_date), schedule["id"]
                                        )
                                        if delete_result["status"] == "success":
                                            st.success("일정이 삭제되었습니다")
                                            st.rerun()
                    else:
                        st.info(f"{query_date}에 등록된 일정이 없습니다.")

    with tab3:
        st.subheader("✅ 할일 관리")

        col1, col2 = st.columns([2, 1])

        with col1:
            task = st.text_input("새 할일")

        with col2:
            task_priority = st.selectbox(
                "우선순위", ["low", "medium", "high"], index=1, key="task_priority"
            )

        if st.button("할일 추가"):
            if task:
                result = agents["schedule"].add_todo(task, task_priority)
                if result["status"] == "success":
                    st.success(result["message"])
                    st.rerun()
            else:
                st.warning("할일을 입력해주세요.")

        st.divider()

        # 할일 목록 표시
        pending_todos = agents["schedule"].get_todos(completed=False)
        completed_todos = agents["schedule"].get_todos(completed=True)

        if pending_todos["status"] == "success" and pending_todos["todos"]:
            st.write("**📋 진행중인 할일**")
            for todo in pending_todos["todos"]:
                col1, col2 = st.columns([3, 1])
                with col1:
                    priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                    st.write(f"{priority_icon[todo['priority']]} {todo['task']}")
                with col2:
                    if st.button("완료", key=f"todo_complete_{todo['id']}"):
                        result = agents["schedule"].complete_todo(todo["id"])
                        if result["status"] == "success":
                            st.success("할일이 완료되었습니다")
                            st.rerun()

        if completed_todos["status"] == "success" and completed_todos["todos"]:
            st.divider()
            st.write("**✅ 완료된 할일**")
            for todo in completed_todos["todos"]:
                st.write(f"✅ {todo['task']}")

elif page == "🔧 통합 대시보드":
    st.title("🔧 통합 대시보드")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📰 최신 뉴스")
        headlines = agents["news"].get_top_headlines(limit=3)
        if headlines["status"] == "success":
            for headline in headlines["headlines"]:
                st.write(f"• {headline['title']}")

        st.subheader("📅 오늘 일정")
        today_schedule = agents["schedule"].get_schedule(str(date.today()))
        if today_schedule["status"] == "success" and today_schedule["schedules"]:
            for schedule in today_schedule["schedules"]:
                if not schedule.get("completed", False):
                    priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                    st.write(
                        f"{priority_icon[schedule['priority']]} {schedule['title']}"
                    )
        else:
            st.info("오늘 일정이 없습니다.")

    with col2:
        st.subheader("🌤️ 현재 날씨")
        weather = agents["weather"].get_current_weather("Seoul")
        if weather["status"] == "success":
            st.metric("온도", f"{weather['weather']['temperature']}°C")
            st.write(f"날씨: {weather['weather']['description']}")
            st.write(f"습도: {weather['weather']['humidity']}%")

        st.subheader("💱 주요 환율")
        rates = agents["finance"].get_multiple_exchange_rates("USD")
        if rates["status"] == "success":
            for currency, rate in list(rates["rates"].items())[:3]:
                st.write(f"USD → {currency}: {rate}")

# 사이드바 정보
st.sidebar.divider()
st.sidebar.info(
    """
**🤖 스마트 개인 비서**

- 📰 뉴스 검색 및 감성분석
- 🌤️ 날씨 정보 및 옷차림 추천
- 💱 환율 및 주식 정보
- 📅 일정 및 할일 관리
- 🔧 통합 대시보드

*API 키가 없으면 더미 데이터로 동작합니다*
"""
)
