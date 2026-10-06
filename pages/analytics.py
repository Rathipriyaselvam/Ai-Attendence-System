import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import config
from src.database import get_all_students, get_filtered_attendance, get_db_connection
from src.analytics import (
    create_daily_attendance_trend,
    create_weekly_attendance_chart,
    create_department_attendance_chart,
    create_present_vs_absent_donut,
    create_top_low_attendance_chart
)

def render_analytics_page():
    st.markdown("""
        <div class="main-header">
            <h1>📉 Advanced AI Attendance Analytics</h1>
            <p>In-Depth Data Visualization, Behavioral Distribution & Student Performance Metrics</p>
        </div>
    """, unsafe_allow_html=True)

    df_all_attendance = get_filtered_attendance()

    tab_overview, tab_individual, tab_low_att = st.tabs([
        "📊 Overall Analytics",
        "👤 Individual Student Analytics",
        "⚠️ Defaulter List (< 75%)"
    ])

    # Tab 1: Overall Analytics
    with tab_overview:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Daily Attendance Volume")
            fig_daily = create_daily_attendance_trend(df_all_attendance, days=30)
            st.plotly_chart(fig_daily, use_container_width=True)

        with c2:
            st.markdown("### Weekly Distribution")
            fig_weekly = create_weekly_attendance_chart(df_all_attendance)
            st.plotly_chart(fig_weekly, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("### Department Performance Today")
            fig_dept = create_department_attendance_chart()
            st.plotly_chart(fig_dept, use_container_width=True)

        with c4:
            st.markdown("### Overall Status Distribution")
            if not df_all_attendance.empty:
                status_counts = df_all_attendance["status"].value_counts().reset_index()
                status_counts.columns = ["Status", "Count"]
                fig_pie = px.pie(
                    status_counts,
                    names="Status",
                    values="Count",
                    color="Status",
                    color_discrete_map={"Present": "#10B981", "Absent": "#EF4444"},
                    hole=0.4,
                    title="<b>Total Attendance Status Breakdown</b>"
                )
                fig_pie.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No attendance data recorded.")

    # Tab 2: Individual Student Analytics
    with tab_individual:
        students = get_all_students()
        if not students:
            st.info("No students registered.")
        else:
            student_dict = {f"{s['student_id']} - {s['name']}": s['student_id'] for s in students}
            sel_student_label = st.selectbox("Select Student for Detailed Analysis", list(student_dict.keys()))
            target_sid = student_dict[sel_student_label]

            # Fetch target student stats
            conn = get_db_connection()
            cursor = conn.cursor()

            # Total working dates in system
            cursor.execute("SELECT COUNT(DISTINCT date) FROM attendance")
            total_working_days = cursor.fetchone()[0]
            total_working_days = max(1, total_working_days)

            # Days present for student
            cursor.execute("SELECT COUNT(DISTINCT date) FROM attendance WHERE student_id = ?", (target_sid,))
            present_days = cursor.fetchone()[0]

            absent_days = max(0, total_working_days - present_days)
            att_pct = (present_days / total_working_days) * 100.0

            # Get student metadata
            cursor.execute("SELECT * FROM students WHERE student_id = ?", (target_sid,))
            st_info = dict(cursor.fetchone())
            conn.close()

            st.markdown(f"### Performance Report: **{st_info['name']}** ({target_sid})")

            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.metric("Total Classes / Working Days", total_working_days)
            with sc2:
                st.metric("Days Present", present_days)
            with sc3:
                st.metric("Days Absent", absent_days)
            with sc4:
                color_val = "#10B981" if att_pct >= 75.0 else "#EF4444"
                st.metric("Attendance Percentage", f"{att_pct:.1f}%")

            if att_pct < 75.0:
                st.error(f"⚠️ Warning: {st_info['name']}'s attendance is below the mandatory 75% requirement!")
            else:
                st.success(f"✅ {st_info['name']} satisfies the minimum attendance requirements.")

            # Student attendance timeline chart
            st_logs = df_all_attendance[df_all_attendance["student_id"] == target_sid].copy()
            if not st_logs.empty:
                st_logs["Date"] = pd.to_datetime(st_logs["date"])
                st_logs["Marked"] = 1
                fig_st = px.scatter(
                    st_logs,
                    x="Date",
                    y="time",
                    color="confidence",
                    size="confidence",
                    title=f"<b>Attendance Timeline for {st_info['name']}</b>",
                    labels={"time": "Time Marked", "confidence": "Confidence %"}
                )
                fig_st.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_st, use_container_width=True)
            else:
                st.info(f"No individual attendance records found for {st_info['name']}.")

    # Tab 3: Defaulter List (< 75%)
    with tab_low_att:
        thresh = st.slider("Set Attendance Threshold Criteria (%)", min_value=50, max_value=90, value=75, step=5)
        fig_top, low_df = create_top_low_attendance_chart(threshold=thresh)

        if not low_df.empty:
            st.error(f"🚨 Found **{len(low_df)}** student(s) with attendance below **{thresh}%**:")
            display_low = low_df[["student_id", "name", "department", "days_present", "Attendance %"]].copy()
            display_low.columns = ["Student ID", "Name", "Department", "Days Present", "Attendance %"]
            st.dataframe(display_low, use_container_width=True, hide_index=True)
        else:
            st.success(f"🎉 Excellent! All registered students meet or exceed the {thresh}% threshold.")
