import streamlit as st
import pandas as pd
import datetime
from src.database import get_kpi_summary, get_today_attendance, get_filtered_attendance
from src.analytics import (
    create_present_vs_absent_donut,
    create_daily_attendance_trend,
    create_department_attendance_chart,
    create_weekly_attendance_chart,
    create_top_low_attendance_chart
)
from config import config

def render_dashboard_page():
    st.markdown("""
        <div class="main-header">
            <h1>📊 AI Attendance Analytics Dashboard</h1>
            <p>Real-Time Face Recognition & Attendance Intelligence System</p>
        </div>
    """, unsafe_allow_html=True)

    # Fetch KPI Summary
    kpis = get_kpi_summary()

    # KPI Cards Row
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Students</div>
                <div class="kpi-value">{kpis['total_students']}</div>
                <div class="kpi-subtext">Registered in System</div>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Present Today</div>
                <div class="kpi-value" style="color: #10B981;">{kpis['present_today']}</div>
                <div class="kpi-subtext">Marked Present</div>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Absent Today</div>
                <div class="kpi-value" style="color: #EF4444;">{kpis['absent_today']}</div>
                <div class="kpi-subtext">Not Recognized Yet</div>
            </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Attendance Rate</div>
                <div class="kpi-value" style="color: #3B82F6;">{kpis['attendance_percentage']}%</div>
                <div class="kpi-subtext">Today's Ratio</div>
            </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Records</div>
                <div class="kpi-value">{kpis['total_records']}</div>
                <div class="kpi-subtext">All-Time Logs</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Today's Attendance Table Section
    st.subheader("📋 Today's Attendance Log")

    # Filters row
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        dept_filter = st.selectbox("Department", ["All"] + config["departments"], key="dash_dept")
    with col_f2:
        year_filter = st.selectbox("Year", ["All"] + config["years"], key="dash_year")
    with col_f3:
        sec_filter = st.selectbox("Section", ["All"] + config["sections"], key="dash_sec")

    df_today = get_today_attendance()

    if not df_today.empty:
        # Apply UI filters
        if dept_filter != "All":
            df_today = df_today[df_today["department"] == dept_filter]
        if year_filter != "All":
            df_today = df_today[df_today["year"] == year_filter]
        if sec_filter != "All":
            df_today = df_today[df_today["section"] == sec_filter]

    if not df_today.empty:
        display_df = df_today[["student_id", "name", "department", "time", "confidence", "status"]].copy()
        display_df.columns = ["Student ID", "Name", "Department", "Time Marked", "Confidence (%)", "Status"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No attendance records logged for today with the selected filters.")

    st.markdown("---")

    # Analytics Visualizations Section (6 Charts)
    st.subheader("📈 Analytics & Trends")

    # Fetch all historical attendance records for charts
    df_all_attendance = get_filtered_attendance()

    row1_left, row1_right = st.columns(2)

    with row1_left:
        fig_donut = create_present_vs_absent_donut(kpis['present_today'], kpis['absent_today'])
        st.plotly_chart(fig_donut, use_container_width=True)

    with row1_right:
        fig_trend = create_daily_attendance_trend(df_all_attendance, days=14)
        st.plotly_chart(fig_trend, use_container_width=True)

    row2_left, row2_right = st.columns(2)

    with row2_left:
        fig_dept = create_department_attendance_chart()
        st.plotly_chart(fig_dept, use_container_width=True)

    with row2_right:
        fig_weekly = create_weekly_attendance_chart(df_all_attendance)
        st.plotly_chart(fig_weekly, use_container_width=True)

    row3_left, row3_right = st.columns(2)

    with row3_left:
        fig_top, low_att_df = create_top_low_attendance_chart(threshold=75.0)
        st.plotly_chart(fig_top, use_container_width=True)

    with row3_right:
        st.markdown("### ⚠️ Low Attendance Warning (< 75%)")
        if not low_att_df.empty:
            st.warning(f"Found {len(low_att_df)} student(s) below the 75% attendance criteria:")
            disp_low = low_att_df[["student_id", "name", "department", "Attendance %"]].copy()
            disp_low.columns = ["Student ID", "Name", "Department", "Attendance %"]
            st.dataframe(disp_low, use_container_width=True, hide_index=True)
        else:
            st.success("🎉 All registered students have attendance above 75%!")
