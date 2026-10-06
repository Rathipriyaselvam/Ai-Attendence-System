import streamlit as st
import pandas as pd
import datetime

from config import config
from src.attendance import fetch_attendance_logs
from src.reports import export_to_csv, export_to_excel

def render_attendance_records_page():
    st.markdown("""
        <div class="main-header">
            <h1>📑 Attendance Records & Export</h1>
            <p>Filter, search, inspect historical logs, and export attendance data to CSV or Excel.</p>
        </div>
    """, unsafe_allow_html=True)

    # Filter Controls Expandable / Columns
    st.subheader("🔍 Filter & Search Criteria")

    col_d1, col_d2, col_d3 = st.columns(3)

    today = datetime.date.today()
    thirty_days_ago = today - datetime.timedelta(days=30)

    with col_d1:
        start_date = st.date_input("Start Date", value=thirty_days_ago)
    with col_d2:
        end_date = st.date_input("End Date", value=today)
    with col_d3:
        search_term = st.text_input("Search (ID or Name)", "", placeholder="e.g. CS101 or Aarav").strip()

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    with col_f1:
        department = st.selectbox("Department", ["All"] + config["departments"], key="rec_dept")
    with col_f2:
        year = st.selectbox("Year", ["All"] + config["years"], key="rec_year")
    with col_f3:
        section = st.selectbox("Section", ["All"] + config["sections"], key="rec_sec")
    with col_f4:
        status = st.selectbox("Status", ["All", "Present", "Absent"], key="rec_status")

    # Fetch Filtered Attendance Logs
    df_logs = fetch_attendance_logs(
        start_date=start_date,
        end_date=end_date,
        department=department,
        year=year,
        section=section,
        search_term=search_term,
        status=status
    )

    st.markdown("---")

    # Summary Metrics Row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Records Found", len(df_logs))
    with m2:
        unique_students = df_logs["student_id"].nunique() if not df_logs.empty else 0
        st.metric("Unique Students Present", unique_students)
    with m3:
        avg_conf = df_logs["confidence"].mean() if not df_logs.empty else 0.0
        st.metric("Average Confidence Score", f"{avg_conf:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Display Logs Table
    if not df_logs.empty:
        display_df = df_logs[["date", "time", "student_id", "name", "department", "year", "section", "confidence", "status"]].copy()
        display_df.columns = ["Date", "Time", "Student ID", "Name", "Department", "Year", "Section", "Confidence (%)", "Status"]

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Download Buttons
        st.subheader("📥 Export Records")
        ex1, ex2 = st.columns(2)

        with ex1:
            csv_data = export_to_csv(display_df)
            st.download_button(
                label="📄 Download Filtered Records as CSV",
                data=csv_data,
                file_name=f"attendance_records_{start_date}_to_{end_date}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with ex2:
            excel_data = export_to_excel(display_df)
            st.download_button(
                label="📊 Download Filtered Records as Excel (.xlsx)",
                data=excel_data,
                file_name=f"attendance_records_{start_date}_to_{end_date}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    else:
        st.warning("No attendance records matched the specified filter criteria.")
