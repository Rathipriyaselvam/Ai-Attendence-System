import streamlit as st
import pandas as pd
import datetime

from config import config
from src.reports import (
    generate_attendance_report,
    export_to_csv,
    export_to_excel,
    export_to_pdf
)

def render_reports_page():
    st.markdown("""
        <div class="main-header">
            <h1>📑 Attendance Report Generator</h1>
            <p>Generate, Review and Export Official Student Attendance Reports in CSV, Excel, and PDF formats.</p>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("1. Report Parameters")

    today = datetime.date.today()
    thirty_days_ago = today - datetime.timedelta(days=30)

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        start_date = st.date_input("Report Start Date", value=thirty_days_ago)
        department = st.selectbox("Department", ["All"] + config["departments"], key="rep_dept")
    with r_col2:
        end_date = st.date_input("Report End Date", value=today)
        year = st.selectbox("Academic Year", ["All"] + config["years"], key="rep_year")

    section = st.selectbox("Section", ["All"] + config["sections"], key="rep_sec")

    st.markdown("---")

    if st.button("📊 Generate Report", type="primary", use_container_width=True):
        st.session_state["generated_report_df"] = generate_attendance_report(
            start_date=start_date,
            end_date=end_date,
            department=department,
            year=year,
            section=section
        )
        st.session_state["report_start_date"] = start_date
        st.session_state["report_end_date"] = end_date
        st.session_state["report_dept"] = department

    # Display Report Table if generated
    if "generated_report_df" in st.session_state:
        df_rep = st.session_state["generated_report_df"]
        s_date = st.session_state["report_start_date"]
        e_date = st.session_state["report_end_date"]
        d_name = st.session_state["report_dept"]

        st.subheader("2. Generated Report Preview")

        if not df_rep.empty:
            st.dataframe(df_rep, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("3. Export & Download Options")

            exp_col1, exp_col2, exp_col3 = st.columns(3)

            with exp_col1:
                csv_bytes = export_to_csv(df_rep)
                st.download_button(
                    label="📄 Download CSV Report",
                    data=csv_bytes,
                    file_name=f"attendance_report_{s_date}_to_{e_date}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with exp_col2:
                excel_bytes = export_to_excel(df_rep)
                st.download_button(
                    label="📊 Download Excel Report (.xlsx)",
                    data=excel_bytes,
                    file_name=f"attendance_report_{s_date}_to_{e_date}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            with exp_col3:
                pdf_bytes = export_to_pdf(df_rep, start_date=s_date, end_date=e_date, department=d_name)
                st.download_button(
                    label="📕 Download Official PDF Report",
                    data=pdf_bytes,
                    file_name=f"attendance_report_{s_date}_to_{e_date}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.warning("No student attendance data found matching the selected parameters.")
