import streamlit as st
import os

# Page configuration MUST be first Streamlit command
st.set_page_config(
    page_title="AI Attendance Analytics Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

from config import ASSETS_DIR, config
from src.database import init_db
from src.utils import load_css

# Page modules
from pages.dashboard import render_dashboard_page
from pages.live_attendance import render_live_attendance_page
from pages.register_student import render_register_student_page
from pages.students import render_students_page
from pages.attendance_records import render_attendance_records_page
from pages.analytics import render_analytics_page
from pages.reports import render_reports_page
from pages.settings import render_settings_page

def main():
    # 1. Initialize SQLite Database tables & indexes
    init_db()

    # 2. Inject Custom Stylesheet
    css_path = os.path.join(ASSETS_DIR, "style.css")
    load_css(css_path)

    # 3. Sidebar Navigation Panel
    st.sidebar.markdown("""
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <h2 style="color: #F8FAFC; font-weight: 700; margin:0;">🤖 AI Attendance</h2>
            <p style="color: #3B82F6; font-size: 0.85rem; margin-top:2px;">Face Recognition Analytics</p>
        </div>
    """, unsafe_allow_html=True)

    nav_options = [
        "Dashboard",
        "Live Attendance",
        "Register Student",
        "Students",
        "Attendance Records",
        "Analytics",
        "Reports",
        "Settings"
    ]

    icons = {
        "Dashboard": "📊",
        "Live Attendance": "📷",
        "Register Student": "👤",
        "Students": "👥",
        "Attendance Records": "📑",
        "Analytics": "📉",
        "Reports": "📝",
        "Settings": "⚙️"
    }

    # Sidebar Navigation Selection
    selected_page = st.sidebar.radio(
        "Navigation Menu",
        nav_options,
        format_func=lambda page: f"{icons.get(page, '▪')} {page}"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
        <div style="font-size: 0.75rem; color: #64748B; text-align: center;">
            <p>Model: <b>{config.get('model_name', 'Facenet512')}</b></p>
            <p>Backend: <b>{config.get('detector_backend', 'opencv')}</b></p>
            <p>© 2026 AI Attendance System</p>
        </div>
    """, unsafe_allow_html=True)

    # 4. Render Selected Page Component
    if selected_page == "Dashboard":
        render_dashboard_page()
    elif selected_page == "Live Attendance":
        render_live_attendance_page()
    elif selected_page == "Register Student":
        render_register_student_page()
    elif selected_page == "Students":
        render_students_page()
    elif selected_page == "Attendance Records":
        render_attendance_records_page()
    elif selected_page == "Analytics":
        render_analytics_page()
    elif selected_page == "Reports":
        render_reports_page()
    elif selected_page == "Settings":
        render_settings_page()

if __name__ == "__main__":
    main()
