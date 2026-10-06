from src.database import (
    record_attendance,
    get_today_attendance,
    get_student,
    get_all_students,
    get_filtered_attendance
)
import datetime

def mark_live_attendance(student_id, confidence, cooldown_seconds=60, check_daily=True):
    """
    Attempt to log attendance for a recognized student ID.
    Returns dict containing status, message, and record info.
    """
    if not student_id:
        return {
            "success": False,
            "message": "No student ID provided.",
            "record": None
        }

    logged, message, record_info = record_attendance(
        student_id=student_id,
        confidence=confidence,
        status="Present",
        cooldown_seconds=cooldown_seconds,
        check_daily_duplicate=check_daily
    )

    return {
        "success": logged,
        "message": message,
        "record": record_info
    }

def fetch_attendance_logs(start_date=None, end_date=None, department="All", year="All", section="All", search_term="", status="All"):
    """Fetch filtered attendance records."""
    df = get_filtered_attendance(
        start_date=start_date,
        end_date=end_date,
        department=department,
        year=year,
        section=section,
        student_id=search_term,
        status=status
    )
    return df
