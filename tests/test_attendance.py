import pytest
import datetime
from src.database import (
    init_db,
    add_student,
    delete_student,
    record_attendance,
    get_today_attendance
)
from src.attendance import mark_live_attendance

@pytest.fixture(autouse=True)
def setup_attendance_db():
    init_db()
    yield

def test_attendance_marking_and_daily_duplicate_prevention():
    sid = "TEST_ATT_01"
    delete_student(sid)
    add_student(sid, "Attendance Test User", "Computer Science", "3rd Year", "Section A")

    # Mark attendance first time today
    res1 = mark_live_attendance(sid, confidence=92.5, cooldown_seconds=60, check_daily=True)
    assert res1["success"] is True
    assert res1["record"]["status"] == "Present"

    # Attempt second marking immediately (daily duplicate check)
    res2 = mark_live_attendance(sid, confidence=94.0, cooldown_seconds=60, check_daily=True)
    assert res2["success"] is False
    assert "already marked today" in res2["message"].lower()

def test_cooldown_logic():
    sid = "TEST_ATT_COOL"
    delete_student(sid)
    add_student(sid, "Cooldown User", "Information Technology", "2nd Year", "Section B")

    # Mark attendance without daily constraint to test cooldown
    logged1, msg1, _ = record_attendance(sid, confidence=90.0, cooldown_seconds=60, check_daily_duplicate=False)
    assert logged1 is True

    # Immediate second attempt within 60s cooldown
    logged2, msg2, _ = record_attendance(sid, confidence=91.0, cooldown_seconds=60, check_daily_duplicate=False)
    assert logged2 is False
    assert "cooldown active" in msg2.lower()
