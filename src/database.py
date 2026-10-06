import sqlite3
import json
import datetime
import os
import pandas as pd
from config import DB_PATH

def get_db_connection():
    """Establish connection to SQLite database."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    """Initialize database tables and indexes."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Students table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            year TEXT NOT NULL,
            section TEXT NOT NULL,
            email TEXT,
            created_at TEXT NOT NULL
        );
    """)

    # Face embeddings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            embedding TEXT NOT NULL,
            model_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students (student_id) ON DELETE CASCADE
        );
    """)

    # Attendance log table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            confidence REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'Present',
            FOREIGN KEY (student_id) REFERENCES students (student_id) ON DELETE CASCADE
        );
    """)

    # Indexes for performance optimization
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_students_student_id ON students(student_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_student_id ON face_embeddings(student_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_student_id ON attendance(student_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_date_student ON attendance(date, student_id);")

    conn.commit()
    conn.close()

# --- STUDENT MANAGEMENT ---

def add_student(student_id, name, department, year, section, email=""):
    """Register a new student."""
    conn = get_db_connection()
    cursor = conn.cursor()
    created_at = datetime.datetime.now().isoformat()
    try:
        cursor.execute("""
            INSERT INTO students (student_id, name, department, year, section, email, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (student_id.strip(), name.strip(), department, year, section, email.strip(), created_at))
        conn.commit()
        return True, "Student added successfully."
    except sqlite3.IntegrityError:
        return False, f"Student ID '{student_id}' already exists."
    except Exception as e:
        return False, f"Database error: {str(e)}"
    finally:
        conn.close()

def get_student(student_id):
    """Retrieve details for a specific student."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_students():
    """Retrieve all students as a DataFrame or list of dicts."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.*, COUNT(fe.id) as sample_count
        FROM students s
        LEFT JOIN face_embeddings fe ON s.student_id = fe.student_id
        GROUP BY s.student_id
        ORDER BY s.student_id ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def update_student(student_id, name, department, year, section, email=""):
    """Update student profile details."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE students
            SET name = ?, department = ?, year = ?, section = ?, email = ?
            WHERE student_id = ?
        """, (name.strip(), department, year, section, email.strip(), student_id))
        conn.commit()
        return True, "Student details updated successfully."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def delete_student(student_id):
    """Delete a student and cascade delete embeddings and attendance logs."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
        conn.commit()
        return True, f"Student '{student_id}' deleted."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

# --- EMBEDDING MANAGEMENT ---

def save_face_embedding(student_id, embedding, model_name="Facenet512"):
    """Save face embedding vector as JSON string in SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()
    created_at = datetime.datetime.now().isoformat()
    emb_json = json.dumps(embedding if isinstance(embedding, list) else embedding.tolist())
    cursor.execute("""
        INSERT INTO face_embeddings (student_id, embedding, model_name, created_at)
        VALUES (?, ?, ?, ?)
    """, (student_id, emb_json, model_name, created_at))
    conn.commit()
    conn.close()

def delete_student_embeddings(student_id):
    """Clear saved face embeddings for a student."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM face_embeddings WHERE student_id = ?", (student_id,))
    conn.commit()
    conn.close()

def get_all_embeddings(model_name=None):
    """Retrieve all embeddings from DB grouped by student_id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if model_name:
        cursor.execute("SELECT student_id, embedding FROM face_embeddings WHERE model_name = ?", (model_name,))
    else:
        cursor.execute("SELECT student_id, embedding FROM face_embeddings")
    rows = cursor.fetchall()
    conn.close()

    embeddings_by_student = {}
    for r in rows:
        sid = r["student_id"]
        vec = json.loads(r["embedding"])
        if sid not in embeddings_by_student:
            embeddings_by_student[sid] = []
        embeddings_by_student[sid].append(vec)
    return embeddings_by_student

def get_embeddings_count(student_id):
    """Get number of face samples saved for student."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM face_embeddings WHERE student_id = ?", (student_id,))
    res = cursor.fetchone()
    conn.close()
    return res["count"] if res else 0

# --- ATTENDANCE MANAGEMENT ---

def record_attendance(student_id, confidence, status="Present", cooldown_seconds=60, check_daily_duplicate=True):
    """
    Log student attendance if not already logged within cooldown or today.
    Returns: (bool logged, str message, dict record_info)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    timestamp_iso = now.isoformat()

    # Get student info
    cursor.execute("SELECT name, department, year, section FROM students WHERE student_id = ?", (student_id,))
    student = cursor.fetchone()
    if not student:
        conn.close()
        return False, f"Student ID {student_id} not registered.", None

    student_name = student["name"]

    # Check daily duplicate check
    if check_daily_duplicate:
        cursor.execute("SELECT id, time FROM attendance WHERE student_id = ? AND date = ?", (student_id, today_str))
        existing_today = cursor.fetchone()
        if existing_today:
            conn.close()
            return False, f"Attendance already marked today for {student_name} ({today_str} at {existing_today['time']}).", {
                "student_id": student_id,
                "name": student_name,
                "time": existing_today["time"],
                "status": "Already Marked Today"
            }

    # Check cooldown threshold
    cursor.execute("SELECT timestamp FROM attendance WHERE student_id = ? ORDER BY id DESC LIMIT 1", (student_id,))
    last_rec = cursor.fetchone()
    if last_rec:
        try:
            last_dt = datetime.datetime.fromisoformat(last_rec["timestamp"])
            elapsed = (now - last_dt).total_seconds()
            if elapsed < cooldown_seconds:
                conn.close()
                return False, f"Cooldown active ({int(cooldown_seconds - elapsed)}s remaining).", {
                    "student_id": student_id,
                    "name": student_name,
                    "time": time_str,
                    "status": "Cooldown Active"
                }
        except Exception:
            pass

    # Insert record
    cursor.execute("""
        INSERT INTO attendance (student_id, date, time, timestamp, confidence, status)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (student_id, today_str, time_str, timestamp_iso, round(float(confidence), 2), status))
    conn.commit()
    conn.close()

    return True, f"Attendance marked for {student_name} ({student_id})", {
        "student_id": student_id,
        "name": student_name,
        "date": today_str,
        "time": time_str,
        "confidence": round(float(confidence), 2),
        "status": status
    }

def get_today_attendance():
    """Get all attendance records for today with student details."""
    conn = get_db_connection()
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    query = """
        SELECT a.id, a.student_id, s.name, s.department, s.year, s.section, a.date, a.time, a.confidence, a.status
        FROM attendance a
        JOIN students s ON a.student_id = s.student_id
        WHERE a.date = ?
        ORDER BY a.id DESC
    """
    df = pd.read_sql_query(query, conn, params=[today_str])
    conn.close()
    return df

def get_filtered_attendance(start_date=None, end_date=None, department=None, year=None, section=None, student_id=None, status=None):
    """Retrieve attendance records matching filter criteria."""
    conn = get_db_connection()
    conditions = ["1=1"]
    params = []

    if start_date:
        conditions.append("a.date >= ?")
        params.append(str(start_date))
    if end_date:
        conditions.append("a.date <= ?")
        params.append(str(end_date))
    if department and department != "All":
        conditions.append("s.department = ?")
        params.append(department)
    if year and year != "All":
        conditions.append("s.year = ?")
        params.append(year)
    if section and section != "All":
        conditions.append("s.section = ?")
        params.append(section)
    if student_id and student_id.strip():
        conditions.append("(a.student_id LIKE ? OR s.name LIKE ?)")
        params.append(f"%{student_id.strip()}%")
        params.append(f"%{student_id.strip()}%")
    if status and status != "All":
        conditions.append("a.status = ?")
        params.append(status)

    where_clause = " AND ".join(conditions)
    query = f"""
        SELECT a.id, a.student_id, s.name, s.department, s.year, s.section, a.date, a.time, a.confidence, a.status
        FROM attendance a
        JOIN students s ON a.student_id = s.student_id
        WHERE {where_clause}
        ORDER BY a.date DESC, a.time DESC
    """
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def get_kpi_summary():
    """Calculate key performance metrics for Dashboard."""
    conn = get_db_connection()
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")

    cursor = conn.cursor()

    # Total registered students
    cursor.execute("SELECT COUNT(*) FROM students")
    total_students = cursor.fetchone()[0]

    # Present today (unique students)
    cursor.execute("SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date = ?", (today_str,))
    present_today = cursor.fetchone()[0]

    absent_today = max(0, total_students - present_today)
    att_percentage = (present_today / total_students * 100) if total_students > 0 else 0.0

    # Total records ever
    cursor.execute("SELECT COUNT(*) FROM attendance")
    total_records = cursor.fetchone()[0]

    conn.close()

    return {
        "total_students": total_students,
        "present_today": present_today,
        "absent_today": absent_today,
        "attendance_percentage": round(att_percentage, 1),
        "total_records": total_records
    }
