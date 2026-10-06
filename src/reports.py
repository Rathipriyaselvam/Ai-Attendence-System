import pandas as pd
import io
import datetime
from fpdf import FPDF
from src.database import get_db_connection

def generate_attendance_report(start_date, end_date, department="All", year="All", section="All"):
    """
    Generate comprehensive student attendance report for a specified date range.
    Returns pandas DataFrame.
    """
    conn = get_db_connection()

    # Calculate total working days (distinct dates attendance was taken in range)
    date_query = "SELECT COUNT(DISTINCT date) FROM attendance WHERE date >= ? AND date <= ?"
    cursor = conn.cursor()
    cursor.execute(date_query, (str(start_date), str(end_date)))
    total_days = cursor.fetchone()[0]
    total_days = max(1, total_days)

    # Base student filter
    student_conditions = ["1=1"]
    params = [str(start_date), str(end_date)]

    if department and department != "All":
        student_conditions.append("s.department = ?")
        params.append(department)
    if year and year != "All":
        student_conditions.append("s.year = ?")
        params.append(year)
    if section and section != "All":
        student_conditions.append("s.section = ?")
        params.append(section)

    where_clause = " AND ".join(student_conditions)

    query = f"""
        SELECT s.student_id,
               s.name,
               s.department,
               s.year,
               s.section,
               COUNT(DISTINCT a.date) as days_present
        FROM students s
        LEFT JOIN attendance a ON s.student_id = a.student_id
                               AND a.date >= ? AND a.date <= ?
        WHERE {where_clause}
        GROUP BY s.student_id
        ORDER BY s.student_id ASC
    """
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty:
        return pd.DataFrame()

    df["total_working_days"] = total_days
    df["days_absent"] = df["total_working_days"] - df["days_present"]
    df["days_absent"] = df["days_absent"].clip(lower=0)
    df["attendance_percentage"] = ((df["days_present"] / df["total_working_days"]) * 100).round(1)

    # Rename columns for presentation
    df.rename(columns={
        "student_id": "Student ID",
        "name": "Student Name",
        "department": "Department",
        "year": "Year",
        "section": "Section",
        "total_working_days": "Total Working Days",
        "days_present": "Days Present",
        "days_absent": "Days Absent",
        "attendance_percentage": "Attendance %"
    }, inplace=True)

    return df

def export_to_csv(df):
    """Convert DataFrame to CSV bytes."""
    return df.to_csv(index=False).encode('utf-8')

def export_to_excel(df):
    """Convert DataFrame to Excel bytes using openpyxl."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Attendance Report')
    return output.getvalue()

class PDFReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(30, 41, 59)
        self.cell(0, 10, 'AI Attendance System - Analytics & Attendance Report', 0, 1, 'C')
        self.set_font('Helvetica', 'I', 10)
        self.set_text_color(100, 116, 139)
        self.cell(0, 6, f'Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def export_to_pdf(df, start_date, end_date, department="All"):
    """Generate styled PDF report file bytes."""
    pdf = PDFReport(orientation='L', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_font('Helvetica', '', 10)

    # Filter metadata banner
    pdf.set_fill_color(241, 245, 249)
    pdf.rect(10, 26, 277, 12, 'F')
    pdf.set_text_color(15, 23, 42)
    pdf.set_xy(12, 29)
    pdf.cell(0, 6, f'Filter Criteria: Date Range ({start_date} to {end_date}) | Department: {department} | Total Students: {len(df)}', 0, 1)
    pdf.ln(8)

    if df.empty:
        pdf.cell(0, 10, 'No attendance records found for the selected criteria.', 0, 1, 'C')
        return bytes(pdf.output())

    # Table Headers
    cols = ["Student ID", "Student Name", "Department", "Total Days", "Present", "Absent", "Att %"]
    col_widths = [32, 60, 65, 30, 25, 25, 30]

    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_fill_color(30, 41, 59)
    pdf.set_text_color(255, 255, 255)

    for i, col in enumerate(cols):
        pdf.cell(col_widths[i], 9, col, 1, 0, 'C', fill=True)
    pdf.ln()

    # Table Body
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(15, 23, 42)
    fill = False

    for idx, row in df.iterrows():
        pdf.set_fill_color(248, 250, 252) if fill else pdf.set_fill_color(255, 255, 255)
        
        pdf.cell(col_widths[0], 8, str(row.get("Student ID", "")), 1, 0, 'C', fill=fill)
        pdf.cell(col_widths[1], 8, str(row.get("Student Name", "")), 1, 0, 'L', fill=fill)
        pdf.cell(col_widths[2], 8, str(row.get("Department", "")), 1, 0, 'L', fill=fill)
        pdf.cell(col_widths[3], 8, str(row.get("Total Working Days", "")), 1, 0, 'C', fill=fill)
        pdf.cell(col_widths[4], 8, str(row.get("Days Present", "")), 1, 0, 'C', fill=fill)
        pdf.cell(col_widths[5], 8, str(row.get("Days Absent", "")), 1, 0, 'C', fill=fill)
        
        att_pct = float(row.get("Attendance %", 0))
        # Highlight low attendance
        if att_pct < 75.0:
            pdf.set_text_color(220, 38, 38) # Red
        else:
            pdf.set_text_color(16, 185, 129) # Green

        pdf.cell(col_widths[6], 8, f"{att_pct:.1f}%", 1, 0, 'C', fill=fill)
        pdf.set_text_color(15, 23, 42) # reset
        pdf.ln()
        fill = not fill

    return bytes(pdf.output())
