import plotly.express as px
import plotly.graph_objects as gg
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import datetime
from src.database import get_db_connection

# Shared Plotly Dark Theme layout adjustments
PLOTLY_TEMPLATE = "plotly_dark"
COLOR_PRIMARY = "#3B82F6"   # Blue
COLOR_SUCCESS = "#10B981"   # Green
COLOR_DANGER = "#EF4444"    # Red
COLOR_WARNING = "#F59E0B"   # Amber
COLOR_PURPLE = "#8B5CF6"    # Purple
BG_COLOR = "rgba(15, 23, 42, 0.6)"

def create_present_vs_absent_donut(present_count, absent_count):
    """Donut chart for Present vs Absent today."""
    labels = ["Present", "Absent"]
    values = [present_count, absent_count]
    colors = [COLOR_SUCCESS, COLOR_DANGER]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.5,
        marker_colors=colors,
        textinfo='label+percent',
        hoverinfo='label+value+percent',
        textfont_size=13
    )])

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title_text="<b>Today's Attendance Status</b>",
        title_x=0.0,
        margin=dict(t=40, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig

def create_daily_attendance_trend(df_attendance, days=14):
    """Line chart showing attendance trend over the last N days."""
    if df_attendance.empty:
        fig = go.Figure()
        fig.update_layout(template=PLOTLY_TEMPLATE, title="Daily Attendance Trend (No Data)")
        return fig

    # Group by date and count distinct present students
    daily = df_attendance.groupby("date")["student_id"].nunique().reset_index()
    daily.columns = ["Date", "Present Students"]
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.sort_values("Date").tail(days)

    fig = px.line(
        daily,
        x="Date",
        y="Present Students",
        markers=True,
        line_shape="spline",
        title="<b>Daily Attendance Trend</b>"
    )

    fig.update_traces(
        line_color=COLOR_PRIMARY,
        line_width=3,
        marker=dict(size=8, color=COLOR_PRIMARY, line=dict(width=2, color="#FFFFFF"))
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=20, l=20, r=20),
        xaxis_title="Date",
        yaxis_title="Count of Present Students"
    )
    return fig

def create_department_attendance_chart():
    """Bar chart comparing present percentage across departments."""
    conn = get_db_connection()
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")

    query = """
        SELECT s.department,
               COUNT(DISTINCT s.student_id) as total_dept_students,
               COUNT(DISTINCT a.student_id) as present_dept_students
        FROM students s
        LEFT JOIN attendance a ON s.student_id = a.student_id AND a.date = ?
        GROUP BY s.department
    """
    df = pd.read_sql_query(query, conn, params=[today_str])
    conn.close()

    if df.empty or df["total_dept_students"].sum() == 0:
        fig = go.Figure()
        fig.update_layout(template=PLOTLY_TEMPLATE, title="Department-wise Attendance (No Data)")
        return fig

    df["Attendance %"] = (df["present_dept_students"] / df["total_dept_students"] * 100).round(1)

    fig = px.bar(
        df,
        x="department",
        y=["present_dept_students", "total_dept_students"],
        barmode="group",
        title="<b>Department-wise Attendance Today</b>",
        labels={"value": "Student Count", "department": "Department", "variable": "Metric"}
    )

    # Customize legend labels
    newnames = {'present_dept_students': 'Present Today', 'total_dept_students': 'Total Registered'}
    fig.for_each_trace(lambda t: t.update(name = newnames[t.name]))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=40, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_weekly_attendance_chart(df_attendance):
    """Bar chart showing attendance grouped by day of week."""
    if df_attendance.empty:
        fig = go.Figure()
        fig.update_layout(template=PLOTLY_TEMPLATE, title="Weekly Attendance Distribution (No Data)")
        return fig

    df = df_attendance.copy()
    df["dt"] = pd.to_datetime(df["date"])
    df["DayOfWeek"] = df["dt"].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    weekly = df.groupby("DayOfWeek")["student_id"].count().reindex(day_order).fillna(0).reset_index()
    weekly.columns = ["Day", "Record Count"]

    fig = px.bar(
        weekly,
        x="Day",
        y="Record Count",
        color="Record Count",
        color_continuous_scale="Blues",
        title="<b>Weekly Attendance Volume</b>"
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=20, l=20, r=20),
        coloraxis_showscale=False
    )
    return fig

def create_top_low_attendance_chart(threshold=75.0):
    """Bar chart showing top and lowest attendance percentage students."""
    conn = get_db_connection()

    # Total unique dates in attendance database
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(DISTINCT date) FROM attendance")
    total_dates = cursor.fetchone()[0]
    total_dates = max(1, total_dates)

    query = """
        SELECT s.student_id, s.name, s.department, COUNT(DISTINCT a.date) as days_present
        FROM students s
        LEFT JOIN attendance a ON s.student_id = a.student_id
        GROUP BY s.student_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        fig = go.Figure()
        fig.update_layout(template=PLOTLY_TEMPLATE, title="Student Performance (No Data)")
        return fig, pd.DataFrame()

    df["Attendance %"] = (df["days_present"] / total_dates * 100).round(1)
    df["Label"] = df["name"] + " (" + df["student_id"] + ")"

    # Filter low attendance
    low_att_df = df[df["Attendance %"] < threshold].sort_values("Attendance %")

    df_sorted = df.sort_values("Attendance %", ascending=False)

    fig = px.bar(
        df_sorted.head(10),
        x="Attendance %",
        y="Label",
        orientation="h",
        color="Attendance %",
        color_continuous_scale="Viridis",
        title="<b>Top 10 Highest Attendance Students</b>"
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=20, l=20, r=20),
        yaxis=dict(autorange="reversed")
    )

    return fig, low_att_df
