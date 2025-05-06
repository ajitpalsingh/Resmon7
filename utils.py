import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# ---------- Load Data ----------
@st.cache_data
def load_data(file):
    """
    Load data from Excel file with caching.
    
    Args:
        file: Uploaded file or file path
    
    Returns:
        Tuple of dataframes (issues, skills, worklogs, leaves)
    """
    if file is not None:
        try:
            xls = pd.ExcelFile(file)
            sheet_names = xls.sheet_names
            issues = xls.parse("Issues")
            skills = xls.parse("Skills")
            worklogs = xls.parse("Worklogs")
            leaves = xls.parse("Non_Availability")
            
            # Check if there are additional worksheets for specific features
            tech_debt = None
            if "Technical Debt" in sheet_names:
                tech_debt = xls.parse("Technical Debt")
                # Ensure datetime parsing for tech debt
                if 'Created Date' in tech_debt.columns:
                    tech_debt['Created Date'] = pd.to_datetime(tech_debt['Created Date'], errors='coerce')
            
            # Ensure datetime parsing
            for df in [issues]:
                if 'Start Date' in df.columns:
                    df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
                if 'Due Date' in df.columns:
                    df['Due Date'] = pd.to_datetime(df['Due Date'], errors='coerce')
                if 'Resolution Date' in df.columns:
                    df['Resolution Date'] = pd.to_datetime(df['Resolution Date'], errors='coerce')
            
            if 'Date' in worklogs.columns:
                worklogs['Date'] = pd.to_datetime(worklogs['Date'], errors='coerce')
            
            if 'Start Date' in leaves.columns:
                leaves['Start Date'] = pd.to_datetime(leaves['Start Date'], errors='coerce')
            if 'End Date' in leaves.columns:
                leaves['End Date'] = pd.to_datetime(leaves['End Date'], errors='coerce')
                
            # Return tech_debt as the fifth element in the tuple if available
            return issues, skills, worklogs, leaves, tech_debt
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None, None, None, None
    return None, None, None, None, None

# ---------- Data Integrity Checker ----------
def check_data_integrity(issues_df, skills_df, worklogs_df, leaves_df):
    """
    Check data integrity and consistency across all dataframes.
    
    Args:
        issues_df: Issues dataframe
        skills_df: Skills dataframe
        worklogs_df: Worklogs dataframe
        leaves_df: Non-availability/leaves dataframe
        
    Returns:
        Dictionary with integrity check results and recommendations
    """
    integrity_results = {
        "errors": [],
        "warnings": [],
        "info": [],
        "recommendations": []
    }
    
    if issues_df is None or skills_df is None or worklogs_df is None or leaves_df is None:
        integrity_results["errors"].append("One or more required datasets are missing")
        return integrity_results
    
    # Check 1: Dates logic - Start Date should be before Due Date
    if 'Start Date' in issues_df.columns and 'Due Date' in issues_df.columns:
        invalid_dates = issues_df[(~issues_df['Start Date'].isna()) & 
                               (~issues_df['Due Date'].isna()) & 
                               (issues_df['Start Date'] > issues_df['Due Date'])]
        
        if len(invalid_dates) > 0:
            integrity_results["errors"].append(
                f"Found {len(invalid_dates)} tasks where Start Date is after Due Date")
            integrity_results["recommendations"].append(
                "Review tasks with invalid date ranges (Start Date > Due Date)")
    
    # Check 2: Find ghost users (users in worklogs but not in skills)
    # Handle both Resource and Name column naming in skills_df
    if 'Resource' in skills_df.columns and 'Name' not in skills_df.columns:
        # Rename Resource to Name for consistency in checks
        skills_df = skills_df.rename(columns={'Resource': 'Name'})
            
    if 'User' in worklogs_df.columns and 'Name' in skills_df.columns:
        worklog_users = set(worklogs_df['User'].dropna().unique())
        skill_users = set(skills_df['Name'].dropna().unique())
        ghost_users = worklog_users - skill_users
        
        if ghost_users:
            integrity_results["warnings"].append(
                f"Found {len(ghost_users)} users with worklogs but no skills data")
            if len(ghost_users) <= 10:  # Only show if not too many
                integrity_results["info"].append(
                    f"Ghost users: {', '.join(ghost_users)}")
            integrity_results["recommendations"].append(
                "Add missing users to the Skills dataset")
    
    # Check 3: Check for inconsistent skill data
    # Handle both Resource and Name column naming
    if 'Resource' in skills_df.columns and 'Name' not in skills_df.columns:
        # Rename Resource to Name for consistency in checks
        skills_df = skills_df.rename(columns={'Resource': 'Name'})
        
    if 'Name' in skills_df.columns and 'Skillset' in skills_df.columns:
        missing_skills = skills_df[skills_df['Skillset'].isna()]
        if len(missing_skills) > 0:
            integrity_results["warnings"].append(
                f"Found {len(missing_skills)} users with missing skill information")
            integrity_results["recommendations"].append(
                "Complete skill information for team members")
    
    # Check 4: Detect overlapping leaves and tasks
    if ('User' in leaves_df.columns and 'Start Date' in leaves_df.columns and 'End Date' in leaves_df.columns 
        and 'Assignee' in issues_df.columns and 'Due Date' in issues_df.columns):
        
        conflicts = []
        leaves_by_user = {}
        
        # Organize leaves by user
        for _, leave in leaves_df.iterrows():
            user = leave.get('User')
            start = leave.get('Start Date')
            end = leave.get('End Date')
            
            if pd.notna(user) and pd.notna(start) and pd.notna(end):
                if user not in leaves_by_user:
                    leaves_by_user[user] = []
                leaves_by_user[user].append((start, end))
        
        # Check each task with a due date against user leaves
        for _, issue in issues_df.iterrows():
            assignee = issue.get('Assignee')
            due_date = issue.get('Due Date')
            task_id = issue.get('Issue key', 'Unknown')
            
            if pd.notna(assignee) and pd.notna(due_date) and assignee in leaves_by_user:
                for leave_start, leave_end in leaves_by_user[assignee]:
                    # Check if due date falls within leave period
                    if leave_start <= due_date <= leave_end:
                        conflicts.append((task_id, assignee, due_date, leave_start, leave_end))
        
        if conflicts:
            integrity_results["warnings"].append(
                f"Found {len(conflicts)} tasks due when assignee is on leave")
            if len(conflicts) <= 5:  # Show limited examples
                for task_id, assignee, due, leave_s, leave_e in conflicts[:5]:
                    integrity_results["info"].append(
                        f"Task {task_id} assigned to {assignee} is due on {due.date()} during leave ({leave_s.date()} to {leave_e.date()})")
            integrity_results["recommendations"].append(
                "Review task assignments for users with planned leaves")
    
    # Check 5: Check for overdue unresolved issues
    if 'Status' in issues_df.columns and 'Due Date' in issues_df.columns:
        today = datetime.now().date()
        overdue = issues_df[(issues_df['Status'] != 'Done') & 
                         (~issues_df['Due Date'].isna()) & 
                         (issues_df['Due Date'].dt.date < today)]
        
        if len(overdue) > 0:
            integrity_results["warnings"].append(
                f"Found {len(overdue)} incomplete tasks that are past due date")
            integrity_results["recommendations"].append(
                "Review and update overdue tasks or adjust deadlines")
    
    return integrity_results

# ---------- Data Visualization Helper Functions ----------
def format_integrity_results(integrity_results):
    """
    Format integrity check results for display in Streamlit.
    
    Args:
        integrity_results: Dictionary with integrity check results
        
    Returns:
        Formatted markdown string for display
    """
    markdown = ""
    
    # Format errors (high priority)
    if integrity_results["errors"]:
        markdown += "### ❌ Critical Issues\n"
        for error in integrity_results["errors"]:
            markdown += f"- {error}\n"
        markdown += "\n"
    
    # Format warnings (medium priority)
    if integrity_results["warnings"]:
        markdown += "### ⚠️ Warnings\n"
        for warning in integrity_results["warnings"]:
            markdown += f"- {warning}\n"
        markdown += "\n"
    
    # Format info (details)
    if integrity_results["info"]:
        markdown += "### ℹ️ Details\n"
        for info in integrity_results["info"]:
            markdown += f"- {info}\n"
        markdown += "\n"
    
    # Format recommendations
    if integrity_results["recommendations"]:
        markdown += "### 💡 Recommendations\n"
        for rec in integrity_results["recommendations"]:
            markdown += f"- {rec}\n"
    
    # If no issues found
    if not integrity_results["errors"] and not integrity_results["warnings"]:
        markdown += "### ✅ All checks passed\n"
        markdown += "- No data integrity issues detected\n"
    
    return markdown
