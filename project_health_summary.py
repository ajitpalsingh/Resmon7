import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import numpy as np
import json
import os

# Import OpenAI for AI-powered summaries
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_project_health_summary(issues_df, worklogs_df, skills_df, leaves_df):
    """
    Generate AI-powered project health summary based on JIRA data.
    
    Args:
        issues_df: DataFrame containing issue data
        worklogs_df: DataFrame containing worklog data
        skills_df: DataFrame containing skills data
        leaves_df: DataFrame containing leave/non-availability data
    """
    st.header("🔍 AI-Generated Project Health Summary")
    
    # Create tabs for different summary views
    tab1, tab2, tab3 = st.tabs(["Overall Health", "Risk Analysis", "Team Performance"])
    
    with tab1:
        generate_overall_health_summary(issues_df, worklogs_df, leaves_df)
        
    with tab2:
        generate_risk_analysis(issues_df)
        
    with tab3:
        generate_team_performance_summary(issues_df, worklogs_df, skills_df)


def generate_overall_health_summary(issues_df, worklogs_df, leaves_df):
    """
    Generate overall project health summary using existing data and AI
    """
    st.subheader("Project Health Overview")
    
    # Calculate key metrics
    total_issues = len(issues_df)
    completed_issues = len(issues_df[issues_df['Status'].isin(['Done', 'Closed', 'Resolved'])])
    in_progress = len(issues_df[issues_df['Status'].isin(['In Progress', 'In Review'])])
    not_started = len(issues_df[issues_df['Status'].isin(['To Do', 'Open', 'Backlog'])])
    
    # Calculate completion percentage
    completion_percentage = (completed_issues / total_issues * 100) if total_issues > 0 else 0
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tasks", total_issues)
    col2.metric("Completed", completed_issues)
    col3.metric("In Progress", in_progress)
    col4.metric("Not Started", not_started)
    
    # Progress bar
    st.progress(float(completion_percentage/100))
    st.caption(f"Overall Completion: {completion_percentage:.1f}%")
    
    # Create a dataset for AI summary
    # Count tasks by status
    status_counts = issues_df['Status'].value_counts().to_dict()
    
    # Count tasks by priority
    priority_counts = issues_df['Priority'].value_counts().to_dict()
    
    # Calculate upcoming deadlines
    today = pd.Timestamp.today()
    upcoming_deadlines = issues_df[
        (issues_df['Due Date'] >= today) & 
        (issues_df['Due Date'] <= today + pd.Timedelta(days=14)) &
        (~issues_df['Status'].isin(['Done', 'Closed', 'Resolved']))
    ]
    
    # Calculate overdue tasks
    overdue_tasks = issues_df[
        (issues_df['Due Date'] < today) & 
        (~issues_df['Status'].isin(['Done', 'Closed', 'Resolved']))
    ]
    
    # Calculate resource availability
    today_leaves = leaves_df[
        (leaves_df['Start Date'] <= today) & 
        (leaves_df['End Date'] >= today)
    ]
    num_resources_unavailable = len(today_leaves['Resource'].unique())
    
    # Prepare AI prompt
    summary_data = {
        "total_tasks": total_issues,
        "completed_tasks": completed_issues,
        "completion_percentage": completion_percentage,
        "in_progress": in_progress,
        "not_started": not_started,
        "status_breakdown": status_counts,
        "priority_breakdown": priority_counts,
        "upcoming_deadlines": len(upcoming_deadlines),
        "overdue_tasks": len(overdue_tasks),
        "resources_unavailable": num_resources_unavailable
    }
    
    # Get AI-generated summary
    ai_summary = get_ai_health_summary(summary_data)
    
    # Display the AI-generated summary
    st.markdown("### AI-Generated Summary")
    st.markdown(ai_summary)
    
    # Show some supporting charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Create a pie chart for task status
        status_df = pd.DataFrame(list(status_counts.items()), columns=['Status', 'Count'])
        fig = px.pie(status_df, values='Count', names='Status', title='Task Status Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Create a bar chart for task priority
        priority_df = pd.DataFrame(list(priority_counts.items()), columns=['Priority', 'Count'])
        fig = px.bar(priority_df, x='Priority', y='Count', title='Tasks by Priority')
        st.plotly_chart(fig, use_container_width=True)


def generate_risk_analysis(issues_df):
    """
    Generate risk analysis based on due dates, priorities, and risk fields
    """
    st.subheader("Project Risk Analysis")
    
    # Check if we have risk data
    has_risk_data = 'Risk Level' in issues_df.columns and not issues_df['Risk Level'].isna().all()
    
    if has_risk_data:
        # Filter for risks
        risks_df = issues_df[~issues_df['Risk Level'].isna()]
        st.write(f"Found {len(risks_df)} documented risks in the project")
        
        # Display risks in a table
        if not risks_df.empty:
            st.markdown("### Documented Risks")
            # Check which risk columns are available
            risk_columns = []
            for col in ['Issue Key', 'Risk Level', 'Risk Description', 'Risk Due Date', 'Risk Owner']:
                if col in risks_df.columns:
                    risk_columns.append(col)
            
            if risk_columns:
                risk_table = risks_df[risk_columns]
                st.dataframe(risk_table, use_container_width=True)
            else:
                st.info("Risk data available but no detailed risk columns found.")
    else:
        st.info("No explicit risk data found in the dataset. Analyzing implicit risks...")
    
    # Calculate deadline risks
    today = pd.Timestamp.today()
    
    # Overdue tasks are high risk
    overdue_tasks = issues_df[
        (issues_df['Due Date'] < today) & 
        (~issues_df['Status'].isin(['Done', 'Closed', 'Resolved']))
    ]
    
    # Tasks due soon are medium risk
    soon_due_tasks = issues_df[
        (issues_df['Due Date'] >= today) & 
        (issues_df['Due Date'] <= today + pd.Timedelta(days=7)) &
        (~issues_df['Status'].isin(['Done', 'Closed', 'Resolved']))
    ]
    
    # High priority tasks are elevated risk
    high_priority_tasks = issues_df[
        (issues_df['Priority'].isin(['Highest', 'High'])) &
        (~issues_df['Status'].isin(['Done', 'Closed', 'Resolved']))
    ]
    
    # Calculate risk metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Overdue Tasks", len(overdue_tasks), delta=len(overdue_tasks), delta_color="inverse")
    col2.metric("Due This Week", len(soon_due_tasks))
    col3.metric("High Priority Open", len(high_priority_tasks))
    
    # Create risk breakdown
    st.markdown("### Tasks at Risk")
    
    # Create risk score
    def calculate_risk_score(row):
        score = 0
        # Overdue tasks get highest risk
        if row['Due Date'] < today and row['Status'] not in ['Done', 'Closed', 'Resolved']:
            score += 30
        # Due soon tasks get medium risk
        elif row['Due Date'] <= today + pd.Timedelta(days=7) and row['Status'] not in ['Done', 'Closed', 'Resolved']:
            score += 15
        # High priority adds risk
        if row['Priority'] in ['Highest', 'High']:
            score += 10
        # Story points add risk proportionally
        if pd.notna(row['Story Points']):
            score += min(int(row['Story Points']), 10)  # Cap at 10 points
        return score
    
    # Apply risk scoring
    issues_df['Risk Score'] = issues_df.apply(calculate_risk_score, axis=1)
    
    # Get high risk tasks
    high_risk_tasks = issues_df[issues_df['Risk Score'] > 20].sort_values('Risk Score', ascending=False)
    
    if not high_risk_tasks.empty:
        risk_display = high_risk_tasks[['Issue Key', 'Summary', 'Status', 'Priority', 'Due Date', 'Assignee', 'Risk Score']]
        st.dataframe(risk_display, use_container_width=True)
        
        # Generate AI risk assessment
        risk_data = {
            "overdue_count": len(overdue_tasks),
            "due_soon_count": len(soon_due_tasks),
            "high_priority_count": len(high_priority_tasks),
            "high_risk_count": len(high_risk_tasks)
        }
        
        risk_summary = get_ai_risk_assessment(risk_data)
        st.markdown("### AI Risk Assessment")
        st.markdown(risk_summary)
    else:
        st.success("No high-risk tasks identified in the project. Well done!")


def generate_team_performance_summary(issues_df, worklogs_df, skills_df):
    """
    Generate team performance summary
    """
    st.subheader("Team Performance")
    
    # Check if we have the required data
    if worklogs_df is None or worklogs_df.empty:
        st.warning("No worklog data available for team performance analysis")
        return
    
    # Calculate key metrics by resource
    if 'Resource' in worklogs_df.columns and 'Time Spent (hrs)' in worklogs_df.columns:
        # Group by resource and sum time spent
        resource_hours = worklogs_df.groupby('Resource')['Time Spent (hrs)'].sum().reset_index()
        resource_hours = resource_hours.sort_values('Time Spent (hrs)', ascending=False)
        
        # Create a bar chart of hours by resource
        fig = px.bar(
            resource_hours, 
            x='Resource', 
            y='Time Spent (hrs)', 
            title='Hours Logged by Team Member',
            color='Time Spent (hrs)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate tasks completed by assignee
        if 'Assignee' in issues_df.columns and 'Status' in issues_df.columns:
            completed_by_assignee = issues_df[issues_df['Status'].isin(['Done', 'Closed', 'Resolved'])]
            
            if not completed_by_assignee.empty:
                completed_count = completed_by_assignee['Assignee'].value_counts().reset_index()
                completed_count.columns = ['Assignee', 'Completed Tasks']
                
                st.markdown("### Tasks Completed by Assignee")
                st.dataframe(completed_count, use_container_width=True)
                
                # Merge with resource hours for efficiency calculation
                performance_df = pd.merge(
                    resource_hours, 
                    completed_count, 
                    left_on='Resource', 
                    right_on='Assignee', 
                    how='left'
                )
                
                # Fill NA values for resources with no completed tasks
                performance_df['Completed Tasks'] = performance_df['Completed Tasks'].fillna(0)
                
                # Calculate efficiency (tasks per 10 hours)
                performance_df['Efficiency'] = (performance_df['Completed Tasks'] / performance_df['Time Spent (hrs)']) * 10
                performance_df['Efficiency'] = performance_df['Efficiency'].round(2)
                
                # Display efficiency metrics
                st.markdown("### Team Efficiency (Tasks completed per 10 hours)")
                efficiency_df = performance_df[['Resource', 'Time Spent (hrs)', 'Completed Tasks', 'Efficiency']]
                st.dataframe(efficiency_df, use_container_width=True)
                
                # Generate AI team performance assessment
                team_data = {
                    "resource_hours": resource_hours.to_dict('records'),
                    "completed_tasks": completed_count.to_dict('records') if not completed_count.empty else [],
                    "efficiency": efficiency_df.to_dict('records') if not efficiency_df.empty else []
                }
                
                team_summary = get_ai_team_assessment(team_data)
                st.markdown("### AI Team Assessment")
                st.markdown(team_summary)
        else:
            st.warning("Missing Assignee or Status data for team performance analysis")
    else:
        st.warning("Missing Resource or Time Spent data for team performance analysis")


def get_ai_health_summary(summary_data):
    """
    Generate an AI-powered summary of project health
    
    Args:
        summary_data: Dictionary containing project metrics
        
    Returns:
        String containing AI-generated project health summary
    """
    try:
        # Format the prompt with the data
        prompt = f"""
        You are a Project Management AI Assistant. Based on the following JIRA project data, create a concise 
        project health summary that highlights key metrics, trends, and potential issues. 
        Be brief but insightful, focusing on actionable information.
        
        Project data:
        - Total tasks: {summary_data['total_tasks']}
        - Completed tasks: {summary_data['completed_tasks']} ({summary_data['completion_percentage']:.1f}%)
        - Tasks in progress: {summary_data['in_progress']}
        - Tasks not started: {summary_data['not_started']}
        - Status breakdown: {summary_data['status_breakdown']}
        - Priority breakdown: {summary_data['priority_breakdown']}
        - Upcoming deadline count (next 14 days): {summary_data['upcoming_deadlines']}
        - Overdue tasks: {summary_data['overdue_tasks']}
        - Resources currently unavailable: {summary_data['resources_unavailable']}
        
        Provide a summary in 3-5 bullet points that assesses overall health, highlights risks,
        and suggests actions to improve project health.
        """
        
        # Call OpenAI API
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional project management assistant that provides concise, data-driven insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=350
        )
        
        # Return the summary
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error generating AI summary: {e}")
        return "*Unable to generate AI summary. Please check your OpenAI API key and try again.*"


def get_ai_risk_assessment(risk_data):
    """
    Generate an AI-powered risk assessment
    
    Args:
        risk_data: Dictionary containing risk metrics
        
    Returns:
        String containing AI-generated risk assessment
    """
    try:
        # Format the prompt with the risk data
        prompt = f"""
        You are a Project Risk Analysis AI. Based on the following project risk data, create a concise 
        risk assessment that highlights key risk areas and suggests mitigation strategies.
        
        Risk data:
        - Overdue tasks: {risk_data['overdue_count']}
        - Tasks due in the next 7 days: {risk_data['due_soon_count']}
        - High priority open tasks: {risk_data['high_priority_count']}
        - High risk tasks (composite score): {risk_data['high_risk_count']}
        
        Provide a risk assessment in 3-4 bullet points that evaluates overall project risk,
        identifies critical risk factors, and suggests specific mitigation actions.
        """
        
        # Call OpenAI API
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional risk management assistant that provides concise, actionable risk assessments."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        # Return the risk assessment
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error generating AI risk assessment: {e}")
        return "*Unable to generate AI risk assessment. Please check your OpenAI API key and try again.*"


def get_ai_team_assessment(team_data):
    """
    Generate an AI-powered team performance assessment
    
    Args:
        team_data: Dictionary containing team performance metrics
        
    Returns:
        String containing AI-generated team assessment
    """
    try:
        # Format the prompt with the team data
        prompt = f"""
        You are a Team Performance Analysis AI. Based on the following team data, create a concise 
        assessment that highlights team performance, identifies top performers, and suggests improvements.
        
        Team data:
        - Hours logged by resource: {team_data['resource_hours']}
        - Completed tasks by assignee: {team_data['completed_tasks']}
        - Team efficiency metrics: {team_data['efficiency']}
        
        Provide a team assessment in 3-4 bullet points that evaluates overall team performance,
        recognizes high performers, identifies potential areas for improvement, and suggests specific actions.
        """
        
        # Call OpenAI API
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional team management assistant that provides concise, data-driven performance assessments."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        # Return the team assessment
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error generating AI team assessment: {e}")
        return "*Unable to generate AI team assessment. Please check your OpenAI API key and try again.*"
