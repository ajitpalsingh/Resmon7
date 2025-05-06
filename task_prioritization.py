import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import datetime
import os

# Import OpenAI for AI-powered prioritization
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def ai_driven_task_prioritization(issues_df, worklogs_df, skills_df, leaves_df):
    """
    AI-driven task prioritization based on multiple factors including deadlines,
    dependencies, resource availability, and project priorities.
    
    Args:
        issues_df: DataFrame containing issue data
        worklogs_df: DataFrame containing worklog data
        skills_df: DataFrame containing skills data
        leaves_df: DataFrame containing leave/non-availability data
    """
    st.header("🔄 AI-Driven Task Prioritization")
    
    # Create tabs for different prioritization views
    tab1, tab2, tab3 = st.tabs(["Priority Dashboard", "Team Tasks", "Prioritization Settings"])
    
    with tab1:
        display_priority_dashboard(issues_df, worklogs_df, leaves_df)
    
    with tab2:
        display_team_prioritization(issues_df, leaves_df)
    
    with tab3:
        prioritization_settings()


def display_priority_dashboard(issues_df, worklogs_df, leaves_df):
    """
    Display the main prioritization dashboard
    """
    st.subheader("Task Priority Dashboard")
    
    # Only consider open tasks for prioritization
    open_issues = issues_df[~issues_df['Status'].isin(['Done', 'Closed', 'Resolved'])]
    
    if open_issues.empty:
        st.info("No open tasks to prioritize.")
        return
    
    # Calculate enhanced priority scores
    prioritized_tasks = calculate_priority_scores(open_issues, worklogs_df, leaves_df)
    
    # Show prioritized tasks
    st.markdown("### Prioritized Tasks")
    
    # Add color coding based on priority score
    def color_priority(val):
        if val >= 80:
            return 'background-color: rgba(255, 0, 0, 0.2)'
        elif val >= 60:
            return 'background-color: rgba(255, 165, 0, 0.2)'
        elif val >= 40:
            return 'background-color: rgba(255, 255, 0, 0.2)'
        else:
            return ''
    
    # Display prioritized tasks with styling
    display_df = prioritized_tasks[['Issue Key', 'Summary', 'Due Date', 'Assignee', 'Priority', 'AI Priority Score']]
    # Use Styler.map instead of applymap (which is deprecated)
    styled_df = display_df.style.map(color_priority, subset=['AI Priority Score'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Priority distribution chart
    st.markdown("### Priority Score Distribution")
    fig = px.histogram(
        prioritized_tasks, 
        x="AI Priority Score", 
        nbins=10, 
        color_discrete_sequence=['#3366cc'],
        title="Distribution of Priority Scores"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Get AI-driven recommendations for top priority tasks
    with st.expander("AI Recommendations for High Priority Tasks"):
        top_tasks = prioritized_tasks.head(5)
        recommendations = get_ai_prioritization_recommendations(top_tasks)
        st.markdown(recommendations)


def display_team_prioritization(issues_df, leaves_df):
    """
    Display task prioritization by team member
    """
    st.subheader("Team Task Prioritization")
    
    # Filter for open tasks with assignees
    assigned_tasks = issues_df[
        (~issues_df['Status'].isin(['Done', 'Closed', 'Resolved'])) &
        (~issues_df['Assignee'].isna())
    ]
    
    if assigned_tasks.empty:
        st.info("No assigned open tasks found.")
        return
    
    # Get unique assignees
    assignees = assigned_tasks['Assignee'].unique()
    
    # Select a team member
    selected_assignee = st.selectbox("Select Team Member", options=assignees)
    
    if selected_assignee:
        # Get tasks for the selected assignee
        assignee_tasks = assigned_tasks[assigned_tasks['Assignee'] == selected_assignee]
        
        # Check for leaves/unavailability
        today = pd.Timestamp.today()
        assignee_leaves = leaves_df[
            (leaves_df['Resource'] == selected_assignee) &
            (leaves_df['End Date'] >= today)
        ]
        
        if not assignee_leaves.empty:
            st.warning(f"⚠️ {selected_assignee} has upcoming leave/unavailability:")
            leave_info = assignee_leaves[['Start Date', 'End Date', 'Reason']]
            st.dataframe(leave_info, use_container_width=True)
        
        # Calculate priority scores for assignee's tasks
        prioritized_assignee_tasks = calculate_priority_scores(assignee_tasks, None, leaves_df)
        
        # Display prioritized tasks for the assignee
        st.markdown(f"### Prioritized Tasks for {selected_assignee}")
        display_df = prioritized_assignee_tasks[['Issue Key', 'Summary', 'Due Date', 'Priority', 'AI Priority Score']]
        st.dataframe(display_df.sort_values('AI Priority Score', ascending=False), use_container_width=True)
        
        # Get AI-driven personalized recommendations
        recommendations = get_ai_personal_recommendations(selected_assignee, prioritized_assignee_tasks, assignee_leaves)
        st.markdown("### AI-Generated Recommendations")
        st.markdown(recommendations)
        
        # Visualize timeline of tasks for this assignee
        st.markdown("### Task Timeline")
        create_assignee_task_timeline(prioritized_assignee_tasks, assignee_leaves)


def create_assignee_task_timeline(tasks_df, leaves_df):
    """
    Create a timeline visualization of tasks and leaves for an assignee
    """
    timeline_data = []
    
    # Add task data
    for _, task in tasks_df.iterrows():
        # Use start date or estimate 14 days before due date if not available
        if pd.isna(task['Start Date']) and not pd.isna(task['Due Date']):
            start_date = task['Due Date'] - pd.Timedelta(days=14)
        else:
            start_date = task['Start Date']
        
        # Skip tasks with missing dates
        if pd.isna(start_date) or pd.isna(task['Due Date']):
            continue
        
        # Add to timeline data
        timeline_data.append({
            'Task': f"{task['Issue Key']}: {task['Summary'][:20]}...",
            'Start': start_date,
            'Finish': task['Due Date'],
            'Type': 'Task',
            'Priority Score': task['AI Priority Score']
        })
    
    # Add leave data
    for _, leave in leaves_df.iterrows():
        timeline_data.append({
            'Task': f"Unavailable: {leave['Reason']}",
            'Start': leave['Start Date'],
            'Finish': leave['End Date'],
            'Type': 'Leave',
            'Priority Score': 0
        })
    
    # Create timeline visualization if we have data
    if timeline_data:
        timeline_df = pd.DataFrame(timeline_data)
        
        # Create color map
        color_map = {
            'Task': 'rgba(55, 128, 191, 0.8)',
            'Leave': 'rgba(200, 0, 0, 0.7)'
        }
        
        # Create the Gantt chart
        fig = px.timeline(
            timeline_df, 
            x_start="Start", 
            x_end="Finish", 
            y="Task",
            color="Type",
            color_discrete_map=color_map,
            title="Task and Unavailability Timeline"
        )
        
        # Add a vertical line for today using shapes instead of add_vline
        today = pd.Timestamp.today()
        # Add vertical line as a shape
        fig.add_shape(
            type="line",
            x0=today,
            x1=today,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dash"),
        )
        
        # Add the "Today" annotation
        fig.add_annotation(
            x=today,
            y=1.05,
            yref="paper",
            text="Today",
            showarrow=False,
            font=dict(color="red")
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Insufficient date information for timeline visualization.")


def prioritization_settings():
    """
    Settings for prioritization algorithm
    """
    st.subheader("Prioritization Settings")
    
    # Create weight sliders for different factors
    st.write("Adjust weights for prioritization factors:")
    
    # Use session state to persist settings
    if 'priority_weights' not in st.session_state:
        st.session_state.priority_weights = {
            'due_date': 30,
            'priority': 25,
            'dependencies': 15,
            'story_points': 15,
            'leave_conflict': 15
        }
    
    # Create sliders for each weight
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.priority_weights['due_date'] = st.slider(
            "Due Date Proximity Weight", 0, 50, st.session_state.priority_weights['due_date'],
            help="How heavily to weigh task due dates in priority calculation"
        )
        
        st.session_state.priority_weights['priority'] = st.slider(
            "JIRA Priority Weight", 0, 50, st.session_state.priority_weights['priority'],
            help="How heavily to weigh the original JIRA priority field"
        )
        
        st.session_state.priority_weights['dependencies'] = st.slider(
            "Dependencies Weight", 0, 50, st.session_state.priority_weights['dependencies'],
            help="How heavily to weigh task dependencies"
        )
    
    with col2:
        st.session_state.priority_weights['story_points'] = st.slider(
            "Story Points Weight", 0, 50, st.session_state.priority_weights['story_points'],
            help="How heavily to weigh the task size/complexity"
        )
        
        st.session_state.priority_weights['leave_conflict'] = st.slider(
            "Leave Conflict Weight", 0, 50, st.session_state.priority_weights['leave_conflict'],
            help="How heavily to weigh conflicts with team member leaves"
        )
    
    # Optional: Reset button
    if st.button("Reset to Default Weights"):
        st.session_state.priority_weights = {
            'due_date': 30,
            'priority': 25,
            'dependencies': 15,
            'story_points': 15,
            'leave_conflict': 15
        }
        st.rerun()


def calculate_priority_scores(issues_df, worklogs_df, leaves_df):
    """
    Calculate priority scores based on multiple factors
    
    Args:
        issues_df: DataFrame of issues
        worklogs_df: DataFrame of worklogs
        leaves_df: DataFrame of non-availability
        
    Returns:
        DataFrame with priority scores added
    """
    # Create a copy to avoid modifying the original
    df = issues_df.copy()
    
    # Get weights from session state or use defaults
    weights = st.session_state.get('priority_weights', {
        'due_date': 30,
        'priority': 25,
        'dependencies': 15,
        'story_points': 15,
        'leave_conflict': 15
    })
    
    # 1. Due Date Score (0-100)
    today = pd.Timestamp.today()
    df['Due Date Score'] = 0
    
    # For tasks with due dates
    mask = ~df['Due Date'].isna()
    if mask.any():
        # Calculate days until due
        df.loc[mask, 'Days Until Due'] = (df.loc[mask, 'Due Date'] - today).dt.days
        
        # Overdue tasks get max score
        df.loc[df['Days Until Due'] < 0, 'Due Date Score'] = 100
        
        # Tasks due within 7 days get high scores
        week_mask = (df['Days Until Due'] >= 0) & (df['Days Until Due'] <= 7)
        df.loc[week_mask, 'Due Date Score'] = 100 - (df.loc[week_mask, 'Days Until Due'] * 10)
        
        # Tasks due within 14 days get medium scores
        two_week_mask = (df['Days Until Due'] > 7) & (df['Days Until Due'] <= 14)
        df.loc[two_week_mask, 'Due Date Score'] = 50 - ((df.loc[two_week_mask, 'Days Until Due'] - 7) * 5)
        
        # Tasks due beyond 14 days get lower scores based on logarithmic scale
        far_mask = df['Days Until Due'] > 14
        if far_mask.any():
            # Use logarithmic scale to avoid too low values for far-future tasks
            # Cast to int to avoid FutureWarning
            df.loc[far_mask, 'Due Date Score'] = (20 - np.log10(df.loc[far_mask, 'Days Until Due'] - 13) * 10).astype(int)
            df.loc[df['Due Date Score'] < 0, 'Due Date Score'] = 0
    
    # 2. Priority Score (0-100)
    priority_map = {
        'Highest': 100,
        'High': 80,
        'Medium': 60,
        'Low': 40,
        'Lowest': 20
    }
    df['Priority Score'] = df['Priority'].map(priority_map).fillna(50)
    
    # 3. Story Points Score (0-100)
    df['Story Points Score'] = 0
    mask = ~df['Story Points'].isna()
    if mask.any():
        # Normalize to 0-100 scale, with higher points getting higher priority
        max_points = df.loc[mask, 'Story Points'].max()
        if max_points > 0:
            # Create intermediate float values, then convert explicitly to int to avoid dtype warning
            float_scores = (df.loc[mask, 'Story Points'] / max_points) * 100
            df.loc[mask, 'Story Points Score'] = float_scores.astype(int)
    
    # 4. Leave Conflict Score (0-100)
    df['Leave Conflict Score'] = 0
    
    if leaves_df is not None and not leaves_df.empty and 'Assignee' in df.columns:
        for idx, task in df.iterrows():
            if pd.isna(task['Assignee']) or pd.isna(task['Due Date']):
                continue
                
            # Check if assignee has leaves near the task due date
            assignee_leaves = leaves_df[leaves_df['Resource'] == task['Assignee']]
            
            for _, leave in assignee_leaves.iterrows():
                # Skip leaves without proper dates
                if pd.isna(leave['Start Date']) or pd.isna(leave['End Date']):
                    continue
                    
                # Calculate how close the due date is to the leave
                if task['Due Date'] >= leave['Start Date'] and task['Due Date'] <= leave['End Date']:
                    # Due during leave - highest priority
                    df.loc[idx, 'Leave Conflict Score'] = 100
                    break
                elif task['Due Date'] < leave['Start Date']:
                    # Due before leave starts
                    days_before = (leave['Start Date'] - task['Due Date']).days
                    if days_before <= 3:
                        df.loc[idx, 'Leave Conflict Score'] = 80
                    elif days_before <= 7:
                        df.loc[idx, 'Leave Conflict Score'] = 60
                elif task['Due Date'] > leave['End Date']:
                    # Due after leave ends
                    days_after = (task['Due Date'] - leave['End Date']).days
                    if days_after <= 3:
                        df.loc[idx, 'Leave Conflict Score'] = 70
                    elif days_after <= 7:
                        df.loc[idx, 'Leave Conflict Score'] = 50
    
    # 5. Calculate composite score (weighted average)
    df['AI Priority Score'] = (
        df['Due Date Score'] * weights['due_date'] / 100 +
        df['Priority Score'] * weights['priority'] / 100 +
        df['Story Points Score'] * weights['story_points'] / 100 +
        df['Leave Conflict Score'] * weights['leave_conflict'] / 100
    ) / (weights['due_date'] + weights['priority'] + weights['story_points'] + weights['leave_conflict']) * 100
    
    # Round scores for display
    df['AI Priority Score'] = df['AI Priority Score'].round(1)
    
    # Sort by priority score
    return df.sort_values('AI Priority Score', ascending=False)


def get_ai_prioritization_recommendations(top_tasks):
    """
    Generate AI recommendations for top priority tasks
    
    Args:
        top_tasks: DataFrame of top priority tasks
        
    Returns:
        String with AI-generated recommendations
    """
    try:
        # Format task data for the prompt
        task_data = ""
        for i, (_, task) in enumerate(top_tasks.iterrows(), 1):
            due_date = task['Due Date'].strftime('%Y-%m-%d') if not pd.isna(task['Due Date']) else 'No due date'
            task_data += f"Task {i}: {task['Issue Key']} - {task['Summary']}\n"
            task_data += f"   Priority: {task['Priority']}, Due Date: {due_date}, Assignee: {task['Assignee']}\n"
            task_data += f"   AI Priority Score: {task['AI Priority Score']}\n\n"
        
        # Create the prompt
        prompt = f"""
        You are a Project Management AI Assistant specializing in task prioritization.
        Based on the following high-priority tasks, provide specific recommendations
        for addressing them effectively. Focus on sequencing, potential blockers, and strategies
        for completing them on time.
        
        HIGH PRIORITY TASKS:
        {task_data}
        
        Provide 3-5 bullet point recommendations that are specific to these tasks.
        Focus on actionable advice for task sequencing, resource allocation, and risk mitigation.
        """
        
        # Call OpenAI API
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional project management assistant that specializes in task prioritization and resource optimization."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=400
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error generating AI recommendations: {e}")
        return "*Unable to generate AI recommendations. Please check your OpenAI API key and try again.*"


def get_ai_personal_recommendations(assignee, tasks_df, leaves_df):
    """
    Generate personalized AI recommendations for a team member
    
    Args:
        assignee: Team member name
        tasks_df: DataFrame of tasks assigned to the team member
        leaves_df: DataFrame of team member's leaves
        
    Returns:
        String with AI-generated personalized recommendations
    """
    try:
        # Format task data for the prompt
        task_data = ""
        for i, (_, task) in enumerate(tasks_df.head(5).iterrows(), 1):
            due_date = task['Due Date'].strftime('%Y-%m-%d') if not pd.isna(task['Due Date']) else 'No due date'
            task_data += f"Task {i}: {task['Issue Key']} - {task['Summary']}\n"
            task_data += f"   Priority: {task['Priority']}, Due Date: {due_date}\n"
            task_data += f"   AI Priority Score: {task['AI Priority Score']}\n\n"
        
        # Format leave data
        leave_data = ""
        for i, (_, leave) in enumerate(leaves_df.iterrows(), 1):
            start_date = leave['Start Date'].strftime('%Y-%m-%d') if not pd.isna(leave['Start Date']) else 'Unknown'
            end_date = leave['End Date'].strftime('%Y-%m-%d') if not pd.isna(leave['End Date']) else 'Unknown'
            leave_data += f"Leave {i}: {start_date} to {end_date} - {leave['Reason']}\n"
        
        if not leave_data:
            leave_data = "No upcoming leaves or unavailability periods.\n"
        
        # Create the prompt
        prompt = f"""
        You are a Personal Project Management Assistant for {assignee}. Based on their
        assigned tasks and upcoming unavailability periods, provide personalized
        recommendations to help them effectively manage their workload.
        
        ASSIGNED TASKS (Highest priority first):
        {task_data}
        
        UPCOMING UNAVAILABILITY:
        {leave_data}
        
        Provide 3-4 bullet point personalized recommendations for {assignee} that address:
        1. How to sequence their work effectively
        2. Preparation needed before any upcoming leaves
        3. Specific strategies for their highest priority tasks
        4. Any potential risks or conflicts to be aware of
        
        Make recommendations specific to {assignee}'s situation and task list.
        """
        
        # Call OpenAI API
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional personal project management assistant that provides personalized, actionable advice."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=400
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error generating personalized AI recommendations: {e}")
        return "*Unable to generate personalized AI recommendations. Please check your OpenAI API key and try again.*"
