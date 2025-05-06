import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Import OpenAI for AI-powered estimations
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def effort_estimation_refinement(issues_df, worklogs_df, skills_df, leaves_df):
    """
    AI-powered effort estimation refinement based on historical data
    
    Args:
        issues_df: DataFrame containing issue data
        worklogs_df: DataFrame containing worklog data
        skills_df: DataFrame containing skills data
        leaves_df: DataFrame containing leave/non-availability data
    """
    st.header("📈 Effort Estimation Refinement")
    
    # Create tabs for different estimation views
    tab1, tab2, tab3 = st.tabs(["Estimation Analysis", "Effort Predictor", "Team Velocity"])
    
    with tab1:
        estimation_analysis(issues_df, worklogs_df)
    
    with tab2:
        effort_predictor(issues_df, worklogs_df)
    
    with tab3:
        team_velocity(issues_df, worklogs_df, skills_df)


def estimation_analysis(issues_df, worklogs_df):
    """
    Analyze estimation accuracy based on historical data
    """
    st.subheader("Estimation Accuracy Analysis")
    
    # Check if we have the required data
    if worklogs_df is None or worklogs_df.empty:
        st.warning("No worklog data available for estimation analysis")
        return
    
    if issues_df is None or issues_df.empty:
        st.warning("No issue data available for estimation analysis")
        return
        
    # Only analyze completed tasks with both estimates and actuals
    completed_issues = issues_df[issues_df['Status'].isin(['Done', 'Closed', 'Resolved'])]
    
    if completed_issues.empty:
        st.info("No completed tasks found for analysis.")
        return
    
    # Calculate actual hours from worklogs
    task_hours = worklogs_df.groupby('Issue Key')['Time Spent (hrs)'].sum().reset_index()
    
    # Merge with task data
    estimation_df = pd.merge(
        completed_issues,
        task_hours,
        on='Issue Key',
        how='inner'
    )
    
    # Filter for tasks with original estimates
    estimation_df = estimation_df[~estimation_df['Original Estimate (days)'].isna()]
    
    if estimation_df.empty:
        st.info("No completed tasks with both estimates and actuals found.")
        return
    
    # Convert days to hours (assuming 8-hour days)
    estimation_df['Estimated Hours'] = estimation_df['Original Estimate (days)'] * 8
    
    # Calculate estimation accuracy
    estimation_df['Estimation Difference (hrs)'] = estimation_df['Time Spent (hrs)'] - estimation_df['Estimated Hours']
    estimation_df['Estimation Ratio'] = estimation_df['Time Spent (hrs)'] / estimation_df['Estimated Hours']
    estimation_df['Accuracy %'] = 100 - abs((estimation_df['Estimation Ratio'] - 1) * 100)
    estimation_df['Accuracy %'] = estimation_df['Accuracy %'].clip(0, 100).round(1)
    
    # Display overall accuracy metrics
    average_accuracy = estimation_df['Accuracy %'].mean()
    underestimated = len(estimation_df[estimation_df['Estimation Difference (hrs)'] > 0])
    overestimated = len(estimation_df[estimation_df['Estimation Difference (hrs)'] < 0])
    accurate = len(estimation_df[abs(estimation_df['Estimation Difference (hrs)']) <= 2])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Accuracy", f"{average_accuracy:.1f}%")
    col2.metric("Underestimated", f"{underestimated} tasks")
    col3.metric("Overestimated", f"{overestimated} tasks")
    col4.metric("Accurately Estimated", f"{accurate} tasks")
    
    # Display estimation accuracy chart
    st.markdown("### Estimation vs. Actual Hours")
    fig = px.scatter(
        estimation_df,
        x="Estimated Hours",
        y="Time Spent (hrs)",
        color="Accuracy %",
        size="Story Points",
        hover_name="Issue Key",
        hover_data=["Summary", "Estimation Difference (hrs)", "Accuracy %"],
        color_continuous_scale="RdYlGn",
        title="Estimation Accuracy Analysis"
    )
    
    # Add reference line (perfect estimation)
    max_val = max(estimation_df['Estimated Hours'].max(), estimation_df['Time Spent (hrs)'].max())
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            line=dict(color='rgba(0,0,0,0.5)', dash='dash'),
            name='Perfect Estimation'
        )
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show estimation accuracy by issue type
    st.markdown("### Estimation Accuracy by Issue Type")
    type_accuracy = estimation_df.groupby('Issue Type')['Accuracy %'].mean().reset_index()
    type_count = estimation_df.groupby('Issue Type').size().reset_index(name='Count')
    type_stats = pd.merge(type_accuracy, type_count, on='Issue Type')
    type_stats = type_stats.sort_values('Accuracy %', ascending=False)
    
    fig = px.bar(
        type_stats,
        x="Issue Type",
        y="Accuracy %",
        color="Accuracy %",
        text="Count",
        color_continuous_scale="RdYlGn",
        title="Average Estimation Accuracy by Issue Type"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Get AI-generated insights
    ai_insights = get_ai_estimation_insights(estimation_df)
    st.markdown("### AI-Generated Estimation Insights")
    st.markdown(ai_insights)


def effort_predictor(issues_df, worklogs_df):
    """
    AI-powered effort prediction for new tasks
    """
    st.subheader("AI Effort Predictor")
    
    # First, train the prediction model based on historical data
    model_data = prepare_estimation_model_data(issues_df, worklogs_df)
    
    if model_data is None or model_data.empty:
        st.warning("Insufficient historical data for effort prediction")
        return
    
    # Form for entering new task details
    st.markdown("### Predict Effort for New Task")
    
    col1, col2 = st.columns(2)
    
    with col1:
        issue_type = st.selectbox(
            "Issue Type",
            options=sorted(issues_df['Issue Type'].unique())
        )
        
        theme = st.selectbox(
            "Theme",
            options=[None] + sorted(issues_df['Theme'].dropna().unique().tolist())
        )
        
        priority = st.selectbox(
            "Priority",
            options=sorted(issues_df['Priority'].unique())
        )
    
    with col2:
        story_points = st.number_input(
            "Story Points",
            min_value=0,
            max_value=20,
            value=5
        )
        
        assignee = st.selectbox(
            "Assignee",
            options=[None] + sorted(issues_df['Assignee'].dropna().unique().tolist())
        )
    
    task_summary = st.text_area("Task Summary", "")
    
    if st.button("Predict Effort"):
        if not task_summary:
            st.error("Please enter a task summary")
        else:
            # Create task data for prediction
            task_data = {
                'Issue Type': issue_type,
                'Theme': theme,
                'Priority': priority,
                'Story Points': story_points,
                'Assignee': assignee,
                'Summary': task_summary
            }
            
            # Get AI prediction
            prediction = get_ai_effort_prediction(task_data, model_data)
            
            # Display prediction
            st.markdown("### Effort Prediction Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Estimated Hours", f"{prediction['estimated_hours']:.1f}")
                st.metric("Estimated Days", f"{prediction['estimated_days']:.2f}", help="Based on 8-hour workdays")
            
            with col2:
                st.metric("Confidence Level", f"{prediction['confidence']}%")
                st.metric("Adjustment Factor", f"{prediction['adjustment_factor']}x")
            
            st.markdown("### Prediction Rationale")
            st.markdown(prediction['explanation'])


def team_velocity(issues_df, worklogs_df, skills_df):
    """
    Analyze team velocity metrics
    """
    st.subheader("Team Velocity Analysis")
    
    # Check if we have sprint data
    if 'Sprint' not in issues_df.columns or issues_df['Sprint'].isna().all():
        st.warning("No sprint data available for velocity analysis")
        return
    
    # Calculate sprint velocity
    completed_in_sprint = issues_df[
        (issues_df['Status'].isin(['Done', 'Closed', 'Resolved'])) &
        (~issues_df['Sprint'].isna())
    ]
    
    if completed_in_sprint.empty:
        st.info("No completed sprint tasks found for velocity analysis.")
        return
    
    # Calculate points per sprint
    sprint_velocity = completed_in_sprint.groupby('Sprint')['Story Points'].sum().reset_index()
    sprint_velocity = sprint_velocity.sort_values('Sprint')
    
    # Create velocity chart
    st.markdown("### Sprint Velocity Trend")
    fig = px.bar(
        sprint_velocity,
        x="Sprint",
        y="Story Points",
        title="Story Points Completed per Sprint"
    )
    
    # Add trend line
    fig.add_trace(
        go.Scatter(
            x=sprint_velocity['Sprint'],
            y=sprint_velocity['Story Points'].rolling(window=3, min_periods=1).mean(),
            mode='lines',
            name='3-Sprint Moving Average',
            line=dict(color='red', width=2)
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate team member velocity if we have assignee data
    if 'Assignee' in completed_in_sprint.columns and not completed_in_sprint['Assignee'].isna().all():
        member_velocity = completed_in_sprint.groupby('Assignee')['Story Points'].sum().reset_index()
        member_count = completed_in_sprint.groupby('Assignee').size().reset_index(name='Task Count')
        member_stats = pd.merge(member_velocity, member_count, on='Assignee')
        member_stats['Avg Points per Task'] = (member_stats['Story Points'] / member_stats['Task Count']).round(1)
        member_stats = member_stats.sort_values('Story Points', ascending=False)
        
        st.markdown("### Team Member Velocity")
        st.dataframe(member_stats, use_container_width=True)
        
        # Calculate hours per story point if we have worklog data
        if worklogs_df is not None and not worklogs_df.empty:
            # Join worklogs with completed tasks
            task_hours = worklogs_df.groupby('Issue Key')['Time Spent (hrs)'].sum().reset_index()
            merged_data = pd.merge(
                completed_in_sprint[['Issue Key', 'Assignee', 'Story Points']],
                task_hours,
                on='Issue Key',
                how='inner'
            )
            
            # Calculate hours per point by assignee
            if not merged_data.empty and not merged_data['Story Points'].isna().all():
                # Filter out zero story points to avoid division by zero
                merged_data = merged_data[merged_data['Story Points'] > 0]
                merged_data['Hours per Point'] = merged_data['Time Spent (hrs)'] / merged_data['Story Points']
                
                assignee_efficiency = merged_data.groupby('Assignee')['Hours per Point'].mean().reset_index()
                assignee_efficiency['Hours per Point'] = assignee_efficiency['Hours per Point'].round(1)
                assignee_efficiency = assignee_efficiency.sort_values('Hours per Point')
                
                st.markdown("### Hours per Story Point by Team Member")
                fig = px.bar(
                    assignee_efficiency,
                    x="Assignee",
                    y="Hours per Point",
                    title="Efficiency: Average Hours per Story Point"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Get AI-generated velocity insights
                velocity_insights = get_ai_velocity_insights(sprint_velocity, assignee_efficiency)
                st.markdown("### AI-Generated Velocity Insights")
                st.markdown(velocity_insights)


def prepare_estimation_model_data(issues_df, worklogs_df):
    """
    Prepare historical data for estimation model
    
    Returns:
        DataFrame with features for estimation model
    """
    if issues_df is None or issues_df.empty or worklogs_df is None or worklogs_df.empty:
        return None
    
    # Only use completed tasks
    completed_issues = issues_df[issues_df['Status'].isin(['Done', 'Closed', 'Resolved'])]
    
    if completed_issues.empty:
        return None
    
    # Calculate actual hours from worklogs
    task_hours = worklogs_df.groupby('Issue Key')['Time Spent (hrs)'].sum().reset_index()
    
    # Merge with task data
    model_data = pd.merge(
        completed_issues,
        task_hours,
        on='Issue Key',
        how='inner'
    )
    
    return model_data


def get_ai_estimation_insights(estimation_df):
    """
    Generate AI insights about estimation patterns
    
    Args:
        estimation_df: DataFrame with estimation accuracy data
        
    Returns:
        String with AI-generated insights
    """
    try:
        # Prepare data for the prompt
        avg_accuracy = estimation_df['Accuracy %'].mean().round(1)
        underestimated = len(estimation_df[estimation_df['Estimation Difference (hrs)'] > 0])
        overestimated = len(estimation_df[estimation_df['Estimation Difference (hrs)'] < 0])
        accurate = len(estimation_df[abs(estimation_df['Estimation Difference (hrs)']) <= 2])
        total_tasks = len(estimation_df)
        
        # Calculate accuracy by issue type
        type_accuracy = estimation_df.groupby('Issue Type')['Accuracy %'].mean().round(1).to_dict()
        
        # Find patterns by story points
        if 'Story Points' in estimation_df.columns:
            points_accuracy = estimation_df.groupby('Story Points')['Accuracy %'].mean().round(1).to_dict()
        else:
            points_accuracy = {}
        
        # Format the prompt
        prompt = f"""
        You are an Estimation Analysis AI for project management. Based on the following data
        about estimation accuracy in completed tasks, provide insightful analysis and actionable
        recommendations for improving future estimates.
        
        ESTIMATION ACCURACY DATA:
        - Average estimation accuracy: {avg_accuracy}%
        - Underestimated tasks: {underestimated} out of {total_tasks} ({underestimated/total_tasks*100:.1f}%)
        - Overestimated tasks: {overestimated} out of {total_tasks} ({overestimated/total_tasks*100:.1f}%)
        - Accurately estimated tasks: {accurate} out of {total_tasks} ({accurate/total_tasks*100:.1f}%)
        
        ACCURACY BY ISSUE TYPE:
        {type_accuracy}
        
        ACCURACY BY STORY POINTS:
        {points_accuracy}
        
        Provide 4-5 bullet points that:
        1. Analyze the current estimation patterns and identify any biases
        2. Highlight which types of tasks are most accurately/inaccurately estimated
        3. Provide specific, actionable recommendations for improving estimation accuracy
        4. Suggest estimation adjustment factors for different types of tasks
        """
        
        # Call OpenAI API
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional project estimation analyst who provides concise, data-driven insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=400
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error generating AI insights: {e}")
        return "*Unable to generate AI insights. Please check your OpenAI API key and try again.*"


def get_ai_effort_prediction(task_data, model_data):
    """
    Generate AI prediction for task effort
    
    Args:
        task_data: Dictionary with new task information
        model_data: DataFrame with historical task data
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Find similar tasks in historical data
        similar_tasks = find_similar_tasks(task_data, model_data)
        
        if similar_tasks.empty:
            # Fall back to type-based estimation if no similar tasks
            similar_tasks = model_data[model_data['Issue Type'] == task_data['Issue Type']]
        
        # Calculate average hours for similar tasks
        avg_hours = similar_tasks['Time Spent (hrs)'].mean()
        
        # Prepare data for the prompt
        similar_task_data = []
        for _, task in similar_tasks.head(5).iterrows():
            similar_task_data.append({
                'Issue Key': task['Issue Key'],
                'Summary': task['Summary'],
                'Story Points': task['Story Points'] if 'Story Points' in task and not pd.isna(task['Story Points']) else 'N/A',
                'Hours': task['Time Spent (hrs)'],
                'Issue Type': task['Issue Type']
            })
        
        # Format the prompt
        prompt = f"""
        You are an AI Effort Estimation Assistant. Based on historical task data and the details of a new task,
        predict the effort required (in hours) to complete the new task.
        
        NEW TASK:
        - Summary: {task_data['Summary']}
        - Issue Type: {task_data['Issue Type']}
        - Theme: {task_data['Theme']}
        - Priority: {task_data['Priority']}
        - Story Points: {task_data['Story Points']}
        - Assignee: {task_data['Assignee']}
        
        SIMILAR HISTORICAL TASKS:
        {similar_task_data}
        
        Average Hours for Similar Tasks: {avg_hours:.2f}
        
        Please provide:
        1. A precise estimate of effort in hours for the new task
        2. A confidence percentage (0-100) for this estimate
        3. An adjustment factor that should be applied to the raw average
        4. A brief explanation of your reasoning and any factors that influenced your estimate
        
        Format your response as JSON with the following fields:
        {{"estimated_hours": float, "confidence": int, "adjustment_factor": float, "explanation": "string"}}
        """
        
        # Call OpenAI API
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional effort estimation assistant that provides precise, data-driven estimates."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=500
        )
        
        # Parse the JSON response
        result = eval(response.choices[0].message.content)
        
        # Add estimated days (assuming 8-hour workdays)
        result['estimated_days'] = result['estimated_hours'] / 8
        
        return result
        
    except Exception as e:
        st.error(f"Error generating AI prediction: {e}")
        return {
            'estimated_hours': 0,
            'estimated_days': 0,
            'confidence': 0,
            'adjustment_factor': 0,
            'explanation': f"Error: {str(e)}"
        }


def find_similar_tasks(task_data, model_data):
    """
    Find similar tasks in historical data
    
    Args:
        task_data: Dictionary with new task details
        model_data: DataFrame with historical task data
        
    Returns:
        DataFrame with similar tasks
    """
    # Start with all completed tasks
    similar_tasks = model_data.copy()
    
    # Filter by issue type (strongest indicator of similarity)
    if task_data['Issue Type'] in similar_tasks['Issue Type'].values:
        similar_tasks = similar_tasks[similar_tasks['Issue Type'] == task_data['Issue Type']]
    
    # Filter by story points if available
    if 'Story Points' in similar_tasks.columns and task_data['Story Points'] > 0:
        # Look for tasks with similar story points (±2)
        story_point_similar = similar_tasks[
            (similar_tasks['Story Points'] >= task_data['Story Points'] - 2) &
            (similar_tasks['Story Points'] <= task_data['Story Points'] + 2)
        ]
        
        # Only apply this filter if it doesn't eliminate too many tasks
        if len(story_point_similar) >= 3:
            similar_tasks = story_point_similar
    
    # Filter by theme if available and if it doesn't eliminate too many tasks
    if 'Theme' in similar_tasks.columns and task_data['Theme'] is not None:
        theme_similar = similar_tasks[similar_tasks['Theme'] == task_data['Theme']]
        if len(theme_similar) >= 3:
            similar_tasks = theme_similar
    
    # Filter by assignee if available and if it doesn't eliminate too many tasks
    if 'Assignee' in similar_tasks.columns and task_data['Assignee'] is not None:
        assignee_similar = similar_tasks[similar_tasks['Assignee'] == task_data['Assignee']]
        if len(assignee_similar) >= 2:
            similar_tasks = assignee_similar
    
    return similar_tasks


def get_ai_velocity_insights(sprint_velocity, assignee_efficiency):
    """
    Generate AI insights about team velocity
    
    Args:
        sprint_velocity: DataFrame with sprint velocity data
        assignee_efficiency: DataFrame with team member efficiency data
        
    Returns:
        String with AI-generated velocity insights
    """
    try:
        # Calculate velocity trends
        if len(sprint_velocity) > 1:
            first_velocity = sprint_velocity.iloc[0]['Story Points']
            last_velocity = sprint_velocity.iloc[-1]['Story Points']
            velocity_change = ((last_velocity - first_velocity) / first_velocity * 100).round(1)
            avg_velocity = sprint_velocity['Story Points'].mean().round(1)
        else:
            velocity_change = 0
            avg_velocity = sprint_velocity['Story Points'].mean().round(1) if not sprint_velocity.empty else 0
        
        # Format the prompt
        prompt = f"""
        You are a Team Velocity Analysis AI for agile project management. Based on the following velocity data,
        provide insights and recommendations for optimizing team performance and sprint planning.
        
        SPRINT VELOCITY DATA:
        - Number of sprints analyzed: {len(sprint_velocity)}
        - Average velocity: {avg_velocity} story points per sprint
        - Velocity change from first to last sprint: {velocity_change}%
        - Sprint-by-sprint points: {sprint_velocity['Story Points'].tolist()}
        
        TEAM MEMBER EFFICIENCY (Hours per Story Point):
        {assignee_efficiency.to_dict('records')}
        
        Provide 4-5 bullet points that:
        1. Analyze the team's velocity trend and identify any patterns or anomalies
        2. Highlight the efficiency differences between team members and potential reasons
        3. Recommend specific actions to improve overall team velocity and consistency
        4. Suggest optimal sprint commitments for future planning based on historical data
        """
        
        # Call OpenAI API
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional agile coach who provides concise, data-driven velocity analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=400
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error generating AI velocity insights: {e}")
        return "*Unable to generate AI velocity insights. Please check your OpenAI API key and try again.*"
