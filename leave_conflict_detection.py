# Advanced Leave Conflict Detection and Resolution Module
# Proactively identifies conflicts between leaves and projects/deadlines
# and provides AI-powered resolution strategies

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from openai import OpenAI
import json

# Initialize OpenAI client
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    client = None
    if not st.session_state.get("openai_error_shown"):
        st.session_state["openai_error_shown"] = True
        st.error(f"Error initializing OpenAI client: {e}. Some AI features will be unavailable.")
        st.info("To enable AI features, please ensure the OPENAI_API_KEY secret is added to your Streamlit secrets.")


def detect_leave_conflicts(issues_df, skills_df, worklogs_df, leaves_df):
    """Main function for detecting leave conflicts with tasks"""
    
    st.markdown("### 🚨 Advanced Leave Conflict Detection & Resolution")
    st.markdown("""
    This feature analyzes upcoming leaves and project schedules to identify potential conflicts 
    and helps you proactively plan to mitigate the impact:
    - Identifies tasks affected by upcoming leaves
    - Suggests preemptive resource reallocation with AI-powered matching
    - Analyzes workload impact and redistribution options
    - Provides detailed resolution plans and knowledge transfer recommendations
    - Simulates the impact of different resolution strategies
    """)
    
    # Add tabs for different aspects of leave conflict management
    detection_tab, resolution_tab, simulation_tab = st.tabs(["📊 Conflict Detection", "🔄 Resolution Planning", "🧮 Impact Simulation"])
    
    # Check if we have the necessary data
    if leaves_df is None or leaves_df.empty:
        with detection_tab, resolution_tab, simulation_tab:
            st.warning("No leave records available. Please upload data with leave information.")
        return
    
    if issues_df is None or issues_df.empty:
        with detection_tab, resolution_tab, simulation_tab:
            st.warning("No issue records available. Please upload data with task information.")
        return
    
    # Transform data
    leaves_df = prepare_leaves_data(leaves_df)
    issues_df = prepare_issues_data(issues_df)
    
    # Get upcoming leaves
    current_date = pd.Timestamp.today()
    with detection_tab:
        upcoming_days = st.slider("Look ahead window (days)", 7, 90, 30)
    lookahead_date = current_date + pd.Timedelta(days=upcoming_days)
    
    upcoming_leaves = leaves_df[
        ((leaves_df['Start Date'] >= current_date) & (leaves_df['Start Date'] <= lookahead_date)) |
        ((leaves_df['End Date'] >= current_date) & (leaves_df['End Date'] <= lookahead_date)) |
        ((leaves_df['Start Date'] <= current_date) & (leaves_df['End Date'] >= lookahead_date))
    ].copy()
    
    if upcoming_leaves.empty:
        with detection_tab, resolution_tab, simulation_tab:
            st.info(f"No leaves scheduled in the next {upcoming_days} days.")
        return
    
    # Process conflicts
    conflicts = identify_conflicts(upcoming_leaves, issues_df, current_date)
    
    # Display detection tab content
    with detection_tab:
        # Display calendar view of upcoming leaves
        show_leave_calendar(upcoming_leaves, current_date, lookahead_date)
        
        # Display conflicts
        if not conflicts.empty:
            st.subheader(f"🔍 Detected {len(conflicts)} Potential Conflicts")
            st.dataframe(conflicts, use_container_width=True)
            
            # Create a timeline visualization of conflicts
            show_conflict_timeline(conflicts, upcoming_leaves, current_date, lookahead_date)
            
            # Generate recommendations
            generate_mitigation_recommendations(conflicts, issues_df, skills_df, worklogs_df, leaves_df)
        else:
            st.success("No immediate conflicts detected between leaves and task deadlines.")
    
    # Resolution planning tab content
    with resolution_tab:
        if not conflicts.empty:
            create_resolution_plan(conflicts, issues_df, skills_df, worklogs_df, leaves_df, client)
        else:
            st.success("No conflicts detected that require resolution planning.")
    
    # Impact simulation tab content
    with simulation_tab:
        if not conflicts.empty:
            simulate_leave_impact(conflicts, issues_df, skills_df, worklogs_df, leaves_df)
        else:
            st.success("No conflicts detected for impact simulation.")


def prepare_leaves_data(leaves_df):
    """Prepare leave data for analysis"""
    # Ensure date columns are datetime
    if 'Start Date' in leaves_df.columns and 'End Date' in leaves_df.columns:
        leaves_df['Start Date'] = pd.to_datetime(leaves_df['Start Date'], errors='coerce')
        leaves_df['End Date'] = pd.to_datetime(leaves_df['End Date'], errors='coerce')
        
        # Add leave duration
        leaves_df['Duration (Days)'] = (leaves_df['End Date'] - leaves_df['Start Date']).dt.days + 1
        
        # Add type classification
        def classify_leave(days):
            if days <= 2:
                return "Short Leave"
            elif days <= 10:
                return "Medium Leave"
            else:
                return "Extended Leave"
        
        leaves_df['Leave Type'] = leaves_df['Duration (Days)'].apply(classify_leave)
    
    return leaves_df

def prepare_issues_data(issues_df):
    """Prepare issue data for conflict analysis"""
    # Convert date columns
    date_columns = ['Due Date', 'Start Date', 'Created', 'Updated']
    for col in date_columns:
        if col in issues_df.columns:
            issues_df[col] = pd.to_datetime(issues_df[col], errors='coerce')
    
    # Add inferred completion date if not present
    if 'Due Date' not in issues_df.columns and 'Start Date' in issues_df.columns:
        # Estimate 10 days per task if due date is missing
        issues_df['Due Date'] = issues_df['Start Date'] + pd.Timedelta(days=10)
    
    return issues_df

def show_leave_calendar(leaves_df, start_date, end_date):
    """Display a calendar view of upcoming leaves"""
    st.subheader("📅 Upcoming Leaves Calendar")
    
    # Create a date range
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Create a DataFrame to hold the leave calendar
    calendar_data = []
    
    # Add all leaves to the calendar
    for _, leave in leaves_df.iterrows():
        resource = leave.get('Resource', leave.get('Name', 'Unknown'))
        leave_type = leave.get('Leave Type', 'Unknown')
        leave_start = leave['Start Date']
        leave_end = leave['End Date']
        
        for date in pd.date_range(start=max(leave_start, start_date), end=min(leave_end, end_date)):
            calendar_data.append({
                'Date': date,
                'Resource': resource,
                'Leave Type': leave_type,
                'Status': 'On Leave'
            })
    
    # Create the calendar DataFrame
    if calendar_data:
        calendar_df = pd.DataFrame(calendar_data)
        
        # Create a heatmap
        fig = px.density_heatmap(
            calendar_df,
            x='Date',
            y='Resource',
            # Color by leave type
            color_continuous_scale=[
                [0, 'lightblue'],   # Short leave
                [0.5, 'royalblue'],  # Medium leave
                [1, 'darkblue']     # Extended leave
            ],
            title="Team Leave Calendar"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No leave data available for the selected period.")

def identify_conflicts(leaves_df, issues_df, current_date):
    """Identify conflicts between leaves and task deadlines"""
    conflicts = []
    
    # Focus on open issues with due dates
    open_issues = issues_df[issues_df['Status'] != 'Done'].copy() if 'Status' in issues_df.columns else issues_df.copy()
    
    # Only process if we have due dates
    if 'Due Date' in open_issues.columns and 'Assignee' in open_issues.columns:
        for _, leave in leaves_df.iterrows():
            resource = leave.get('Resource', leave.get('Name', 'Unknown'))
            leave_start = leave['Start Date']
            leave_end = leave['End Date']
            leave_type = leave.get('Leave Type', 'Unknown')
            
            # Find tasks assigned to this resource with deadlines during leave
            resource_tasks = open_issues[open_issues['Assignee'] == resource]
            
            for _, task in resource_tasks.iterrows():
                due_date = task['Due Date']
                
                # Check if due date is during or right after leave
                if pd.notna(due_date):
                    # Define conflict states
                    during_leave = (due_date >= leave_start) and (due_date <= leave_end)
                    right_after_leave = (due_date > leave_end) and (due_date <= leave_end + pd.Timedelta(days=3))
                    
                    if during_leave or right_after_leave:
                        # Calculate risk level
                        if during_leave:
                            risk_level = "High"
                            days_impacted = (min(due_date, leave_end) - max(current_date, leave_start)).days + 1
                        else:  # right after leave
                            risk_level = "Medium"
                            days_impacted = (leave_end - max(current_date, leave_start)).days + 1
                        
                        # Add to conflicts list
                        conflicts.append({
                            'Resource': resource,
                            'Task Key': task.get('Issue Key', 'Unknown'),
                            'Task Summary': task.get('Summary', 'Unknown'),
                            'Due Date': due_date,
                            'Leave Start': leave_start,
                            'Leave End': leave_end,
                            'Risk Level': risk_level,
                            'Days Impacted': days_impacted,
                            'Leave Type': leave_type
                        })
    
    # Convert to DataFrame and sort by risk level and due date
    if conflicts:
        conflict_df = pd.DataFrame(conflicts)
        risk_order = {"High": 0, "Medium": 1, "Low": 2}
        conflict_df['Risk Order'] = conflict_df['Risk Level'].map(risk_order)
        conflict_df = conflict_df.sort_values(['Risk Order', 'Due Date']).drop(columns=['Risk Order'])
        return conflict_df
    
    return pd.DataFrame()

def show_conflict_timeline(conflicts, leaves_df, start_date, end_date):
    """Show a timeline of conflicts with leave periods"""
    st.subheader("⏱️ Conflict Timeline")
    
    try:
        # Create Gantt chart data
        gantt_data = []
        
        # Add leave periods with safe date handling
        for _, leave in leaves_df.iterrows():
            try:
                resource = leave.get('Resource', leave.get('Name', 'Unknown'))
                
                # Convert dates to string format
                start_date_str = pd.to_datetime(leave['Start Date']).strftime('%Y-%m-%d')
                if 'End Date' in leave:
                    end_date_str = pd.to_datetime(leave['End Date']).strftime('%Y-%m-%d')
                else:
                    # Calculate end date from duration if end date not available
                    from datetime import datetime, timedelta
                    start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
                    duration = int(leave.get('Duration (Days)', 1))
                    end_dt = start_dt + timedelta(days=duration)
                    end_date_str = end_dt.strftime('%Y-%m-%d')
                    
                gantt_data.append({
                    'Task': f"{resource} (Leave)",
                    'Start': start_date_str,
                    'Finish': end_date_str,
                    'Resource': resource,
                    'Type': 'Leave',
                    'Leave Type': leave.get('Leave Type', 'Unknown')
                })
            except Exception as e:
                st.warning(f"Could not process leave entry: {e}")
                continue
        
        # Add task deadlines with safe date handling
        for _, conflict in conflicts.iterrows():
            try:
                # Create a task bar that ends at due date
                # Convert dates to string format
                if 'Due Date' in conflict:
                    due_date_str = pd.to_datetime(conflict['Due Date']).strftime('%Y-%m-%d')
                    # Calculate start date for visualization (5 days before)
                    from datetime import datetime, timedelta
                    due_dt = datetime.strptime(due_date_str, '%Y-%m-%d')
                    start_dt = due_dt - timedelta(days=5)  # Arbitrary task length for visualization
                    start_date_str = start_dt.strftime('%Y-%m-%d')
                    
                    task_key = conflict.get('Task Key', conflict.get('Issue Key', 'Unknown'))
                    task_summary = conflict.get('Task Summary', conflict.get('Summary', 'Unknown Task'))
                    
                    gantt_data.append({
                        'Task': f"{task_key} - {task_summary[:20]}...",
                        'Start': start_date_str,
                        'Finish': due_date_str,
                        'Resource': conflict['Resource'],
                        'Type': 'Task',
                        'Risk Level': conflict.get('Risk Level', 'Medium')
                    })
            except Exception as e:
                st.warning(f"Could not process conflict entry: {e}")
                continue
    except Exception as e:
        st.error(f"Error processing conflict data: {e}")
        gantt_data = []
        
    # Create the Gantt chart
    if gantt_data:
        gantt_df = pd.DataFrame(gantt_data)
        
        # Create color map
        color_map = {
            'Leave': 'rgba(200, 200, 200, 0.8)',  # Gray for leaves
            'Task': 'rgba(55, 128, 191, 0.8)'      # Blue for tasks
        }
        
        # Create figure
        fig = px.timeline(
            gantt_df, 
            x_start="Start", 
            x_end="Finish", 
            y="Task",
            color="Type",
            color_discrete_map=color_map,
            title="Leave and Task Timeline"
        )
        
        # Add a vertical line for today using shape instead of vline
        # This approach avoids timestamp math issues
        from datetime import datetime
        today_str = datetime.now().strftime('%Y-%m-%d')
        
        # Use add_shape instead of add_vline for more compatibility
        fig.add_shape(
            type="line",
            x0=today_str,
            x1=today_str,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dash"),
        )
        
        # Add annotation for today
        fig.add_annotation(
            x=today_str,
            y=1.05,
            yref="paper",
            text="Today",
            showarrow=False,
            font=dict(color="red")
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No conflict data available for visualization.")

def generate_mitigation_recommendations(conflicts, issues_df, skills_df, worklogs_df, leaves_df):
    """Generate recommendations for mitigating leave conflicts"""
    st.subheader("🧠 Mitigation Recommendations")
    
    # Group conflicts by resource
    resource_conflicts = conflicts.groupby('Resource')
    
    for resource, resource_conflicts_df in resource_conflicts:
        # Display resource-specific recommendations
        st.markdown(f"**Recommendations for {resource}**")
        
        # Count high-risk tasks
        high_risk = resource_conflicts_df[resource_conflicts_df['Risk Level'] == 'High']
        medium_risk = resource_conflicts_df[resource_conflicts_df['Risk Level'] == 'Medium']
        
        # Get leave information
        leave_info = leaves_df[leaves_df['Resource'] == resource].iloc[0]
        leave_start = leave_info['Start Date']
        leave_duration = leave_info['Duration (Days)']
        
        # Create recommendations based on risk level
        recommendations = []
        
        if not high_risk.empty:
            recommendations.append(f"⚠️ **Critical**: {len(high_risk)} high-risk tasks due during leave period.")
            recommendations.append("💡 Recommendation: Immediately reassign or reschedule these tasks.")
        
        if not medium_risk.empty:
            recommendations.append(f"⚠️ **Attention**: {len(medium_risk)} medium-risk tasks due shortly after leave.")
            recommendations.append("💡 Recommendation: Create knowledge transfer plan before leave.")
        
        # Calculate lead time for knowledge transfer
        try:
            # Convert both dates to datetime.date objects for clean subtraction
            today_date = pd.Timestamp.today().date()
            leave_date = pd.to_datetime(leave_start).date()
            days_to_leave = (leave_date - today_date).days
        except Exception as e:
            st.warning(f"Could not calculate days to leave: {e}")
            days_to_leave = 0
            
        if days_to_leave > 0:
            recommendations.append(f"⏰ {days_to_leave} days remaining to prepare for knowledge transfer.")
            
            # Recommend knowledge transfer time based on leave duration and task count
            total_tasks = len(resource_conflicts_df)
            if leave_duration > 10 or total_tasks > 5:
                recommendations.append("🔄 Recommend 3-5 days for comprehensive knowledge transfer.")
            elif leave_duration > 5 or total_tasks > 2:
                recommendations.append("🔄 Recommend 1-2 days for task handover sessions.")
            else:
                recommendations.append("🔄 Recommend detailed task documentation before leave.")
        
        # Find suitable replacements based on skills if we have skills data
        if skills_df is not None and not skills_df.empty:
            resource_col = 'Resource' if 'Resource' in skills_df.columns else 'Name'
            if resource_col in skills_df.columns and 'Skillset' in skills_df.columns:
                # Get skills of the person on leave
                resource_skills = skills_df[skills_df[resource_col] == resource]['Skillset'].tolist()
                
                if resource_skills:
                    # Find people with similar skills
                    similar_skilled = skills_df[
                        (skills_df[resource_col] != resource) & 
                        (skills_df['Skillset'].isin(resource_skills))
                    ][resource_col].unique().tolist()
                    
                    if similar_skilled:
                        recommendations.append(f"👥 Suggested backup resources with matching skills: {', '.join(similar_skilled[:3])}")
        
        # Display recommendations
        for rec in recommendations:
            st.markdown(rec)
        
        st.markdown("---")


def create_resolution_plan(conflicts, issues_df, skills_df, worklogs_df, leaves_df, client=None):
    """Create an advanced conflict resolution plan with AI recommendations"""
    st.subheader("🔧 Advanced Conflict Resolution Planning")
    
    # Show counters for tracking conflicts
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Conflicts", len(conflicts))
    with col2:
        high_risk_count = len(conflicts[conflicts['Risk Level'] == 'High'])
        st.metric("High Risk", high_risk_count, delta=None, delta_color="inverse")
    with col3:
        medium_risk_count = len(conflicts[conflicts['Risk Level'] == 'Medium'])
        st.metric("Medium Risk", medium_risk_count, delta=None, delta_color="inverse")
    
    # Group conflicts by resource
    resource_conflicts = conflicts.groupby('Resource')
    
    # For each affected team member
    for resource, resource_conflicts_df in resource_conflicts:
        st.markdown(f"### Resolution Plan for {resource}")
        
        # Get leave information
        leave_info = leaves_df[leaves_df['Resource'] == resource].iloc[0]
        leave_start = leave_info['Start Date']
        leave_end = leave_info['End Date']
        leave_duration = leave_info['Duration (Days)']
        
        # Display leave details
        st.markdown(f"**Leave Period:** {leave_start.strftime('%b %d, %Y')} to {leave_end.strftime('%b %d, %Y')} ({leave_duration} days)")
        
        # Create task resolution interface
        with st.expander(f"Resolve {len(resource_conflicts_df)} Tasks", expanded=True):
            # Display tasks in a more actionable format
            for i, (_, conflict) in enumerate(resource_conflicts_df.iterrows()):
                task_key = conflict['Task Key']
                task_summary = conflict['Task Summary']
                due_date = conflict['Due Date']
                risk_level = conflict['Risk Level']
                
                # Create a color-coded task card
                risk_color = "🔴" if risk_level == "High" else "🟠" if risk_level == "Medium" else "🟢"
                
                st.markdown(f"{risk_color} **{task_key}**: {task_summary}")
                st.markdown(f"Due: {due_date.strftime('%b %d, %Y')} | Risk: {risk_level}")
                
                # Get potential assignees with matching skills
                potential_assignees = []
                if skills_df is not None and not skills_df.empty:
                    resource_col = 'Resource' if 'Resource' in skills_df.columns else 'Name'
                    
                    # Get the skillset needed for this task
                    if 'Issue Key' in issues_df.columns and task_key in issues_df['Issue Key'].values:
                        task_info = issues_df[issues_df['Issue Key'] == task_key].iloc[0]
                        task_type = task_info.get('Issue Type', '')
                        task_component = task_info.get('Component', '')
                        
                        # Find resources with similar skills
                        if 'Skillset' in skills_df.columns:
                            skilled_resources = skills_df[
                                (skills_df[resource_col] != resource) & 
                                (skills_df['Skillset'].str.contains(task_type, case=False, na=False) | 
                                 skills_df['Skillset'].str.contains(task_component, case=False, na=False))
                            ][resource_col].unique().tolist()
                            
                            potential_assignees = skilled_resources
                
                # Resolution options
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    # Option to reschedule/delay
                    st.markdown("**Option 1: Reschedule**")
                    new_due_date = due_date + pd.Timedelta(days=leave_duration+3)
                    st.markdown(f"New Due Date: {new_due_date.strftime('%b %d, %Y')}")
                    delay_impact = "Low" if new_due_date.month == due_date.month else "Medium"
                    st.markdown(f"Delay Impact: {delay_impact}")
                
                with col2:
                    # Option to reassign
                    st.markdown("**Option 2: Reassign**")
                    if potential_assignees:
                        assignee_options = ["Select assignee"] + potential_assignees
                        selected_assignee = st.selectbox(
                            "Best match assignees", 
                            assignee_options,
                            key=f"assignee_{task_key}"
                        )
                        if selected_assignee != "Select assignee":
                            st.markdown(f"Selected: {selected_assignee}")
                    else:
                        st.markdown("No skill-matched assignees found")
                
                with col3:
                    # Option for knowledge transfer
                    st.markdown("**Option 3: Knowledge Transfer**")
                    kt_options = ["None", "Documentation", "Pair Programming", "Handover Meeting"]
                    kt_selection = st.selectbox(
                        "Transfer method",
                        kt_options,
                        key=f"kt_{task_key}"
                    )
                    if kt_selection != "None":
                        st.markdown(f"Selected: {kt_selection}")
                
                # Add a divider between tasks
                st.markdown("---")
            
            # Generate AI recommendations if OpenAI client is available
            if client is not None:
                st.markdown("### 🤖 AI-Powered Resolution Recommendations")
                
                try:
                    # Prepare data for AI recommendation
                    resource_conflicts_json = resource_conflicts_df.to_json(orient='records', date_format='iso')
                    leave_info_json = leave_info.to_json(date_format='iso')
                    
                    # Build the prompt
                    prompt = f"""
                    You are an expert project manager. Analyze these task conflicts due to an upcoming leave of team member '{resource}'.
                    
                    Leave details:
                    {leave_info_json}
                    
                    Conflicting tasks:
                    {resource_conflicts_json}
                    
                    Provide a concise, structured resolution plan with specific recommendations for each task.
                    Consider these factors:
                    1. Risk level and due date proximity
                    2. Task complexity and knowledge transfer needs
                    3. Most appropriate action (reschedule, reassign, or advance completion)
                    
                    Format your response as a JSON with this structure:
                    {{
                        "overall_strategy": "Brief 2-3 sentence high-level strategy",
                        "task_recommendations": [
                            {{
                                "task_key": "KEY-123",
                                "recommendation": "Specific action",
                                "rationale": "Brief explanation",
                                "action_type": "Reschedule|Reassign|Transfer"
                            }}
                        ],
                        "knowledge_transfer_plan": "Specific knowledge transfer recommendations"
                    }}
                    """
                    
                    # Send to OpenAI API
                    response = client.chat.completions.create(
                        model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. Do not change this unless explicitly requested by the user
                        messages=[{"role": "system", "content": "You are an AI assistant that helps with project management."}, 
                                 {"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        temperature=0.2,
                    )
                    
                    # Parse the response
                    ai_recommendations = json.loads(response.choices[0].message.content)
                    
                    # Display AI recommendations
                    st.markdown(f"**Overall Strategy:**")
                    st.info(ai_recommendations.get("overall_strategy", "No strategy provided"))
                    
                    st.markdown("**Recommendations by Task:**")
                    for task_rec in ai_recommendations.get("task_recommendations", []):
                        task_key = task_rec.get("task_key", "Unknown")
                        action_type = task_rec.get("action_type", "Unknown")
                        action_icon = "🗓️" if action_type == "Reschedule" else "👤" if action_type == "Reassign" else "📝"
                        
                        st.markdown(f"{action_icon} **{task_key}**: {task_rec.get('recommendation')}")
                        st.markdown(f"_Rationale: {task_rec.get('rationale')}_")
                    
                    st.markdown("**Knowledge Transfer Plan:**")
                    st.info(ai_recommendations.get("knowledge_transfer_plan", "No knowledge transfer plan provided"))
                    
                except Exception as e:
                    st.error(f"Could not generate AI recommendations: {e}")
        
        # Add action buttons for implementing the plan
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply Resolution Plan", key=f"apply_{resource}"):
                st.success("This would update JIRA with the resolution plan (simulation only)")
        with col2:
            if st.button("Export Plan to PDF", key=f"export_{resource}"):
                st.info("This would generate a PDF of the resolution plan (simulation only)")
        
        st.markdown("---")


def simulate_leave_impact(conflicts, issues_df, skills_df, worklogs_df, leaves_df):
    """Simulate the impact of leaves on project timelines and resource allocation"""
    st.subheader("📊 Leave Impact Simulation")
    
    # Group conflicts by resource
    resource_conflicts = conflicts.groupby('Resource')
    
    # Select resource to simulate
    resources = conflicts['Resource'].unique().tolist()
    selected_resource = st.selectbox(
        "Select resource to simulate leave impact",
        resources
    )
    
    # Get conflicts for selected resource
    if selected_resource:
        resource_conflicts_df = conflicts[conflicts['Resource'] == selected_resource]
        
        # Get leave information
        leave_info = leaves_df[leaves_df['Resource'] == selected_resource].iloc[0]
        leave_start = leave_info['Start Date']
        leave_end = leave_info['End Date']
        leave_duration = leave_info['Duration (Days)']
        
        st.markdown(f"**Simulating impact of {selected_resource}'s {leave_duration} day leave**")
        st.markdown(f"Leave period: {leave_start.strftime('%b %d, %Y')} to {leave_end.strftime('%b %d, %Y')}")
        
        # Simulation parameters
        st.markdown("### Simulation Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            strategy = st.radio(
                "Resolution Strategy",
                ["No Action", "Reschedule All", "Reassign High Risk", "Optimized Mixed Strategy"]
            )
        
        with col2:
            knowledge_transfer = st.slider(
                "Knowledge Transfer Days",
                0, 10, 2,
                help="Days spent on knowledge transfer before leave"
            )
        
        # Run simulation based on selected parameters
        if st.button("Run Simulation"):
            # Prepare simulation data
            simulation_data = {}
            
            # Calculate baseline impact (no action)
            baseline_delay = calculate_baseline_impact(resource_conflicts_df, leave_duration)
            
            # Calculate strategy impact
            if strategy == "No Action":
                strategy_impact = baseline_delay
                strategy_description = "No mitigation actions taken"
            elif strategy == "Reschedule All":
                strategy_impact = baseline_delay * 0.5  # Simulated impact reduction
                strategy_description = "All tasks rescheduled to after leave period"
            elif strategy == "Reassign High Risk":
                high_risk_count = len(resource_conflicts_df[resource_conflicts_df['Risk Level'] == 'High'])
                strategy_impact = baseline_delay * (1 - (high_risk_count / len(resource_conflicts_df) * 0.8))
                strategy_description = f"{high_risk_count} high-risk tasks reassigned to other team members"
            else:  # Optimized Mixed Strategy
                strategy_impact = baseline_delay * 0.3  # Simulated optimal impact reduction
                strategy_description = "Optimized combination of rescheduling, reassignment, and acceleration"
            
            # Apply knowledge transfer effect
            if knowledge_transfer > 0:
                kt_reduction = min(0.7, knowledge_transfer * 0.07)  # Up to 70% reduction with 10 days of KT
                final_impact = strategy_impact * (1 - kt_reduction)
                kt_description = f"{knowledge_transfer} days of knowledge transfer reduces impact by {int(kt_reduction*100)}%"
            else:
                final_impact = strategy_impact
                kt_description = "No knowledge transfer conducted"
            
            # Display simulation results
            st.markdown("### 📊 Simulation Results")
            
            # Impact metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Baseline Project Delay", 
                    f"{baseline_delay:.1f} days",
                    delta=None
                )
            with col2:
                st.metric(
                    "With Selected Strategy", 
                    f"{strategy_impact:.1f} days",
                    delta=f"-{baseline_delay - strategy_impact:.1f} days",
                    delta_color="inverse"
                )
            with col3:
                st.metric(
                    "With Knowledge Transfer", 
                    f"{final_impact:.1f} days",
                    delta=f"-{strategy_impact - final_impact:.1f} days",
                    delta_color="inverse"
                )
            
            # Strategy details
            st.markdown("### 📝 Strategy Details")
            st.markdown(f"**Selected Strategy:** {strategy}")
            st.info(strategy_description)
            st.markdown(f"**Knowledge Transfer:** {knowledge_transfer} days")
            st.info(kt_description)
            
            # Visual simulation
            st.markdown("### 📈 Timeline Visualization")
            
            # Create before/after visualization
            create_impact_visualization(
                resource_conflicts_df, 
                leave_start, 
                leave_end,
                strategy,
                knowledge_transfer,
                final_impact
            )
            
            # Recommendations based on simulation
            st.markdown("### 🧠 Simulation-Based Recommendations")
            if final_impact > 5:
                st.warning(f"⚠️ Significant project impact of {final_impact:.1f} days detected even with mitigations")
                st.markdown("**Consider additional measures:**")
                st.markdown("- Evaluate critical path tasks for further optimization")
                st.markdown("- Consider temporarily bringing in additional resources")
                st.markdown("- Evaluate project dependencies for potential adjustments")
            elif final_impact > 2:
                st.info(f"ℹ️ Moderate project impact of {final_impact:.1f} days with current strategy")
                st.markdown("**Recommendations:**")
                st.markdown("- Current strategy appears effective but monitor closely")
                st.markdown("- Ensure knowledge transfer is well-structured and thorough")
                st.markdown("- Prepare contingency plans for high-risk tasks")
            else:
                st.success(f"✅ Minimal project impact of {final_impact:.1f} days with current strategy")
                st.markdown("**Recommendations:**")
                st.markdown("- Strategy is working well, proceed with implementation")
                st.markdown("- Document the approach for future similar situations")
                st.markdown("- Continue with regular project monitoring")


def calculate_baseline_impact(resource_conflicts_df, leave_duration):
    """Calculate baseline impact of leave without any mitigation"""
    # Simple simulation model - can be enhanced with more sophisticated calculations
    high_risk_count = len(resource_conflicts_df[resource_conflicts_df['Risk Level'] == 'High'])
    medium_risk_count = len(resource_conflicts_df[resource_conflicts_df['Risk Level'] == 'Medium'])
    
    # Impact formula: leave duration + weighted impact of high/medium tasks
    impact = leave_duration * 0.7 + high_risk_count * 1.5 + medium_risk_count * 0.7
    
    return impact


def create_impact_visualization(conflicts_df, leave_start, leave_end, strategy, kt_days, impact):
    """Create visualization of leave impact on project timeline"""
    # Create date range for visualization
    start_date = leave_start - pd.Timedelta(days=10)
    end_date = leave_end + pd.Timedelta(days=20)
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Create timeline data
    timeline_data = []
    
    # Add leave period
    for date in pd.date_range(start=leave_start, end=leave_end):
        timeline_data.append({
            'Date': date,
            'Category': 'Leave Period',
            'Value': 1
        })
    
    # Add knowledge transfer period if applicable
    if kt_days > 0:
        for date in pd.date_range(start=leave_start-pd.Timedelta(days=kt_days), end=leave_start-pd.Timedelta(days=1)):
            timeline_data.append({
                'Date': date,
                'Category': 'Knowledge Transfer',
                'Value': 1
            })
    
    # Add tasks based on strategy
    for _, task in conflicts_df.iterrows():
        task_key = task['Task Key']
        due_date = task['Due Date']
        risk_level = task['Risk Level']
        
        # Original due date
        timeline_data.append({
            'Date': due_date,
            'Category': 'Original Due Dates',
            'Value': 1,
            'Task': task_key,
            'Risk': risk_level
        })
        
        # Adjusted due date based on strategy
        if strategy == "No Action":
            new_due_date = due_date + pd.Timedelta(days=impact if due_date >= leave_start else 0)
        elif strategy == "Reschedule All":
            new_due_date = max(due_date, leave_end + pd.Timedelta(days=2))
        elif strategy == "Reassign High Risk":
            if risk_level == "High":
                new_due_date = due_date  # No change as reassigned
            else:
                new_due_date = due_date + pd.Timedelta(days=impact * 0.7 if due_date >= leave_start else 0)
        else:  # Optimized Mixed Strategy
            if risk_level == "High":
                new_due_date = due_date  # High risk reassigned
            elif risk_level == "Medium":
                new_due_date = max(due_date, leave_end)  # Medium risk rescheduled
            else:
                new_due_date = due_date + pd.Timedelta(days=impact * 0.5 if due_date >= leave_start else 0)
        
        timeline_data.append({
            'Date': new_due_date,
            'Category': 'Adjusted Due Dates',
            'Value': 1,
            'Task': task_key,
            'Risk': risk_level
        })
    
    # Create DataFrame
    timeline_df = pd.DataFrame(timeline_data)
    
    # Create visualization
    if not timeline_df.empty:
        # Create a heatmap of the timeline
        fig = px.density_heatmap(
            timeline_df,
            x='Date',
            y='Category',
            title="Leave Impact Simulation"
        )
        
        # Add a vertical line for today
        today = pd.Timestamp.today()
        fig.add_vline(x=today, line_width=2, line_dash="dash", line_color="red")
        
        # Add annotation for today
        fig.add_annotation(
            x=today,
            y=1.05,
            yref="paper",
            text="Today",
            showarrow=False,
            font=dict(color="red")
        )
        
        # Update layout
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a more detailed task visualization
        create_task_shift_visualization(conflicts_df, leave_start, leave_end, strategy, impact)
    else:
        st.info("No data available for visualization")


def create_task_shift_visualization(conflicts_df, leave_start, leave_end, strategy, impact):
    """Create a visualization showing how tasks shift due to leave"""
    # Create Gantt chart data for task shifts
    gantt_data = []
    
    # Add leave period
    leave_start_str = leave_start.strftime('%Y-%m-%d')
    leave_end_str = leave_end.strftime('%Y-%m-%d')
    
    gantt_data.append({
        'Task': "Leave Period",
        'Start': leave_start_str,
        'Finish': leave_end_str,
        'Resource': "Leave",
        'Type': 'Leave',
    })
    
    # Add task shifts
    for _, task in conflicts_df.iterrows():
        task_key = task['Task Key']
        task_summary = task['Task Summary']
        due_date = task['Due Date']
        risk_level = task['Risk Level']
        
        # Convert dates to string format
        due_date_str = due_date.strftime('%Y-%m-%d')
        
        # Calculate task start for visualization (5 days before due date)
        start_date = due_date - pd.Timedelta(days=5)
        start_date_str = start_date.strftime('%Y-%m-%d')
        
        # Add original task timeline
        gantt_data.append({
            'Task': f"{task_key} - Original",
            'Start': start_date_str,
            'Finish': due_date_str,
            'Resource': task_key,
            'Type': 'Original',
            'Risk': risk_level
        })
        
        # Calculate new due date based on strategy
        if strategy == "No Action":
            new_due_date = due_date + pd.Timedelta(days=impact if due_date >= leave_start else 0)
        elif strategy == "Reschedule All":
            new_due_date = max(due_date, leave_end + pd.Timedelta(days=2))
        elif strategy == "Reassign High Risk":
            if risk_level == "High":
                new_due_date = due_date  # No change as reassigned
            else:
                new_due_date = due_date + pd.Timedelta(days=impact * 0.7 if due_date >= leave_start else 0)
        else:  # Optimized Mixed Strategy
            if risk_level == "High":
                new_due_date = due_date  # High risk reassigned
            elif risk_level == "Medium":
                new_due_date = max(due_date, leave_end)  # Medium risk rescheduled
            else:
                new_due_date = due_date + pd.Timedelta(days=impact * 0.5 if due_date >= leave_start else 0)
        
        # Add adjusted task timeline
        new_due_date_str = new_due_date.strftime('%Y-%m-%d')
        new_start_date = new_due_date - pd.Timedelta(days=5)  # Arbitrary task length
        new_start_date_str = new_start_date.strftime('%Y-%m-%d')
        
        # Only add if there's a change
        if new_due_date != due_date:
            gantt_data.append({
                'Task': f"{task_key} - {strategy}",
                'Start': new_start_date_str,
                'Finish': new_due_date_str,
                'Resource': task_key,
                'Type': 'Adjusted',
                'Risk': risk_level
            })
    
    # Create DataFrame
    gantt_df = pd.DataFrame(gantt_data)
    
    # Create color map
    color_map = {
        'Leave': 'rgba(200, 200, 200, 0.8)',  # Gray for leaves
        'Original': 'rgba(55, 128, 191, 0.6)',  # Transparent blue for original
        'Adjusted': 'rgba(255, 0, 0, 0.6)'      # Transparent red for adjusted
    }
    
    # Create Gantt chart
    fig = px.timeline(
        gantt_df, 
        x_start="Start", 
        x_end="Finish", 
        y="Task",
        color="Type",
        color_discrete_map=color_map,
        title="Task Timeline Shifts Due to Leave Impact"
    )
    
    # Add today line
    today = datetime.now().strftime('%Y-%m-%d')
    fig.add_shape(
        type="line",
        x0=today,
        x1=today,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="green", width=2, dash="dash"),
    )
    
    # Add annotation for today
    fig.add_annotation(
        x=today,
        y=1.05,
        yref="paper",
        text="Today",
        showarrow=False,
        font=dict(color="green")
    )
    
    # Update layout
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
