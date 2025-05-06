# Sprint Planning Assistant Module
# Provides intelligent recommendations for sprint planning

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
    if not st.session_state.get("openai_error_shown_sprint"):
        st.session_state["openai_error_shown_sprint"] = True
        st.error(f"Error initializing OpenAI client: {e}. Some AI features will be unavailable.")
        st.info("To enable AI features, please ensure the OPENAI_API_KEY secret is added to your Streamlit secrets.")

def sprint_planning_assistant(issues_df, skills_df, worklogs_df, leaves_df):
    """Main function for intelligent sprint planning"""
    
    st.markdown("### 🏃‍♀️ Sprint Planning Assistant")
    st.markdown("""
    This feature helps you plan efficient and balanced sprints:
    - Analyzes team capacity considering skills and unavailability
    - Recommends optimal sprint composition
    - Identifies dependencies and critical path items
    - Provides AI-based insights for sprint success
    - Simulates different sprint scenarios
    """)
    
    # Add tabs for different aspects of sprint planning
    capacity_tab, composition_tab, simulation_tab = st.tabs(["👥 Team Capacity", "🧩 Sprint Composition", "🔮 Sprint Simulation"])
    
    # Check if we have the necessary data
    if issues_df is None or issues_df.empty:
        with capacity_tab, composition_tab, simulation_tab:
            st.warning("No issue records available. Please upload data with task information.")
        return
    
    # Team Capacity Planning tab
    with capacity_tab:
        team_capacity_planning(issues_df, skills_df, worklogs_df, leaves_df)
    
    # Sprint Composition tab
    with composition_tab:
        sprint_composition_recommendations(issues_df, skills_df, worklogs_df, leaves_df)
    
    # Sprint Simulation tab
    with simulation_tab:
        sprint_simulation(issues_df, skills_df, worklogs_df, leaves_df)


def team_capacity_planning(issues_df, skills_df, worklogs_df, leaves_df):
    """Analyze and visualize team capacity for sprint planning"""
    st.subheader("👥 Team Capacity Analysis")
    
    # Sprint duration setting
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        sprint_duration = st.number_input("Sprint Duration (days)", min_value=5, max_value=30, value=14, step=1)
    with col2:
        hours_per_day = st.number_input("Working Hours per Day", min_value=4, max_value=12, value=8, step=1)
    with col3:
        start_date = st.date_input("Sprint Start Date", value=datetime.now())
    
    # Calculate sprint end date
    end_date = start_date + timedelta(days=sprint_duration-1)
    sprint_date_range = pd.date_range(start=start_date, end=end_date)
    
    # Calculate theoretical capacity
    if 'Resource' in skills_df.columns:
        team_members = skills_df['Resource'].unique().tolist()
    elif 'Name' in skills_df.columns:
        team_members = skills_df['Name'].unique().tolist()
    else:
        st.error("Skills data must contain 'Resource' or 'Name' column.")
        return
    
    total_theoretical_capacity = len(team_members) * sprint_duration * hours_per_day
    
    # Calculate unavailable time due to leaves
    unavailable_hours = 0
    leaves_during_sprint = None
    if leaves_df is not None and not leaves_df.empty:
        # Ensure date columns are datetime
        if 'Start Date' in leaves_df.columns and 'End Date' in leaves_df.columns:
            leaves_df['Start Date'] = pd.to_datetime(leaves_df['Start Date'], errors='coerce')
            leaves_df['End Date'] = pd.to_datetime(leaves_df['End Date'], errors='coerce')
            
            # Filter leaves during sprint
            sprint_start = pd.Timestamp(start_date)
            sprint_end = pd.Timestamp(end_date)
            
            leaves_during_sprint = leaves_df[
                ((leaves_df['Start Date'] <= sprint_end) & (leaves_df['End Date'] >= sprint_start))
            ].copy()
            
            # Calculate overlapping days and hours
            if not leaves_during_sprint.empty:
                for _, leave in leaves_during_sprint.iterrows():
                    leave_start = max(leave['Start Date'], sprint_start)
                    leave_end = min(leave['End Date'], sprint_end)
                    overlap_days = (leave_end - leave_start).days + 1
                    unavailable_hours += overlap_days * hours_per_day
    
    # Calculate effective capacity
    effective_capacity = total_theoretical_capacity - unavailable_hours
    capacity_percentage = (effective_capacity / total_theoretical_capacity) * 100 if total_theoretical_capacity > 0 else 0
    
    # Display capacity metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Theoretical Hours", f"{total_theoretical_capacity:.1f}")
    with col2:
        st.metric("Unavailable Hours", f"{unavailable_hours:.1f}", delta=f"-{unavailable_hours:.1f}", delta_color="inverse")
    with col3:
        st.metric("Effective Capacity", f"{effective_capacity:.1f}", delta=f"{capacity_percentage:.1f}%")
    
    # Display leave information during sprint
    if leaves_during_sprint is not None and not leaves_during_sprint.empty:
        st.subheader("📅 Team Member Unavailability during Sprint")
        # Format for display
        display_leaves = leaves_during_sprint.copy()
        if 'Start Date' in display_leaves.columns:
            display_leaves['Start Date'] = display_leaves['Start Date'].dt.strftime('%Y-%m-%d')
        if 'End Date' in display_leaves.columns:
            display_leaves['End Date'] = display_leaves['End Date'].dt.strftime('%Y-%m-%d')
        st.dataframe(display_leaves, use_container_width=True)
    
    # Individual capacity analysis
    st.subheader("👤 Individual Team Member Capacity")
    
    # Create a dataframe with each team member's capacity
    individual_capacity = []
    for member in team_members:
        # Default capacity
        member_capacity = sprint_duration * hours_per_day
        
        # Subtract leave hours if any
        if leaves_during_sprint is not None and not leaves_during_sprint.empty:
            if 'User' in leaves_during_sprint.columns:
                member_leaves = leaves_during_sprint[leaves_during_sprint['User'] == member]
                
                leave_hours = 0
                for _, leave in member_leaves.iterrows():
                    leave_start = max(leave['Start Date'], pd.Timestamp(start_date))
                    leave_end = min(leave['End Date'], pd.Timestamp(end_date))
                    overlap_days = (leave_end - leave_start).days + 1
                    leave_hours += overlap_days * hours_per_day
                
                member_capacity -= leave_hours
        
        individual_capacity.append({
            'Team Member': member,
            'Available Hours': member_capacity,
            'Utilization %': (member_capacity / (sprint_duration * hours_per_day)) * 100
        })
    
    capacity_df = pd.DataFrame(individual_capacity)
    
    # Display as a chart
    fig = px.bar(
        capacity_df, 
        x='Team Member', 
        y='Available Hours',
        color='Utilization %',
        color_continuous_scale='RdYlGn',  # Red to Yellow to Green
        range_color=[0, 100],
        title='Available Hours per Team Member',
        text='Available Hours'
    )
    fig.update_layout(xaxis_title='Team Member', yaxis_title='Available Hours')
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate skill coverage during sprint
    if 'Skillset' in skills_df.columns:
        st.subheader("🧠 Skill Coverage during Sprint")
        
        # Group skills by team member
        if 'Resource' in skills_df.columns:
            skill_group = skills_df.groupby('Resource')['Skillset'].apply(list).to_dict()
        elif 'Name' in skills_df.columns:
            skill_group = skills_df.groupby('Name')['Skillset'].apply(list).to_dict()
        
        # Get all unique skills
        all_skills = skills_df['Skillset'].unique().tolist()
        
        # Calculate skill availability considering leaves
        skill_availability = {skill: 0 for skill in all_skills}
        
        for member, skills in skill_group.items():
            # Get member's availability
            member_row = capacity_df[capacity_df['Team Member'] == member]
            if not member_row.empty:
                availability_ratio = member_row.iloc[0]['Utilization %'] / 100
                
                # For each skill this member has, add their availability
                for skill in skills:
                    if skill in skill_availability:
                        skill_availability[skill] += availability_ratio
        
        # Convert to dataframe for visualization
        skill_avail_df = pd.DataFrame({
            'Skill': list(skill_availability.keys()),
            'Availability': [min(av, 1.0) for av in skill_availability.values()],  # Cap at 1.0 (100%)
            'Coverage %': [min(av * 100, 100) for av in skill_availability.values()]  # Cap at 100%
        })
        
        # Display as a chart
        fig = px.bar(
            skill_avail_df, 
            x='Skill', 
            y='Coverage %',
            color='Coverage %',
            color_continuous_scale='RdYlGn',  # Red to Yellow to Green
            range_color=[0, 100],
            title='Skill Coverage during Sprint',
            text='Coverage %'
        )
        fig.update_layout(xaxis_title='Skill', yaxis_title='Coverage %')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)


def sprint_composition_recommendations(issues_df, skills_df, worklogs_df, leaves_df):
    """Provide recommendations for sprint composition"""
    st.subheader("🧩 Sprint Composition Recommendations")
    
    # Filter out completed issues
    open_issues = issues_df[issues_df['Status'] != 'Done'].copy() if 'Status' in issues_df.columns else issues_df.copy()
    
    # Allow user to select issues manually
    st.write("Select issues to include in the sprint:")
    
    # Add filters for better issue selection
    col1, col2 = st.columns(2)
    with col1:
        project_filter = st.multiselect(
            "Filter by Project", 
            options=open_issues['Project'].unique().tolist() if 'Project' in open_issues.columns else [],
            default=[]
        )
    with col2:
        priority_filter = st.multiselect(
            "Filter by Priority", 
            options=open_issues['Priority'].unique().tolist() if 'Priority' in open_issues.columns else [],
            default=[]
        )
    
    # Apply filters
    filtered_issues = open_issues.copy()
    if project_filter and 'Project' in filtered_issues.columns:
        filtered_issues = filtered_issues[filtered_issues['Project'].isin(project_filter)]
    if priority_filter and 'Priority' in filtered_issues.columns:
        filtered_issues = filtered_issues[filtered_issues['Priority'].isin(priority_filter)]
    
    # Display filtered issues for selection
    if not filtered_issues.empty:
        # Define required columns
        display_columns = ['Issue Key', 'Summary', 'Project', 'Priority', 'Story Points', 'Assignee']
        # Use only available columns
        available_columns = [col for col in display_columns if col in filtered_issues.columns]
        display_df = filtered_issues[available_columns].copy()
        
        # Make selection interface
        selection = st.data_editor(
            display_df,
            column_config={
                "Selected": st.column_config.CheckboxColumn("Include in Sprint", default=False)
            },
            hide_index=True,
            use_container_width=True,
            num_rows="fixed"
        )
        
        # Get selected issues
        if hasattr(selection, 'get_selected_rows'):
            selected_indices = selection.get_selected_rows()
            selected_issues = filtered_issues.iloc[list(selected_indices)] if selected_indices else pd.DataFrame()
        else:
            selected_issues = pd.DataFrame()
            st.info("Please select issues to include in the sprint.")
    else:
        st.warning("No open issues match the filters. Try adjusting filter criteria.")
        selected_issues = pd.DataFrame()
    
    # Show sprint capacity and composition analysis once issues are selected
    if not selected_issues.empty:
        st.subheader("🔍 Selected Sprint Composition Analysis")
        
        # Calculate metrics for selected issues
        total_story_points = selected_issues['Story Points'].sum() if 'Story Points' in selected_issues.columns else 0
        num_issues = len(selected_issues)
        
        # Display basic metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Issues", num_issues)
        with col2:
            st.metric("Total Story Points", f"{total_story_points:.1f}")
        
        # Distribution by assignee
        if 'Assignee' in selected_issues.columns:
            assignee_counts = selected_issues.groupby('Assignee').agg(
                issues=('Issue Key', 'count'),
                story_points=('Story Points', 'sum') if 'Story Points' in selected_issues.columns else ('Issue Key', 'count')
            ).reset_index()
            
            # Create a horizontal bar chart for workload distribution
            fig = px.bar(
                assignee_counts,
                y='Assignee',
                x='story_points',
                color='issues',
                labels={'story_points': 'Story Points', 'issues': 'Issue Count'},
                title='Workload Distribution by Assignee',
                orientation='h',
                text='story_points'
            )
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution by priority
        if 'Priority' in selected_issues.columns:
            priority_counts = selected_issues.groupby('Priority').size().reset_index(name='Count')
            fig = px.pie(
                priority_counts,
                values='Count',
                names='Priority',
                title='Issue Distribution by Priority',
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Generate AI-powered sprint composition recommendations
        st.subheader("🤖 AI-Powered Recommendations")
        
        if client is not None:
            if st.button("Generate Sprint Recommendations"):
                with st.spinner("Analyzing sprint composition and generating recommendations..."):
                    try:
                        # Create structured data for the model
                        sprint_data = {
                            "total_issues": num_issues,
                            "total_story_points": float(total_story_points),
                            "assignee_distribution": assignee_counts.to_dict(orient='records') if 'Assignee' in selected_issues.columns else []
                        }
                        
                        # Create a sample of issues for context (limit to 10 for token efficiency)
                        issue_sample = selected_issues.head(10).to_dict(orient='records')
                        
                        # Build prompt
                        prompt = f"""
                        You are a sprint planning expert. Analyze this sprint composition and provide recommendations.
                        
                        Sprint Composition Summary:
                        {json.dumps(sprint_data, indent=2)}
                        
                        Sample Issues:
                        {json.dumps(issue_sample, indent=2)}
                        
                        Provide the following:
                        1. Initial analysis of the sprint composition
                        2. Assessment of workload balance among team members
                        3. Potential risks or issues with this sprint plan
                        4. Specific, actionable recommendations to improve the sprint
                        5. Success criteria for this sprint
                        
                        Format your response as JSON with the following structure:
                        {{
                            "analysis": "Initial analysis of the sprint composition",
                            "workload_balance": "Assessment of workload balance",
                            "risks": ["Risk 1", "Risk 2", "Risk 3"],
                            "recommendations": ["Recommendation 1", "Recommendation 2", "Recommendation 3"],
                            "success_criteria": ["Criterion 1", "Criterion 2", "Criterion 3"]
                        }}
                        """
                        
                        # Get AI response
                        response = client.chat.completions.create(
                            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                            messages=[{"role": "user", "content": prompt}],
                            response_format={"type": "json_object"}
                        )
                        
                        # Parse the response
                        result = json.loads(response.choices[0].message.content)
                        
                        # Display the recommendations in a nicer format
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### Sprint Analysis")
                            st.markdown(result.get("analysis", "No analysis provided"))
                            
                            st.markdown("### Workload Balance")
                            st.markdown(result.get("workload_balance", "No workload balance assessment provided"))
                        
                        with col2:
                            st.markdown("### Potential Risks")
                            risks = result.get("risks", [])
                            for risk in risks:
                                st.markdown(f"- {risk}")
                            if not risks:
                                st.markdown("No risks identified")
                        
                        st.markdown("### Recommendations")
                        recommendations = result.get("recommendations", [])
                        for i, rec in enumerate(recommendations):
                            st.markdown(f"**{i+1}.** {rec}")
                        if not recommendations:
                            st.markdown("No recommendations provided")
                        
                        st.markdown("### Success Criteria")
                        criteria = result.get("success_criteria", [])
                        for criterion in criteria:
                            st.markdown(f"- {criterion}")
                        if not criteria:
                            st.markdown("No success criteria provided")
                    
                    except Exception as e:
                        st.error(f"Error generating recommendations: {e}")
        else:
            st.info("AI recommendations require an OpenAI API key. Add OPENAI_API_KEY to your Streamlit secrets to enable this feature.")


def sprint_simulation(issues_df, skills_df, worklogs_df, leaves_df):
    """Simulate different sprint scenarios"""
    st.subheader("🔮 Sprint Simulation")
    
    # Allow users to set up simulation parameters
    st.markdown("Set up different sprint scenarios and simulate outcomes:")
    
    # Sprint parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        sprint_duration = st.number_input("Sprint Duration (days)", min_value=5, max_value=30, value=14, step=1, key="sim_duration")
    with col2:
        team_focus = st.slider("Team Focus Factor (%)", min_value=50, max_value=100, value=80, step=5, 
                           help="Percentage of time dedicated to sprint work vs. meetings and other activities")
    with col3:
        velocity_pessimism = st.slider("Velocity Pessimism (%)", min_value=60, max_value=150, value=100, step=10,
                                  help="Adjustment factor for team velocity estimates (100% = no adjustment)")
    
    # Calculate effective capacity
    open_issues = issues_df[issues_df['Status'] != 'Done'].copy() if 'Status' in issues_df.columns else issues_df.copy()
    
    # Check if we have story points
    if 'Story Points' not in open_issues.columns:
        st.warning("Story Points column not found in issues data. Simulation requires story point estimates.")
        return
    
    # Get past velocity if available (from completed issues)
    completed_issues = issues_df[issues_df['Status'] == 'Done'].copy() if 'Status' in issues_df.columns else pd.DataFrame()
    past_velocity = 0
    if not completed_issues.empty and 'Story Points' in completed_issues.columns:
        past_velocity = completed_issues['Story Points'].sum()
        # Normalize to 2-week sprint
        past_velocity = past_velocity / 10 * 14  # Assuming data is from a 2-week period
    
    # Allow manual adjustment
    past_velocity_input = st.number_input("Past Sprint Velocity (story points)", 
                                      min_value=0.0, 
                                      value=float(past_velocity),
                                      step=1.0,
                                      help="Average story points completed in a past sprint of similar duration")
    
    # Adjust velocity for current sprint
    adjusted_velocity = past_velocity_input * (team_focus / 100) * (velocity_pessimism / 100)
    
    # Display the simulated capacity
    st.metric("Simulated Sprint Capacity (story points)", f"{adjusted_velocity:.1f}", 
           delta=f"{adjusted_velocity - past_velocity_input:.1f}",
           delta_color="normal" if adjusted_velocity >= past_velocity_input else "inverse")
    
    # Allow users to select high-priority tasks
    st.subheader("High-Priority Tasks")
    st.markdown("Select tasks that must be included in the sprint:")
    
    # Use priority to filter likely candidates
    high_priority_candidates = open_issues
    if 'Priority' in open_issues.columns:
        priority_order = {'Highest': 0, 'High': 1, 'Medium': 2, 'Low': 3, 'Lowest': 4}
        high_priority_candidates = open_issues[open_issues['Priority'].isin(['Highest', 'High'])]
    
    if high_priority_candidates.empty:
        high_priority_candidates = open_issues
    
    # Display high priority tasks for selection
    display_columns = ['Issue Key', 'Summary', 'Project', 'Priority', 'Story Points', 'Assignee']
    available_columns = [col for col in display_columns if col in high_priority_candidates.columns]
    
    high_priority_selection = st.data_editor(
        high_priority_candidates[available_columns],
        column_config={
            "Selected": st.column_config.CheckboxColumn("Must Include", default=False)
        },
        hide_index=True,
        use_container_width=True,
        num_rows="fixed"
    )
    
    # Get selected high priority issues
    if hasattr(high_priority_selection, 'get_selected_rows'):
        high_priority_indices = high_priority_selection.get_selected_rows()
        must_include_issues = high_priority_candidates.iloc[list(high_priority_indices)] if high_priority_indices else pd.DataFrame()
    else:
        must_include_issues = pd.DataFrame()
    
    # Calculate remaining capacity
    must_include_points = must_include_issues['Story Points'].sum() if not must_include_issues.empty else 0
    remaining_capacity = adjusted_velocity - must_include_points
    
    # Display remaining capacity
    st.metric("Remaining Capacity", f"{remaining_capacity:.1f} points", 
           delta=f"-{must_include_points:.1f}", 
           delta_color="inverse" if must_include_points > 0 else "normal")
    
    # Run simulation to fill the sprint
    if st.button("Run Sprint Simulation"):
        with st.spinner("Simulating optimal sprint composition..."):
            # Remove already selected issues from pool
            if not must_include_issues.empty:
                available_issues = open_issues[~open_issues['Issue Key'].isin(must_include_issues['Issue Key'])]
            else:
                available_issues = open_issues
            
            # Sort by priority and/or story points
            if 'Priority' in available_issues.columns:
                # Map priority to numeric values
                priority_map = {'Highest': 0, 'High': 1, 'Medium': 2, 'Low': 3, 'Lowest': 4}
                available_issues['Priority_Value'] = available_issues['Priority'].map(priority_map).fillna(5)
                available_issues = available_issues.sort_values(['Priority_Value', 'Story Points'])
            
            # Greedy algorithm to fill the sprint (more sophisticated algorithms could be used)
            selected_issues = must_include_issues.copy() if not must_include_issues.empty else pd.DataFrame(columns=open_issues.columns)
            current_points = must_include_points
            
            for _, issue in available_issues.iterrows():
                issue_points = issue['Story Points']
                if current_points + issue_points <= adjusted_velocity:
                    selected_issues = pd.concat([selected_issues, pd.DataFrame([issue])], ignore_index=True)
                    current_points += issue_points
            
            # Display the simulated sprint
            st.subheader("📋 Simulated Sprint Composition")
            st.write(f"Total Issues: {len(selected_issues)}")
            st.write(f"Total Story Points: {current_points:.1f} out of {adjusted_velocity:.1f} capacity")
            st.write(f"Capacity Utilization: {(current_points / adjusted_velocity * 100):.1f}%")
            
            # Display the selected issues
            st.dataframe(selected_issues[available_columns], use_container_width=True)
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Assignee distribution
                if 'Assignee' in selected_issues.columns:
                    assignee_points = selected_issues.groupby('Assignee')['Story Points'].sum().reset_index()
                    fig = px.bar(
                        assignee_points,
                        x='Assignee',
                        y='Story Points',
                        title="Workload Distribution",
                        color='Story Points',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Priority distribution
                if 'Priority' in selected_issues.columns:
                    priority_points = selected_issues.groupby('Priority')['Story Points'].sum().reset_index()
                    fig = px.pie(
                        priority_points,
                        values='Story Points',
                        names='Priority',
                        title="Distribution by Priority"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Simulation insights
            st.subheader("🧠 Simulation Insights")
            
            # Calculate some insights
            insights = []
            
            # Insight 1: Balance across team members
            if 'Assignee' in selected_issues.columns:
                assignee_counts = selected_issues.groupby('Assignee')['Story Points'].sum()
                max_assignee = assignee_counts.idxmax() if not assignee_counts.empty else "None"
                max_points = assignee_counts.max() if not assignee_counts.empty else 0
                min_assignee = assignee_counts.idxmin() if not assignee_counts.empty else "None"
                min_points = assignee_counts.min() if not assignee_counts.empty else 0
                
                balance_ratio = min_points / max_points if max_points > 0 else 1
                
                if balance_ratio < 0.5:
                    insights.append(f"⚠️ Workload imbalance: {max_assignee} has {max_points:.1f} points while {min_assignee} has only {min_points:.1f} points.")
                elif balance_ratio > 0.8:
                    insights.append(f"✅ Good workload balance across team members (within 20% variance).")
            
            # Insight 2: Priority distribution
            if 'Priority' in selected_issues.columns:
                priority_counts = selected_issues['Priority'].value_counts(normalize=True)
                high_priority_pct = 0
                if 'Highest' in priority_counts:
                    high_priority_pct += priority_counts['Highest']
                if 'High' in priority_counts:
                    high_priority_pct += priority_counts['High']
                
                if high_priority_pct < 0.3:
                    insights.append(f"⚠️ Low percentage of high-priority items ({high_priority_pct:.0%}). Consider prioritizing more critical work.")
                elif high_priority_pct > 0.7:
                    insights.append(f"⚠️ Heavy focus on high-priority items ({high_priority_pct:.0%}). Team might be constantly firefighting.")
                else:
                    insights.append(f"✅ Good balance of priority levels in the sprint.")
            
            # Insight 3: Capacity utilization
            utilization_pct = current_points / adjusted_velocity
            if utilization_pct > 0.95:
                insights.append(f"⚠️ Sprint is at {utilization_pct:.0%} capacity. Consider reducing scope to allow for unexpected work.")
            elif utilization_pct < 0.8:
                insights.append(f"🔍 Sprint is only at {utilization_pct:.0%} capacity. Consider adding more items if team confidence is high.")
            else:
                insights.append(f"✅ Good capacity utilization at {utilization_pct:.0%}.")
            
            # Display insights
            for insight in insights:
                st.markdown(insight)
            
            # Simulation recommendations
            st.subheader("🚀 Recommendations for Sprint Success")
            
            # Generate recommendations based on the simulation
            recommendations = [
                "Consider holding a risk assessment meeting before sprint starts",
                "Review dependencies between tasks and prioritize blockers",
                "Ensure each team member has a primary focus area to minimize context switching"
            ]
            
            # Additional recommendations based on insights
            if 'Assignee' in selected_issues.columns and balance_ratio < 0.5:
                recommendations.append(f"Redistribute some tasks from {max_assignee} to {min_assignee} to balance workload")
            
            if utilization_pct > 0.95:
                recommendations.append("Reduce sprint scope by 10-15% to account for unexpected work")
            
            for i, rec in enumerate(recommendations):
                st.markdown(f"**{i+1}.** {rec}")
