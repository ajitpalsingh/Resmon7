# Technical Debt and Risk Management Module
# Identifies, tracks, and manages technical debt and project risks

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


def technical_debt_risk_management(issues_df, skills_df, worklogs_df, leaves_df, debt_df=None):
    """Main function for managing technical debt and project risks"""
    
    st.markdown("### 🔍 Technical Debt & Risk Management")
    st.markdown("""
    This feature helps you identify, manage, and mitigate technical debt and project risks:
    - Visualizes technical debt accumulation and distribution
    - Forecasts risk impacts with AI-powered analysis
    - Recommends debt reduction strategies
    - Tracks risk mitigation plans and their effectiveness
    - Calculates debt-to-feature ratios and risk exposure metrics
    """)
    
    # Add tabs for different aspects of debt and risk management
    debt_tab, risk_tab, forecast_tab = st.tabs(["💻 Technical Debt", "⚠️ Risk Management", "📈 Impact Forecast"])
    
    # Check if we have the necessary data
    if issues_df is None or issues_df.empty:
        with debt_tab, risk_tab, forecast_tab:
            st.warning("No issue records available. Please upload data with task information.")
        return
    
    # Prepare technical debt data
    if debt_df is None or debt_df.empty:
        with debt_tab, risk_tab, forecast_tab:
            st.info("No dedicated technical debt records found. Extracting from issues data...")
        debt_df = extract_debt_information(issues_df)
    
    # Extract risk information
    risk_df = extract_risk_information(issues_df)
    
    # Technical Debt Management tab
    with debt_tab:
        manage_technical_debt(debt_df, issues_df, worklogs_df)
    
    # Risk Management tab
    with risk_tab:
        manage_risks(risk_df, issues_df, worklogs_df)
    
    # Impact Forecast tab
    with forecast_tab:
        forecast_impacts(debt_df, risk_df, issues_df, client)


def extract_debt_information(issues_df):
    """Extract technical debt information from issues"""
    # Create a technical debt dataframe from issues
    # Look for tasks tagged as debt or containing debt-related keywords
    
    debt_keywords = ['tech debt', 'technical debt', 'refactoring', 'code quality', 'legacy', 'deprecated']
    
    # Filter issues that might be related to technical debt
    # Start with condition for Summary column
    condition = issues_df['Summary'].str.contains('|'.join(debt_keywords), case=False, na=False)
    
    # Add condition for Description column if it exists
    if 'Description' in issues_df.columns:
        description_condition = issues_df['Description'].str.contains('|'.join(debt_keywords), case=False, na=False)
        condition = condition | description_condition
    
    debt_issues = issues_df[condition].copy()
    
    # If no matches found, create a sample structure
    if debt_issues.empty:
        # Create an empty dataframe with the necessary structure
        debt_df = pd.DataFrame(columns=[
            'Debt ID', 'Summary', 'Description', 'Project', 'Component', 
            'Created', 'Size', 'Priority', 'Status', 'Owner', 'Due Date'
        ])
        return debt_df
    
    # Rename and select relevant columns
    debt_df = debt_issues.rename(columns={
        'Issue Key': 'Debt ID',
        'Story Points': 'Size'
    })
    
    # Select relevant columns only
    relevant_columns = [
        'Debt ID', 'Summary', 'Description', 'Project', 'Component', 
        'Created', 'Size', 'Priority', 'Status', 'Assignee', 'Due Date'
    ]
    
    # Map existing columns to relevant columns
    available_columns = [col for col in relevant_columns if col in debt_df.columns]
    debt_df = debt_df[available_columns].copy()
    
    # Rename 'Assignee' to 'Owner' if it exists
    if 'Assignee' in debt_df.columns:
        debt_df = debt_df.rename(columns={'Assignee': 'Owner'})
    
    # Add missing columns with default values
    missing_columns = [col for col in relevant_columns if col not in debt_df.columns]
    for col in missing_columns:
        if col == 'Size':
            debt_df[col] = 1  # Default size
        elif col == 'Priority':
            debt_df[col] = 'Medium'  # Default priority
        else:
            debt_df[col] = None
    
    # Ensure date columns are datetime
    date_columns = ['Created', 'Due Date']
    for col in date_columns:
        if col in debt_df.columns:
            debt_df[col] = pd.to_datetime(debt_df[col], errors='coerce')
    
    return debt_df


def extract_risk_information(issues_df):
    """Extract risk information from issues"""
    # Create a risk dataframe from issues
    # Look for risk-related fields or tags
    
    risk_keywords = ['risk', 'issue', 'blocker', 'impediment', 'dependency', 'constraint']
    
    # Filter issues that might be related to risks
    # Start with condition for Summary column
    condition = issues_df['Summary'].str.contains('|'.join(risk_keywords), case=False, na=False)
    
    # Add condition for Description column if it exists
    if 'Description' in issues_df.columns:
        description_condition = issues_df['Description'].str.contains('|'.join(risk_keywords), case=False, na=False)
        condition = condition | description_condition
    
    risk_issues = issues_df[condition].copy()
    
    # If no matches found, create a sample structure
    if risk_issues.empty:
        # Create an empty dataframe with the necessary structure
        risk_df = pd.DataFrame(columns=[
            'Risk ID', 'Summary', 'Description', 'Project', 'Component', 
            'Identified Date', 'Impact', 'Probability', 'Status', 'Owner', 'Due Date'
        ])
        return risk_df
    
    # Rename and select relevant columns
    risk_df = risk_issues.rename(columns={
        'Issue Key': 'Risk ID',
        'Created': 'Identified Date'
    })
    
    # Select relevant columns only
    relevant_columns = [
        'Risk ID', 'Summary', 'Description', 'Project', 'Component', 
        'Identified Date', 'Priority', 'Status', 'Assignee', 'Due Date'
    ]
    
    # Map existing columns to relevant columns
    available_columns = [col for col in relevant_columns if col in risk_df.columns]
    risk_df = risk_df[available_columns].copy()
    
    # Rename 'Assignee' to 'Owner' if it exists
    if 'Assignee' in risk_df.columns:
        risk_df = risk_df.rename(columns={'Assignee': 'Owner'})
    
    # Add risk specific columns
    risk_df['Impact'] = risk_df['Priority'].map({
        'Highest': 'Critical',
        'High': 'High',
        'Medium': 'Medium',
        'Low': 'Low',
        'Lowest': 'Negligible'
    }).fillna('Medium')
    
    risk_df['Probability'] = 'Medium'  # Default probability
    
    # Add missing columns with default values
    missing_columns = ['Impact', 'Probability']
    for col in missing_columns:
        if col not in risk_df.columns:
            risk_df[col] = 'Medium'  # Default value
    
    # Ensure date columns are datetime
    date_columns = ['Identified Date', 'Due Date']
    for col in date_columns:
        if col in risk_df.columns:
            risk_df[col] = pd.to_datetime(risk_df[col], errors='coerce')
    
    return risk_df


def manage_technical_debt(debt_df, issues_df, worklogs_df):
    """Manage technical debt tracking and reduction"""
    st.subheader("💻 Technical Debt Management")
    
    # Handle column renaming for the Tech Debt worksheet format
    if 'Impact Level' in debt_df.columns and 'Debt ID' not in debt_df.columns:
        # This appears to be the format from the Technical Debt worksheet
        # Rename columns to match the expected format
        column_mapping = {
            'Category': 'Component',
            'Impact Level': 'Priority',
            'Estimated Resolution Time (days)': 'Size'
        }
        for old_col, new_col in column_mapping.items():
            if old_col in debt_df.columns and new_col not in debt_df.columns:
                debt_df = debt_df.rename(columns={old_col: new_col})
        
        # Add a Debt ID column if it doesn't exist
        if 'Debt ID' not in debt_df.columns:
            debt_df['Debt ID'] = [f'TECHDEBT-{i+1}' for i in range(len(debt_df))]
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics
    total_debt_items = len(debt_df)
    open_debt_items = len(debt_df[debt_df['Status'] != 'Done']) if 'Status' in debt_df.columns else total_debt_items
    
    # Check for the Size column or Estimated Resolution Time column
    if 'Size' in debt_df.columns:
        debt_points = debt_df['Size'].sum()
    elif 'Estimated Resolution Time (days)' in debt_df.columns:
        debt_points = debt_df['Estimated Resolution Time (days)'].sum()
    else:
        debt_points = 0
    
    # Calculate debt ratio (percentage of work that is technical debt)
    total_points = issues_df['Story Points'].sum() if 'Story Points' in issues_df.columns else 0
    debt_ratio = (debt_points / total_points * 100) if total_points > 0 else 0
    
    with col1:
        st.metric("Total Debt Items", total_debt_items)
    with col2:
        st.metric("Open Debt Items", open_debt_items)
    with col3:
        st.metric("Total Debt Points", f"{debt_points:.1f}")
    with col4:
        st.metric("Debt Ratio", f"{debt_ratio:.1f}%")
    
    # Display technical debt items
    st.subheader("📝 Technical Debt Items")
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.multiselect(
            "Filter by Status", 
            options=debt_df['Status'].unique().tolist() if 'Status' in debt_df.columns else [],
            default=[]
        )
    with col2:
        project_filter = st.multiselect(
            "Filter by Project", 
            options=debt_df['Project'].unique().tolist() if 'Project' in debt_df.columns else [],
            default=[]
        )
    
    # Apply filters
    filtered_debt = debt_df.copy()
    if status_filter and 'Status' in filtered_debt.columns:
        filtered_debt = filtered_debt[filtered_debt['Status'].isin(status_filter)]
    if project_filter and 'Project' in filtered_debt.columns:
        filtered_debt = filtered_debt[filtered_debt['Project'].isin(project_filter)]
    
    # Display filtered debt
    if not filtered_debt.empty:
        st.dataframe(filtered_debt, use_container_width=True)
    else:
        st.info("No technical debt items match the current filters.")
    
    # Technical Debt Visualization
    st.subheader("📊 Technical Debt Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Debt by component
        if 'Component' in debt_df.columns and not debt_df['Component'].isna().all():
            component_counts = debt_df.groupby('Component').size().reset_index(name='Count')
            component_counts = component_counts.sort_values('Count', ascending=False)
            
            fig = px.bar(
                component_counts, 
                x='Component', 
                y='Count',
                title='Technical Debt by Component',
                color='Count',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No component data available for visualization.")
    
    with col2:
        # Debt by status
        if 'Status' in debt_df.columns and not debt_df['Status'].isna().all():
            status_counts = debt_df.groupby('Status').size().reset_index(name='Count')
            
            fig = px.pie(
                status_counts, 
                values='Count', 
                names='Status',
                title='Technical Debt by Status',
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No status data available for visualization.")
    
    # Technical Debt Growth Over Time
    if 'Created' in debt_df.columns and not debt_df['Created'].isna().all():
        st.subheader("📈 Technical Debt Accumulation Over Time")
        
        # Create a time series of debt creation
        debt_df['Month'] = debt_df['Created'].dt.to_period('M')
        debt_over_time = debt_df.groupby('Month').size().reset_index(name='New Debt Items')
        debt_over_time['Month'] = debt_over_time['Month'].dt.to_timestamp()
        
        # Calculate cumulative debt
        debt_over_time['Cumulative Debt'] = debt_over_time['New Debt Items'].cumsum()
        
        # Create dual-axis chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add bar chart for new debt items
        fig.add_trace(
            go.Bar(
                x=debt_over_time['Month'],
                y=debt_over_time['New Debt Items'],
                name="New Debt Items"
            ),
            secondary_y=False
        )
        
        # Add line chart for cumulative debt
        fig.add_trace(
            go.Scatter(
                x=debt_over_time['Month'],
                y=debt_over_time['Cumulative Debt'],
                name="Cumulative Debt",
                line=dict(color='red')
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title="Technical Debt Growth Over Time",
            xaxis_title="Month",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Set y-axes titles
        fig.update_yaxes(title_text="New Debt Items", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative Debt", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Creation date information not available for debt items.")
    
    # Debt Reduction Strategies
    st.subheader("🧠 Debt Reduction Strategies")
    
    # Create example strategies
    strategies = [
        {
            "name": "Incremental Refactoring",
            "description": "Allocate 20% of each sprint to addressing technical debt items.",
            "benefit": "Steady, sustainable reduction without major disruption.",
            "cost": "Medium",
            "timeline": "3-6 months"
        },
        {
            "name": "Debt-Focused Sprint",
            "description": "Dedicate an entire sprint exclusively to eliminating technical debt.",
            "benefit": "Rapid reduction of debt in critical areas.",
            "cost": "High",
            "timeline": "Immediate (1 sprint)"
        },
        {
            "name": "Continuous Integration Enforcement",
            "description": "Implement stricter code quality gates in CI pipeline.",
            "benefit": "Prevents new debt accumulation.",
            "cost": "Low",
            "timeline": "1-2 sprints to implement"
        }
    ]
    
    # Display strategies as expandable sections
    for i, strategy in enumerate(strategies):
        with st.expander(f"Strategy {i+1}: {strategy['name']}"):
            st.markdown(f"**Description:** {strategy['description']}")
            st.markdown(f"**Benefit:** {strategy['benefit']}")
            st.markdown(f"**Cost:** {strategy['cost']}")
            st.markdown(f"**Timeline:** {strategy['timeline']}")
            
            if st.button("Simulate Impact", key=f"simulate_{i}"):
                with st.spinner("Simulating debt reduction impact..."):
                    st.success("Simulation complete")
                    
                    # Display simulated metrics based on the strategy
                    if i == 0:  # Incremental Refactoring
                        reduced_items = int(open_debt_items * 0.4)  # 40% reduction
                        reduced_ratio = debt_ratio * 0.7  # 30% improvement
                    elif i == 1:  # Debt-Focused Sprint
                        reduced_items = int(open_debt_items * 0.7)  # 70% reduction
                        reduced_ratio = debt_ratio * 0.5  # 50% improvement
                    else:  # Continuous Integration
                        reduced_items = int(open_debt_items * 0.2)  # 20% reduction
                        reduced_ratio = debt_ratio * 0.8  # 20% improvement
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Projected Open Debt Items", 
                            open_debt_items - reduced_items,
                            delta=f"-{reduced_items}",
                            delta_color="inverse"
                        )
                    with col2:
                        st.metric(
                            "Projected Debt Ratio", 
                            f"{reduced_ratio:.1f}%",
                            delta=f"-{(debt_ratio - reduced_ratio):.1f}%",
                            delta_color="inverse"
                        )
    
    # AI-powered Debt Reduction Recommendations
    if client is not None:
        st.subheader("🤖 AI-Powered Debt Analysis")
        
        if st.button("Generate Debt Analysis"):
            with st.spinner("Analyzing technical debt patterns..."):
                try:
                    # Prepare data for analysis
                    debt_summary = {
                        "total_items": total_debt_items,
                        "open_items": open_debt_items,
                        "debt_points": debt_points,
                        "debt_ratio": debt_ratio
                    }
                    
                    # Add component data if available
                    if 'Component' in debt_df.columns and not debt_df['Component'].isna().all():
                        debt_summary["components"] = debt_df['Component'].value_counts().to_dict()
                    else:
                        debt_summary["components"] = {}
                    
                    # Sample debt items limited to a few for API efficiency
                    sample_items = debt_df.head(5).to_dict(orient='records') if not debt_df.empty else []
                    
                    # Build prompt
                    prompt = f"""
                    You are a technical debt consultant. Analyze this technical debt summary and provide actionable recommendations.
                    
                    Debt Summary:
                    {json.dumps(debt_summary, indent=2)}
                    
                    Sample Debt Items:
                    {json.dumps(sample_items, indent=2)}
                    
                    Provide the following:
                    1. Brief analysis of the current technical debt situation
                    2. Recommended prioritization approach
                    3. Three specific, actionable recommendations for debt reduction
                    4. Suggested metrics to track debt management progress
                    
                    Format your response as JSON with the following structure:
                    {{
                        "analysis": "Brief situation analysis",
                        "prioritization": "Recommended prioritization approach",
                        "recommendations": [
                            {{"title": "Recommendation 1", "details": "Description", "impact": "Expected impact"}},
                            {{"title": "Recommendation 2", "details": "Description", "impact": "Expected impact"}},
                            {{"title": "Recommendation 3", "details": "Description", "impact": "Expected impact"}}
                        ],
                        "metrics": ["Metric 1", "Metric 2", "Metric 3"]
                    }}
                    """
                    
                    # Send to OpenAI API
                    response = client.chat.completions.create(
                        model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. Do not change this unless explicitly requested by the user
                        messages=[{"role": "system", "content": "You are an AI assistant that helps with technical debt analysis."}, 
                                 {"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        temperature=0.2,
                    )
                    
                    # Parse the response
                    ai_analysis = json.loads(response.choices[0].message.content)
                    
                    # Display AI analysis
                    st.markdown("### 🔍 AI Analysis")
                    st.markdown(ai_analysis.get("analysis", "No analysis provided"))
                    
                    st.markdown("### 📋 Recommended Prioritization")
                    st.info(ai_analysis.get("prioritization", "No prioritization approach provided"))
                    
                    st.markdown("### 🚀 Recommended Actions")
                    for rec in ai_analysis.get("recommendations", []):
                        with st.expander(rec.get("title", "Recommendation")):
                            st.markdown(f"**Details:** {rec.get('details', '')}")
                            st.markdown(f"**Impact:** {rec.get('impact', '')}")
                    
                    st.markdown("### 📊 Suggested Metrics")
                    metrics = ai_analysis.get("metrics", [])
                    for i, metric in enumerate(metrics):
                        st.markdown(f"{i+1}. {metric}")
                    
                except Exception as e:
                    st.error(f"Could not generate AI analysis: {e}")


def manage_risks(risk_df, issues_df, worklogs_df):
    """Manage project risks"""
    st.subheader("⚠️ Risk Management")
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics
    total_risks = len(risk_df)
    open_risks = len(risk_df[risk_df['Status'] != 'Done']) if 'Status' in risk_df.columns else total_risks
    high_impact_risks = len(risk_df[risk_df['Impact'] == 'High']) if 'Impact' in risk_df.columns else 0
    critical_risks = len(risk_df[risk_df['Impact'] == 'Critical']) if 'Impact' in risk_df.columns else 0
    
    with col1:
        st.metric("Total Risks", total_risks)
    with col2:
        st.metric("Open Risks", open_risks)
    with col3:
        st.metric("High Impact", high_impact_risks)
    with col4:
        st.metric("Critical", critical_risks)
    
    # Display risk items
    st.subheader("📝 Risk Register")
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.multiselect(
            "Filter by Status", 
            options=risk_df['Status'].unique().tolist() if 'Status' in risk_df.columns else [],
            default=[]
        )
    with col2:
        impact_filter = st.multiselect(
            "Filter by Impact", 
            options=risk_df['Impact'].unique().tolist() if 'Impact' in risk_df.columns else [],
            default=[]
        )
    with col3:
        project_filter = st.multiselect(
            "Filter by Project", 
            options=risk_df['Project'].unique().tolist() if 'Project' in risk_df.columns else [],
            default=[]
        )
    
    # Apply filters
    filtered_risks = risk_df.copy()
    if status_filter and 'Status' in filtered_risks.columns:
        filtered_risks = filtered_risks[filtered_risks['Status'].isin(status_filter)]
    if impact_filter and 'Impact' in filtered_risks.columns:
        filtered_risks = filtered_risks[filtered_risks['Impact'].isin(impact_filter)]
    if project_filter and 'Project' in filtered_risks.columns:
        filtered_risks = filtered_risks[filtered_risks['Project'].isin(project_filter)]
    
    # Display filtered risks
    if not filtered_risks.empty:
        st.dataframe(filtered_risks, use_container_width=True)
    else:
        st.info("No risks match the current filters.")
    
    # Risk Matrix
    st.subheader("🎯 Risk Matrix")
    
    if 'Impact' in risk_df.columns and 'Probability' in risk_df.columns:
        # Create a mapping for impact and probability to numeric values
        impact_map = {
            'Critical': 5,
            'High': 4,
            'Medium': 3,
            'Low': 2,
            'Negligible': 1
        }
        
        probability_map = {
            'Very High': 5,
            'High': 4,
            'Medium': 3,
            'Low': 2,
            'Very Low': 1
        }
        
        # Create a copy with numeric values
        matrix_df = risk_df.copy()
        matrix_df['Impact_Value'] = matrix_df['Impact'].map(impact_map).fillna(3)
        matrix_df['Probability_Value'] = matrix_df['Probability'].map(probability_map).fillna(3)
        
        # Count risks by impact and probability
        risk_counts = matrix_df.groupby(['Impact_Value', 'Probability_Value']).size().reset_index(name='Count')
        
        # Create a heatmap
        fig = px.density_heatmap(
            risk_counts,
            x='Probability_Value',
            y='Impact_Value',
            z='Count',
            title='Risk Matrix',
            labels={
                'Probability_Value': 'Probability',
                'Impact_Value': 'Impact',
                'Count': 'Number of Risks'
            },
            color_continuous_scale=[
                [0, 'white'],
                [0.01, 'lightyellow'],
                [0.4, 'orange'],
                [0.7, 'orangered'],
                [1.0, 'darkred']
            ],
        )
        
        # Add custom tick labels
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=[1, 2, 3, 4, 5],
                ticktext=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=[1, 2, 3, 4, 5],
                ticktext=['Negligible', 'Low', 'Medium', 'High', 'Critical']
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation
        with st.expander("Risk Matrix Explanation"):
            st.markdown("""
            **Risk Matrix Usage:**
            - The risk matrix plots risks by their probability (x-axis) and impact (y-axis).
            - The color intensity represents the number of risks in each cell.
            - Risks in the upper-right corner (high impact, high probability) require immediate attention.
            - Risks in the lower-left corner (low impact, low probability) can be monitored with less urgency.
            
            **Risk Levels:**
            - 🔴 **Critical Risk Zone (dark red)**: High impact and high probability risks that require immediate mitigation.
            - 🟠 **High Risk Zone (orange)**: Significant risks that should be actively managed.
            - 🟡 **Medium Risk Zone (yellow)**: Moderate risks that should be monitored.
            - ⚪ **Low Risk Zone (white)**: Low impact and low probability risks that can be accepted or monitored periodically.
            """)
    else:
        st.info("Impact and probability information not available for risk matrix visualization.")
    
    # Risk Mitigation Planning
    st.subheader("🧩 Risk Mitigation Planning")
    
    # Sample risk for detailed planning
    if not risk_df.empty:
        selected_risk_id = st.selectbox(
            "Select risk for mitigation planning:",
            options=risk_df['Risk ID'].tolist(),
            format_func=lambda x: f"{x} - {risk_df[risk_df['Risk ID'] == x]['Summary'].iloc[0] if len(risk_df[risk_df['Risk ID'] == x]) > 0 else ''}"
        )
        
        selected_risk = risk_df[risk_df['Risk ID'] == selected_risk_id].iloc[0]
        
        st.markdown(f"**Risk ID:** {selected_risk['Risk ID']}")
        st.markdown(f"**Summary:** {selected_risk.get('Summary', 'N/A')}")
        st.markdown(f"**Impact:** {selected_risk.get('Impact', 'N/A')}")
        st.markdown(f"**Probability:** {selected_risk.get('Probability', 'N/A')}")
        
        # Mitigation strategy planning
        st.markdown("### Mitigation Strategy")
        
        # Define mitigation strategies
        mitigation_strategies = [
            "Accept - Monitor the risk without taking immediate action",
            "Avoid - Eliminate the risk by changing the approach",
            "Transfer - Shift responsibility to a third party",
            "Mitigate - Take action to reduce impact or probability"
        ]
        
        selected_strategy = st.selectbox("Select mitigation strategy:", mitigation_strategies)
        
        # Action plan
        st.markdown("### Action Plan")
        action_description = st.text_area("Action plan description:", "")
        
        col1, col2 = st.columns(2)
        with col1:
            action_owner = st.text_input("Responsible person:")
        with col2:
            due_date = st.date_input("Due date:")
        
        # Risk status tracking
        st.markdown("### Status Tracking")
        
        col1, col2 = st.columns(2)
        with col1:
            status = st.selectbox("Current status:", ["Open", "In Progress", "Mitigated", "Closed"])
        with col2:
            new_probability = st.selectbox("Revised probability:", ["Very High", "High", "Medium", "Low", "Very Low"])
        
        # Save button
        if st.button("Save Mitigation Plan"):
            st.success("Mitigation plan saved (simulation only)")
            
            # Create and display the updated risk entry
            st.markdown("### Updated Risk Entry")
            updated_risk = selected_risk.copy()
            updated_risk['Mitigation Strategy'] = selected_strategy
            updated_risk['Action Plan'] = action_description
            updated_risk['Action Owner'] = action_owner
            updated_risk['Action Due Date'] = due_date
            updated_risk['Status'] = status
            updated_risk['Revised Probability'] = new_probability
            
            # Display the updated risk as a styled table
            styled_data = pd.DataFrame([updated_risk]).T.reset_index()
            styled_data.columns = ['Field', 'Value']
            st.table(styled_data)
    else:
        st.info("No risks available for detailed planning.")


def forecast_impacts(debt_df, risk_df, issues_df, client=None):
    """Forecast impacts of technical debt and risks"""
    st.subheader("📈 Impact Forecast")
    
    # Choose what to forecast
    forecast_type = st.radio(
        "What would you like to forecast?",
        ["Technical Debt Impact", "Risk Exposure", "Combined Impact"]
    )
    
    if forecast_type == "Technical Debt Impact":
        forecast_debt_impact(debt_df, issues_df, client)
    elif forecast_type == "Risk Exposure":
        forecast_risk_exposure(risk_df, issues_df, client)
    else:
        forecast_combined_impact(debt_df, risk_df, issues_df, client)


def forecast_debt_impact(debt_df, issues_df, client=None):
    """Forecast the impact of technical debt"""
    # Simulation parameters
    st.markdown("### Technical Debt Impact Simulation")
    
    col1, col2 = st.columns(2)
    with col1:
        simulation_months = st.slider("Simulation timeframe (months)", 1, 24, 6)
    with col2:
        debt_reduction_effort = st.slider("Monthly debt reduction effort (%)", 0, 50, 10)
    
    debt_growth_rate = st.slider("Monthly debt growth rate without intervention (%)", 0, 30, 15)
    
    # Calculate initial metrics
    initial_debt_items = len(debt_df)
    initial_debt_points = debt_df['Size'].sum() if 'Size' in debt_df.columns else initial_debt_items
    
    # Run simulation
    if st.button("Run Debt Impact Simulation"):
        # Create simulation data
        months = list(range(simulation_months + 1))
        no_action_debt = [initial_debt_points * (1 + debt_growth_rate/100)**month for month in months]
        
        # With reduction efforts
        with_action_debt = [initial_debt_points]
        current_debt = initial_debt_points
        for month in range(1, simulation_months + 1):
            # New debt added this month
            new_debt = current_debt * (debt_growth_rate/100)
            # Debt reduced this month
            reduced_debt = current_debt * (debt_reduction_effort/100)
            # Net change
            current_debt = current_debt + new_debt - reduced_debt
            with_action_debt.append(current_debt)
        
        # Create debt forecast visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=months,
            y=no_action_debt,
            mode='lines+markers',
            name='No Action',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=months,
            y=with_action_debt,
            mode='lines+markers',
            name=f'{debt_reduction_effort}% Monthly Reduction',
            line=dict(color='green', width=2)
        ))
        
        # Calculate crossover point (when debt starts decreasing)
        crossover = None
        for i in range(1, len(with_action_debt)):
            if with_action_debt[i] < with_action_debt[i-1]:
                crossover = i
                break
        
        if crossover is not None:
            fig.add_vline(
                x=crossover, 
                line_width=2, 
                line_dash="dash", 
                line_color="blue",
                annotation_text="Debt Reduction Point",
                annotation_position="top right"
            )
        
        fig.update_layout(
            title="Technical Debt Forecast",
            xaxis_title="Months",
            yaxis_title="Technical Debt Points",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Debt impact metrics
        st.markdown("### Impact Metrics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            final_without_action = no_action_debt[-1]
            final_with_action = with_action_debt[-1]
            debt_difference = final_without_action - final_with_action
            
            st.metric(
                "Debt Points Saved", 
                f"{debt_difference:.1f}",
                delta=f"{debt_difference/initial_debt_points*100:.1f}% of initial"
            )
        
        with col2:
            if final_with_action < initial_debt_points:
                reduction = initial_debt_points - final_with_action
                st.metric(
                    "Net Debt Reduction", 
                    f"{reduction:.1f}",
                    delta=f"-{reduction/initial_debt_points*100:.1f}%",
                    delta_color="inverse"
                )
            else:
                increase = final_with_action - initial_debt_points
                st.metric(
                    "Net Debt Change", 
                    f"+{increase:.1f}",
                    delta=f"+{increase/initial_debt_points*100:.1f}%",
                    delta_color="normal"
                )
        
        with col3:
            if crossover is not None:
                st.metric(
                    "Months to Reduction", 
                    crossover,
                    delta=f"After {crossover} months debt decreases"
                )
            else:
                st.metric(
                    "Debt Reduction Point", 
                    "Not reached",
                    delta="Increase effort to reach reduction",
                    delta_color="normal"
                )
        
        # Business impact analysis
        st.markdown("### 💰 Business Impact Analysis")
        
        # Define cost per debt point (story point)
        avg_dev_cost_per_day = 500  # Average developer cost per day
        avg_story_points_per_day = 0.5  # Average story points completed per developer day
        cost_per_debt_point = avg_dev_cost_per_day / avg_story_points_per_day
        
        # Calculate metrics
        debt_cost_without_action = final_without_action * cost_per_debt_point
        debt_cost_with_action = final_with_action * cost_per_debt_point
        cost_savings = debt_cost_without_action - debt_cost_with_action
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Debt Cost Without Action", 
                f"${debt_cost_without_action:,.0f}",
                delta=f"+{debt_cost_without_action/initial_debt_points/cost_per_debt_point*100:.1f}% growth"
            )
        with col2:
            st.metric(
                "Potential Cost Savings", 
                f"${cost_savings:,.0f}",
                delta=f"{cost_savings/debt_cost_without_action*100:.1f}% of potential cost",
                delta_color="inverse"
            )
        
        # Developer productivity impact
        maintenance_overhead_factor = 0.05 * (final_without_action / initial_debt_points)
        maintenance_time_percent = maintenance_overhead_factor * 100
        
        st.markdown(f"**Developer Productivity Impact:** With no action, maintenance overhead could reach approximately **{maintenance_time_percent:.1f}%** of developer time by month {simulation_months}.")
        
        # Software quality impact
        quality_impact_factor = min(0.9, 0.1 * (final_without_action / initial_debt_points))
        additional_bugs = int(quality_impact_factor * 50)  # Arbitrary scale for visualization
        
        st.markdown(f"**Software Quality Impact:** Technical debt growth could lead to approximately **{additional_bugs}** additional bugs or issues by month {simulation_months}.")
        
        # AI-powered impact analysis
        if client is not None:
            st.markdown("### 🤖 AI-Powered Impact Analysis")
            
            with st.spinner("Generating AI analysis of technical debt impact..."):
                try:
                    # Prepare simulation data for the analysis
                    simulation_data = {
                        "initial_debt": initial_debt_points,
                        "simulation_months": simulation_months,
                        "debt_reduction_effort": debt_reduction_effort,
                        "debt_growth_rate": debt_growth_rate,
                        "no_action_final_debt": final_without_action,
                        "with_action_final_debt": final_with_action,
                        "crossover_month": crossover,
                        "productivity_impact": maintenance_time_percent,
                        "estimated_cost_savings": cost_savings
                    }
                    
                    # Build prompt
                    prompt = f"""
                    You are a technical debt consultant analyzing simulation results. 
                    Review this technical debt impact simulation data and provide strategic insights:
                    
                    {json.dumps(simulation_data, indent=2)}
                    
                    Provide the following:
                    1. Assessment of the current technical debt trajectory
                    2. Business impact analysis (time to market, maintenance costs, innovation capacity)
                    3. Recommendations to optimize debt reduction strategy
                    4. Key performance indicators to track
                    
                    Format your response as JSON with the following structure:
                    {{
                        "assessment": "Assessment of the technical debt trajectory",
                        "business_impact": "Analysis of business impacts",
                        "strategy_recommendations": [
                            {{"title": "Recommendation 1", "rationale": "Why this matters", "implementation": "How to implement"}},
                            {{"title": "Recommendation 2", "rationale": "Why this matters", "implementation": "How to implement"}},
                            {{"title": "Recommendation 3", "rationale": "Why this matters", "implementation": "How to implement"}}
                        ],
                        "key_indicators": ["KPI 1", "KPI 2", "KPI 3"]
                    }}
                    """
                    
                    # Send to OpenAI API
                    response = client.chat.completions.create(
                        model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. Do not change this unless explicitly requested by the user
                        messages=[{"role": "system", "content": "You are an AI assistant that helps with technical debt analysis."}, 
                                 {"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        temperature=0.2,
                    )
                    
                    # Parse the response
                    ai_analysis = json.loads(response.choices[0].message.content)
                    
                    # Display AI analysis
                    st.markdown("#### Debt Trajectory Assessment")
                    st.info(ai_analysis.get("assessment", "No assessment provided"))
                    
                    st.markdown("#### Business Impact Analysis")
                    st.warning(ai_analysis.get("business_impact", "No business impact analysis provided"))
                    
                    st.markdown("#### Strategy Recommendations")
                    for i, rec in enumerate(ai_analysis.get("strategy_recommendations", [])):
                        with st.expander(f"{i+1}. {rec.get('title', 'Recommendation')}"):
                            st.markdown(f"**Why it matters:** {rec.get('rationale', '')}")
                            st.markdown(f"**Implementation:** {rec.get('implementation', '')}")
                    
                    st.markdown("#### Key Performance Indicators")
                    kpis = ai_analysis.get("key_indicators", [])
                    for i, kpi in enumerate(kpis):
                        st.markdown(f"- {kpi}")
                    
                except Exception as e:
                    st.error(f"Could not generate AI analysis: {e}")


def forecast_risk_exposure(risk_df, issues_df, client=None):
    """Forecast risk exposure over time"""
    st.markdown("### Risk Exposure Simulation")
    
    # Simulation parameters
    col1, col2 = st.columns(2)
    with col1:
        simulation_months = st.slider("Risk simulation timeframe (months)", 1, 24, 6)
    with col2:
        mitigation_effectiveness = st.slider("Mitigation effectiveness (%)", 0, 100, 70)
    
    # Get initial risk counts by impact level
    impact_levels = ['Critical', 'High', 'Medium', 'Low', 'Negligible']
    initial_risks = {}
    for level in impact_levels:
        count = len(risk_df[risk_df['Impact'] == level]) if 'Impact' in risk_df.columns else 0
        initial_risks[level] = count
    
    # Risk emergence rates (new risks per month)
    risk_emergence = {
        'Critical': st.slider("New Critical risks per month", 0, 5, 1),
        'High': st.slider("New High risks per month", 0, 10, 2),
        'Medium': st.slider("New Medium risks per month", 0, 15, 4),
        'Low': st.slider("New Low risks per month", 0, 20, 6),
        'Negligible': st.slider("New Negligible risks per month", 0, 20, 5)
    }
    
    # Run simulation
    if st.button("Run Risk Exposure Simulation"):
        # Create simulation data structures
        months = list(range(simulation_months + 1))
        
        # No mitigation scenario
        no_mitigation_risks = {level: [initial_risks.get(level, 0)] for level in impact_levels}
        for level in impact_levels:
            current = initial_risks.get(level, 0)
            for _ in range(simulation_months):
                current += risk_emergence.get(level, 0)
                no_mitigation_risks[level].append(current)
        
        # With mitigation scenario
        with_mitigation_risks = {level: [initial_risks.get(level, 0)] for level in impact_levels}
        for level in impact_levels:
            current = initial_risks.get(level, 0)
            for _ in range(simulation_months):
                new_risks = risk_emergence.get(level, 0)
                mitigated_risks = int(current * (mitigation_effectiveness / 100))
                current = current + new_risks - mitigated_risks
                current = max(0, current)  # Ensure no negative values
                with_mitigation_risks[level].append(current)
        
        # Calculate total risks over time
        total_no_mitigation = [sum(no_mitigation_risks[level][i] for level in impact_levels) for i in range(simulation_months + 1)]
        total_with_mitigation = [sum(with_mitigation_risks[level][i] for level in impact_levels) for i in range(simulation_months + 1)]
        
        # Calculate weighted risk exposure (higher weights for more severe risks)
        risk_weights = {'Critical': 5, 'High': 4, 'Medium': 3, 'Low': 2, 'Negligible': 1}
        weighted_no_mitigation = [sum(no_mitigation_risks[level][i] * risk_weights[level] for level in impact_levels) for i in range(simulation_months + 1)]
        weighted_with_mitigation = [sum(with_mitigation_risks[level][i] * risk_weights[level] for level in impact_levels) for i in range(simulation_months + 1)]
        
        # Create visualizations
        st.markdown("#### Total Risk Count Forecast")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=months,
            y=total_no_mitigation,
            mode='lines+markers',
            name='No Mitigation',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=months,
            y=total_with_mitigation,
            mode='lines+markers',
            name=f'With {mitigation_effectiveness}% Mitigation',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title="Risk Count Forecast",
            xaxis_title="Months",
            yaxis_title="Total Risks",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Weighted risk exposure
        st.markdown("#### Weighted Risk Exposure Forecast")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=months,
            y=weighted_no_mitigation,
            mode='lines+markers',
            name='No Mitigation',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=months,
            y=weighted_with_mitigation,
            mode='lines+markers',
            name=f'With {mitigation_effectiveness}% Mitigation',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title="Weighted Risk Exposure Forecast",
            xaxis_title="Months",
            yaxis_title="Risk Exposure Score",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display forecast metrics
        st.markdown("### 📊 Risk Exposure Metrics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            final_risks_no_mitigation = total_no_mitigation[-1]
            final_risks_with_mitigation = total_with_mitigation[-1]
            risks_avoided = final_risks_no_mitigation - final_risks_with_mitigation
            
            st.metric(
                "Risks Avoided", 
                int(risks_avoided),
                delta=f"{risks_avoided/final_risks_no_mitigation*100:.1f}% reduction",
                delta_color="inverse"
            )
        
        with col2:
            final_exposure_no_mitigation = weighted_no_mitigation[-1]
            final_exposure_with_mitigation = weighted_with_mitigation[-1]
            exposure_reduction = final_exposure_no_mitigation - final_exposure_with_mitigation
            
            st.metric(
                "Exposure Reduction", 
                int(exposure_reduction),
                delta=f"{exposure_reduction/final_exposure_no_mitigation*100:.1f}% reduction",
                delta_color="inverse"
            )
        
        with col3:
            initial_exposure = weighted_no_mitigation[0]
            exposure_growth_factor = final_exposure_no_mitigation / initial_exposure if initial_exposure > 0 else 0
            
            st.metric(
                "Potential Exposure Growth", 
                f"{exposure_growth_factor:.1f}x",
                delta=f"+{(exposure_growth_factor-1)*100:.1f}% without mitigation"
            )
        
        # Risk distribution evolution
        st.markdown("### Risk Distribution Evolution")
        
        # Create stacked bar chart data
        stacked_data = []
        months_to_show = [0, simulation_months // 2, simulation_months]  # Start, middle, end
        scenarios = ["Initial", "No Mitigation", "With Mitigation"]
        
        for i, month in enumerate(months_to_show):
            for level in impact_levels:
                if i == 0:  # Initial state
                    stacked_data.append({
                        "Month": "Initial",
                        "Impact": level,
                        "Count": initial_risks.get(level, 0),
                        "Scenario": "Initial"
                    })
                else:  # Later months
                    stacked_data.append({
                        "Month": f"Month {month}",
                        "Impact": level,
                        "Count": no_mitigation_risks[level][month],
                        "Scenario": "No Mitigation"
                    })
                    stacked_data.append({
                        "Month": f"Month {month}",
                        "Impact": level,
                        "Count": with_mitigation_risks[level][month],
                        "Scenario": "With Mitigation"
                    })
        
        stacked_df = pd.DataFrame(stacked_data)
        
        # Create stacked bar chart
        if not stacked_df.empty:
            fig = px.bar(
                stacked_df,
                x="Month",
                y="Count",
                color="Impact",
                barmode="stack",
                facet_col="Scenario",
                color_discrete_map={
                    "Critical": "darkred",
                    "High": "red",
                    "Medium": "orange",
                    "Low": "gold",
                    "Negligible": "lightgreen"
                },
                title="Risk Distribution Evolution"
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Project impact analysis
        st.markdown("### 🏗️ Project Impact Analysis")
        
        # Create a simple model for project impact
        # Higher risk exposure correlates with increased schedule delays, cost overruns, and quality issues
        
        # Calculate metrics based on weighted risk exposure
        initial_project_overrun_pct = 10  # Base project overrun percentage
        risk_multiplier = weighted_no_mitigation[-1] / initial_exposure if initial_exposure > 0 else 1
        risk_multiplier_mitigated = weighted_with_mitigation[-1] / initial_exposure if initial_exposure > 0 else 1
        
        # Schedule impact
        schedule_impact_pct = initial_project_overrun_pct * risk_multiplier
        schedule_impact_mitigated_pct = initial_project_overrun_pct * risk_multiplier_mitigated
        schedule_improvement = schedule_impact_pct - schedule_impact_mitigated_pct
        
        # Cost impact (slightly amplified from schedule)
        cost_impact_pct = initial_project_overrun_pct * 1.2 * risk_multiplier
        cost_impact_mitigated_pct = initial_project_overrun_pct * 1.2 * risk_multiplier_mitigated
        cost_improvement = cost_impact_pct - cost_impact_mitigated_pct
        
        # Quality impact (different scale)
        quality_impact = min(100, max(0, 100 - (weighted_no_mitigation[-1] / 5)))  # Higher is better
        quality_impact_mitigated = min(100, max(0, 100 - (weighted_with_mitigation[-1] / 5)))
        quality_improvement = quality_impact_mitigated - quality_impact
        
        # Display impact metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Schedule Impact", 
                f"{schedule_impact_pct:.1f}% Delay",
                delta=f"{schedule_improvement:.1f}% improvement with mitigation",
                delta_color="inverse"
            )
        with col2:
            st.metric(
                "Budget Impact", 
                f"{cost_impact_pct:.1f}% Overrun",
                delta=f"{cost_improvement:.1f}% improvement with mitigation",
                delta_color="inverse"
            )
        with col3:
            st.metric(
                "Quality Impact", 
                f"{quality_impact:.1f}%",
                delta=f"{quality_improvement:.1f}% improvement with mitigation",
                delta_color="inverse"
            )
        
        # AI-powered risk analysis
        if client is not None:
            st.markdown("### 🤖 AI-Powered Risk Analysis")
            
            with st.spinner("Generating AI analysis of risk exposure..."):
                try:
                    # Prepare simulation data for the analysis
                    simulation_data = {
                        "initial_risks": initial_risks,
                        "simulation_months": simulation_months,
                        "mitigation_effectiveness": mitigation_effectiveness,
                        "final_no_mitigation": {
                            "total": total_no_mitigation[-1],
                            "weighted": weighted_no_mitigation[-1]
                        },
                        "final_with_mitigation": {
                            "total": total_with_mitigation[-1],
                            "weighted": weighted_with_mitigation[-1]
                        },
                        "project_impacts": {
                            "schedule": {
                                "no_mitigation": schedule_impact_pct,
                                "with_mitigation": schedule_impact_mitigated_pct
                            },
                            "cost": {
                                "no_mitigation": cost_impact_pct,
                                "with_mitigation": cost_impact_mitigated_pct
                            },
                            "quality": {
                                "no_mitigation": quality_impact,
                                "with_mitigation": quality_impact_mitigated
                            }
                        }
                    }
                    
                    # Build prompt
                    prompt = f"""
                    You are a risk management consultant analyzing simulation results. 
                    Review this risk exposure simulation data and provide strategic insights:
                    
                    {json.dumps(simulation_data, indent=2)}
                    
                    Provide the following:
                    1. Assessment of the risk exposure trajectory
                    2. Potential business consequences of unmitigated risks
                    3. Recommendations to improve risk management strategy
                    4. Key performance indicators to track
                    
                    Format your response as JSON with the following structure:
                    {{
                        "assessment": "Assessment of the risk trajectory",
                        "consequences": "Potential business consequences",
                        "recommendations": [
                            {{"title": "Recommendation 1", "details": "Description", "priority": "High/Medium/Low"}},
                            {{"title": "Recommendation 2", "details": "Description", "priority": "High/Medium/Low"}},
                            {{"title": "Recommendation 3", "details": "Description", "priority": "High/Medium/Low"}}
                        ],
                        "kpis": ["KPI 1", "KPI 2", "KPI 3"]
                    }}
                    """
                    
                    # Send to OpenAI API
                    response = client.chat.completions.create(
                        model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. Do not change this unless explicitly requested by the user
                        messages=[{"role": "system", "content": "You are an AI assistant that helps with risk analysis."}, 
                                 {"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        temperature=0.2,
                    )
                    
                    # Parse the response
                    ai_analysis = json.loads(response.choices[0].message.content)
                    
                    # Display AI analysis
                    st.markdown("#### Risk Trajectory Assessment")
                    st.info(ai_analysis.get("assessment", "No assessment provided"))
                    
                    st.markdown("#### Business Consequences")
                    st.warning(ai_analysis.get("consequences", "No business consequences provided"))
                    
                    st.markdown("#### Risk Strategy Recommendations")
                    for rec in ai_analysis.get("recommendations", []):
                        priority = rec.get("priority", "Medium")
                        priority_color = "red" if priority == "High" else "orange" if priority == "Medium" else "green"
                        with st.expander(f"{rec.get('title', 'Recommendation')} ({priority} Priority)"):
                            st.markdown(f"**Details:** {rec.get('details', '')}")
                    
                    st.markdown("#### Key Performance Indicators")
                    kpis = ai_analysis.get("kpis", [])
                    for i, kpi in enumerate(kpis):
                        st.markdown(f"- {kpi}")
                    
                except Exception as e:
                    st.error(f"Could not generate AI analysis: {e}")


def forecast_combined_impact(debt_df, risk_df, issues_df, client=None):
    """Forecast combined impact of technical debt and risks"""
    st.markdown("### Combined Technical Debt & Risk Impact Forecast")
    
    # Simulation parameters
    col1, col2 = st.columns(2)
    with col1:
        simulation_months = st.slider("Combined simulation timeframe (months)", 1, 24, 12)
    with col2:
        action_effectiveness = st.slider("Improvement action effectiveness (%)", 0, 100, 60)
    
    # Initial metrics
    initial_debt_points = debt_df['Size'].sum() if 'Size' in debt_df.columns and not debt_df.empty else len(debt_df)
    initial_debt_ratio = 0.2  # Assume 20% of effort is debt-related
    
    # Risk metrics
    risk_weights = {'Critical': 5, 'High': 4, 'Medium': 3, 'Low': 2, 'Negligible': 1}
    weighted_risk_exposure = sum(
        len(risk_df[risk_df['Impact'] == level]) * risk_weights.get(level, 1) 
        for level in risk_weights.keys() if 'Impact' in risk_df.columns
    )
    
    # Setup project simulation parameters
    project_baseline_velocity = 30  # Story points per month baseline
    monthly_velocity_debt_impact = -0.02  # 2% reduction per month from debt
    monthly_velocity_risk_impact = -0.01  # 1% reduction per month from risk
    
    col1, col2 = st.columns(2)
    with col1:
        tech_debt_growth = st.slider("Monthly tech debt growth rate (%)", 0, 30, 15)
    with col2:
        risk_growth = st.slider("Monthly risk exposure growth (%)", 0, 30, 10)
    
    productivity_decline_rate = st.slider("Monthly productivity decline without action (%)", 0, 5, 2)
    
    # Run simulation
    if st.button("Run Combined Impact Simulation"):
        # Time series
        months = list(range(simulation_months + 1))
        
        # Technical debt projection
        no_action_debt = [initial_debt_points * (1 + tech_debt_growth/100)**month for month in months]
        with_action_debt = [initial_debt_points]
        current_debt = initial_debt_points
        for month in range(1, simulation_months + 1):
            # New debt added & reduced
            new_debt = current_debt * (tech_debt_growth/100)
            reduced_debt = current_debt * (action_effectiveness/100)
            current_debt = current_debt + new_debt - reduced_debt
            with_action_debt.append(max(0, current_debt))
        
        # Risk exposure projection
        no_action_risk = [weighted_risk_exposure * (1 + risk_growth/100)**month for month in months]
        with_action_risk = [weighted_risk_exposure]
        current_risk = weighted_risk_exposure
        for month in range(1, simulation_months + 1):
            # New risk added & mitigated
            new_risk = current_risk * (risk_growth/100)
            mitigated_risk = current_risk * (action_effectiveness/100)
            current_risk = current_risk + new_risk - mitigated_risk
            with_action_risk.append(max(0, current_risk))
        
        # Project productivity impact
        no_action_productivity = [100]
        with_action_productivity = [100]
        for month in range(1, simulation_months + 1):
            # No action: decline based on debt and risk
            debt_factor = no_action_debt[month] / initial_debt_points if initial_debt_points > 0 else 1
            risk_factor = no_action_risk[month] / weighted_risk_exposure if weighted_risk_exposure > 0 else 1
            productivity_decline = productivity_decline_rate * (debt_factor * 0.6 + risk_factor * 0.4)  # Weighted combination
            
            no_action_productivity.append(no_action_productivity[-1] * (1 - productivity_decline/100))
            
            # With action: improved trajectory
            debt_factor_improved = with_action_debt[month] / initial_debt_points if initial_debt_points > 0 else 1
            risk_factor_improved = with_action_risk[month] / weighted_risk_exposure if weighted_risk_exposure > 0 else 1
            productivity_decline_improved = productivity_decline_rate * (debt_factor_improved * 0.6 + risk_factor_improved * 0.4) * (1 - action_effectiveness/100)
            
            with_action_productivity.append(with_action_productivity[-1] * (1 - productivity_decline_improved/100))
        
        # Visualization: Combined Impact
        st.markdown("#### 📉 Team Productivity Impact")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=months,
            y=no_action_productivity,
            mode='lines+markers',
            name='No Action',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=months,
            y=with_action_productivity,
            mode='lines+markers',
            name=f'With {action_effectiveness}% Improvement Actions',
            line=dict(color='green', width=2)
        ))
        
        # Add 80% productivity threshold line
        fig.add_hline(
            y=80, 
            line_width=2, 
            line_dash="dash", 
            line_color="orange",
            annotation_text="Critical Productivity Threshold",
            annotation_position="bottom right"
        )
        
        fig.update_layout(
            title="Team Productivity Projection",
            xaxis_title="Months",
            yaxis_title="Productivity (%)",
            yaxis=dict(range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate the time when productivity falls below 80%
        threshold = 80
        critical_month_no_action = None
        for i, prod in enumerate(no_action_productivity):
            if prod < threshold:
                critical_month_no_action = i
                break
                
        critical_month_with_action = None
        for i, prod in enumerate(with_action_productivity):
            if prod < threshold:
                critical_month_with_action = i
                break
        
        # Calculate velocity impact
        final_velocity_no_action = project_baseline_velocity * (no_action_productivity[-1] / 100)
        final_velocity_with_action = project_baseline_velocity * (with_action_productivity[-1] / 100)
        velocity_difference = final_velocity_with_action - final_velocity_no_action
        
        # Project timeline impact
        project_baseline_duration = 12  # months
        final_duration_no_action = project_baseline_duration * (100 / no_action_productivity[-1])
        final_duration_with_action = project_baseline_duration * (100 / with_action_productivity[-1])
        duration_difference = final_duration_no_action - final_duration_with_action
        
        # Display impact metrics
        st.markdown("### 🎯 Project Impact Metrics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Final Team Productivity", 
                f"{with_action_productivity[-1]:.1f}%",
                delta=f"{with_action_productivity[-1] - no_action_productivity[-1]:.1f}% improvement with action",
                delta_color="inverse"
            )
        with col2:
            if critical_month_no_action:
                if critical_month_with_action:
                    st.metric(
                        "Months before Critical Productivity Drop", 
                        critical_month_with_action,
                        delta=f"+{critical_month_with_action - critical_month_no_action} months with action",
                        delta_color="inverse"
                    )
                else:
                    st.metric(
                        "Critical Productivity Drop", 
                        "Avoided with action",
                        delta=f"Critical drop at month {critical_month_no_action} avoided",
                        delta_color="inverse"
                    )
            else:
                st.metric(
                    "Critical Productivity Drop", 
                    "Not reached",
                    delta="Productivity stays above 80%",
                    delta_color="inverse"
                )
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Final Team Velocity", 
                f"{final_velocity_with_action:.1f} points/month",
                delta=f"+{velocity_difference:.1f} points/month with action",
                delta_color="inverse"
            )
        with col2:
            st.metric(
                "Project Timeline Impact", 
                f"{final_duration_with_action:.1f} months",
                delta=f"-{duration_difference:.1f} months with action",
                delta_color="inverse"
            )
        
        # Combined cost impact
        st.markdown("### 💰 Financial Impact Assessment")
        
        # Financial impact parameters
        avg_developer_cost = 15000  # dollars per month
        team_size = 5  # developers
        monthly_dev_cost = avg_developer_cost * team_size
        
        # Calculate cumulative project cost
        baseline_project_cost = monthly_dev_cost * project_baseline_duration
        project_cost_no_action = monthly_dev_cost * final_duration_no_action
        project_cost_with_action = monthly_dev_cost * final_duration_with_action
        cost_savings = project_cost_no_action - project_cost_with_action
        
        # Quality impact parameters
        defect_rate_baseline = 5  # defects per 100 points delivered
        defect_cost = 2000  # dollars per defect
        
        # Calculate quality costs
        defect_multiplier_no_action = max(1, 2 - no_action_productivity[-1]/100)  # increases as productivity drops
        defect_multiplier_with_action = max(1, 2 - with_action_productivity[-1]/100)
        
        defect_rate_no_action = defect_rate_baseline * defect_multiplier_no_action
        defect_rate_with_action = defect_rate_baseline * defect_multiplier_with_action
        
        quality_cost_no_action = defect_rate_no_action * project_baseline_velocity * 12 * defect_cost / 100
        quality_cost_with_action = defect_rate_with_action * project_baseline_velocity * 12 * defect_cost / 100
        quality_savings = quality_cost_no_action - quality_cost_with_action
        
        # Display financial metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Timeline Cost Impact", 
                f"${project_cost_with_action:,.0f}",
                delta=f"-${cost_savings:,.0f} with action",
                delta_color="inverse"
            )
        with col2:
            st.metric(
                "Quality Cost Impact", 
                f"${quality_cost_with_action:,.0f}",
                delta=f"-${quality_savings:,.0f} with action",
                delta_color="inverse"
            )
        with col3:
            total_savings = cost_savings + quality_savings
            total_cost_with_action = project_cost_with_action + quality_cost_with_action
            total_cost_no_action = project_cost_no_action + quality_cost_no_action
            
            st.metric(
                "Total Financial Impact", 
                f"${total_cost_with_action:,.0f}",
                delta=f"-${total_savings:,.0f} ({total_savings/total_cost_no_action*100:.1f}%)",
                delta_color="inverse"
            )
        
        # Investment needs
        improvement_investment = total_savings * 0.3  # Typically need to invest 30% of potential savings
        
        st.markdown(f"**Estimated Investment Required:** ${improvement_investment:,.0f} (30% of potential savings)")
        st.markdown(f"**Potential ROI:** {(total_savings/improvement_investment):.1f}x return on investment")
        
        # AI-powered combined analysis
        if client is not None:
            st.markdown("### 🤖 AI Strategic Analysis")
            
            with st.spinner("Generating AI strategic analysis..."):
                try:
                    # Prepare simulation data for the analysis
                    simulation_data = {
                        "simulation_months": simulation_months,
                        "action_effectiveness": action_effectiveness,
                        "productivity": {
                            "initial": 100,
                            "final_no_action": no_action_productivity[-1],
                            "final_with_action": with_action_productivity[-1],
                        },
                        "critical_thresholds": {
                            "critical_month_no_action": critical_month_no_action,
                            "critical_month_with_action": critical_month_with_action
                        },
                        "project_impact": {
                            "timeline_months_no_action": final_duration_no_action,
                            "timeline_months_with_action": final_duration_with_action,
                            "cost_no_action": float(project_cost_no_action + quality_cost_no_action),
                            "cost_with_action": float(project_cost_with_action + quality_cost_with_action),
                            "roi": float(total_savings/improvement_investment) if improvement_investment > 0 else 0
                        }
                    }
                    
                    # Build prompt
                    prompt = f"""
                    You are a software delivery consultant analyzing the combined impact of technical debt and risk. 
                    Review this projection data and provide strategic insights and an action plan:
                    
                    {json.dumps(simulation_data, indent=2)}
                    
                    Provide the following:
                    1. Executive summary of the combined technical debt and risk trajectory
                    2. Business case for taking action with ROI analysis
                    3. Strategic action plan with 3-5 specific recommendations
                    4. Implementation roadmap (30-60-90 day plan)
                    
                    Format your response as JSON with the following structure:
                    {{
                        "executive_summary": "Clear summary of the situation and impacts",
                        "business_case": "ROI-focused business case for action",
                        "recommendations": [
                            {{"action": "Recommendation 1", "benefit": "Expected benefit", "timeframe": "Implementation timeframe"}},
                            {{"action": "Recommendation 2", "benefit": "Expected benefit", "timeframe": "Implementation timeframe"}},
                            {{"action": "Recommendation 3", "benefit": "Expected benefit", "timeframe": "Implementation timeframe"}}
                        ],
                        "roadmap": {{
                            "30_days": ["Action 1", "Action 2"],
                            "60_days": ["Action 3", "Action 4"],
                            "90_days": ["Action 5", "Action 6"]
                        }}
                    }}
                    """
                    
                    # Send to OpenAI API
                    response = client.chat.completions.create(
                        model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. Do not change this unless explicitly requested by the user
                        messages=[{"role": "system", "content": "You are an AI assistant that helps with strategic technical planning."}, 
                                 {"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        temperature=0.2,
                    )
                    
                    # Parse the response
                    ai_analysis = json.loads(response.choices[0].message.content)
                    
                    # Display AI analysis
                    st.markdown("#### Executive Summary")
                    st.info(ai_analysis.get("executive_summary", "No summary provided"))
                    
                    st.markdown("#### Business Case for Action")
                    st.success(ai_analysis.get("business_case", "No business case provided"))
                    
                    st.markdown("#### Strategic Recommendations")
                    for i, rec in enumerate(ai_analysis.get("recommendations", [])):
                        with st.expander(f"{i+1}. {rec.get('action', 'Recommendation')}"):
                            st.markdown(f"**Expected Benefit:** {rec.get('benefit', '')}")
                            st.markdown(f"**Implementation Timeframe:** {rec.get('timeframe', '')}")
                    
                    st.markdown("#### Implementation Roadmap")
                    roadmap = ai_analysis.get("roadmap", {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**First 30 Days**")
                        for action in roadmap.get("30_days", []):
                            st.markdown(f"- {action}")
                    with col2:
                        st.markdown("**Days 31-60**")
                        for action in roadmap.get("60_days", []):
                            st.markdown(f"- {action}")
                    with col3:
                        st.markdown("**Days 61-90**")
                        for action in roadmap.get("90_days", []):
                            st.markdown(f"- {action}")
                    
                except Exception as e:
                    st.error(f"Could not generate AI analysis: {e}")
