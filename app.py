# AI PM Buddy - Advanced Project Management Assistant
# Integrated application that combines visualization dashboards with AI-powered project management assistant

import streamlit as st

# Page configuration and title must come before any other Streamlit commands
st.set_page_config(
    page_title="AI PM Buddy",
    page_icon="📊",
    layout="wide"
)

# Continue with other imports
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import xlsxwriter
from openai import OpenAI
from datetime import datetime, timedelta
import os
import re
from utils import load_data
from fpdf import FPDF

# Import our custom modules
from ai_task_redistribution import ai_based_task_redistribution
from leave_conflict_detection import detect_leave_conflicts
from token_management import TokenManager, optimize_prompt
from doc_dock import doc_dock_ui

# Create a token manager instance for tracking usage
token_manager = TokenManager()

# Initialize OpenAI client function
def get_openai_client():
    """Initialize and return the OpenAI client with proper error handling"""
    try:
        # Get OpenAI API key from environment or Streamlit secrets
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
            client = OpenAI(api_key=api_key)
            return client
        else:
            print("OpenAI API key not found in Streamlit secrets")
            return None
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return None

# Global OpenAI client - will be used by all AI functions
openai_client = get_openai_client()

# Import AI features
from project_health_summary import generate_project_health_summary
from task_prioritization import ai_driven_task_prioritization
from effort_estimation import effort_estimation_refinement
from technical_debt_risk_management import technical_debt_risk_management
from sprint_planning_assistant import sprint_planning_assistant

# Initialize session state variables
if 'feedback_history' not in st.session_state:
    st.session_state['feedback_history'] = []
    print("Initializing empty feedback_history list in session state")
else:
    print(f"Current feedback history has {len(st.session_state['feedback_history'])} items")
    
if 'chat_session' not in st.session_state:
    st.session_state['chat_session'] = []
    
# Session state for Daily Brief action buttons
if 'show_overdue' not in st.session_state:
    st.session_state['show_overdue'] = False
    
if 'show_due_soon' not in st.session_state:
    st.session_state['show_due_soon'] = False
    
# Navigation session state variables
if 'sidebar_selection' not in st.session_state:
    st.session_state['sidebar_selection'] = None
    
if 'resource_tab' not in st.session_state:
    st.session_state['resource_tab'] = None

# Function to append to feedback history
def append_to_feedback_history(entry):
    """Helper function to reliably append to feedback history and print debug info"""
    if 'feedback_history' not in st.session_state:
        st.session_state['feedback_history'] = []
    st.session_state['feedback_history'].append(entry)
    print(f"Added entry to feedback history. Now contains {len(st.session_state['feedback_history'])} items.")

# ---------- Sidebar ----------
with st.sidebar:
    st.title("AI PM Buddy")
    
    # Try to load the custom logo based on theme, with fallback to emoji if files not found
    try:
        # Choose logo based on current theme
        if 'theme' in st.session_state and st.session_state.theme == "dark":
            logo_path = "logo_AI_PM_Buddy_dark.png"
        else:
            logo_path = "logo_AI_PM_Buddy.png"
            
        # Check if logo file exists
        if os.path.exists(logo_path):
            st.image(logo_path, width=180)
        else:
            # If preferred theme logo not found, try the other logo
            alternate_logo = "logo_AI_PM_Buddy.png" if logo_path == "logo_AI_PM_Buddy_dark.png" else "logo_AI_PM_Buddy_dark.png"
            if os.path.exists(alternate_logo):
                st.image(alternate_logo, width=180)
                print(f"Using alternate logo: {alternate_logo}")
            else:
                # If no logo files found, use emoji fallback
                st.markdown("### 🤖📊")
                st.info("Logo files not found")
    except Exception as e:
        # In case of any other error, use emoji as fallback
        st.markdown("### 🤖📊")
        print(f"Error loading logo: {e}")

    
    # Theme toggle for dark/light mode
    if 'theme' not in st.session_state:
        st.session_state.theme = "light"
        
    # Add theme toggle in the sidebar
    theme_col1, theme_col2 = st.columns([1, 3])
    with theme_col1:
        dark_mode = st.checkbox("🌙", value=st.session_state.theme == "dark", key="dark_mode_toggle")
    with theme_col2:
        st.markdown("**Dark Mode**" if dark_mode else "**Light Mode**")
    
    if dark_mode and st.session_state.theme == "light":
        st.session_state.theme = "dark"
        # Save dark theme settings to config.toml
        with open(".streamlit/config.toml", "w") as f:
            f.write("""[server]
headless = true
address = "0.0.0.0"
port = 5000

[browser]
gatherUsageStats = false

[theme]
base = "dark"
primaryColor = "#1E88E5"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"
""")
        st.rerun()
    elif not dark_mode and st.session_state.theme == "dark":
        st.session_state.theme = "light"
        # Save light theme settings to config.toml
        with open(".streamlit/config.toml", "w") as f:
            f.write("""[server]
headless = true
address = "0.0.0.0"
port = 5000

[browser]
gatherUsageStats = false

[theme]
base = "light"
primaryColor = "#1E88E5"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
""")
        st.rerun()
    
    # File uploader
    st.subheader("Data Source")
    uploaded_file = st.file_uploader("Upload JIRA Excel File", type=["xlsx"])
    
    # If no file is uploaded, use the default file
    if uploaded_file is None:
        # Try to use the file with user-defined risk fields first (check both paths)
        fallback_paths = [
            "attached_assets/enriched_jira_data_furthercorrected1.xlsx",  # Original path
            "enriched_jira_data_furthercorrected1.xlsx"  # Deployment path
        ]
        
        fallback_file = None
        for path in fallback_paths:
            if os.path.exists(path):
                fallback_file = path
                break
                
        if fallback_file:
            uploaded_file = open(fallback_file, "rb")
            st.sidebar.success(f"Loaded default file: {os.path.basename(fallback_file)}")
        else:
            # Fall back to the simulated data if the corrected file is not available
            fallback_file = "enriched_jira_data_with_simulated.xlsx"
            if os.path.exists(fallback_file):
                uploaded_file = open(fallback_file, "rb")
                st.sidebar.success("Loaded default file: enriched_jira_data_with_simulated.xlsx")

    # Deployment package download options - at the bottom of the sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### Application Downloads")
    
    # Check if zip files exist before showing download buttons
    col1, col2 = st.sidebar.columns(2)
    
    cloud_zip_file_path = "ai_pm_buddy_cloud_deploy.zip"
    if os.path.exists(cloud_zip_file_path):
        with open(cloud_zip_file_path, "rb") as fp:
            cloud_zip_data = fp.read()
        col1.download_button(
            label="☁️ Cloud Deploy",
            data=cloud_zip_data,
            file_name="ai_pm_buddy_cloud_deploy.zip",
            mime="application/zip",
            help="Download files for Streamlit Cloud deployment"
        )

    # Full deployment package
    zip_file_path = "ai_pm_buddy_app.zip"
    if os.path.exists(zip_file_path):
        with open(zip_file_path, "rb") as fp:
            zip_data = fp.read()
        col2.download_button(
            label="📦 Full Deploy",
            data=zip_data,
            file_name="ai_pm_buddy_app.zip",
            mime="application/zip",
            help="Download all project files for deployment"
        )

# ---------- Load Data ----------
# Initialize global variables
issues_df, skills_df, worklogs_df, leaves_df, tech_debt_df = None, None, None, None, None

# Define color palettes for dark and light mode
def get_color_palette():
    """Return appropriate color palette based on current theme"""
    if 'theme' in st.session_state and st.session_state.theme == "dark":
        # Dark mode palette - more vibrant colors that work well on dark backgrounds
        return {
            'primary': '#1E88E5',  # Blue
            'secondary': '#5E35B1',  # Purple
            'success': '#43A047',  # Green
            'warning': '#FDD835',  # Yellow
            'danger': '#E53935',  # Red
            'info': '#00ACC1',  # Cyan
            'light_accent': '#78909C',  # Blue grey
            'dark_accent': '#455A64',  # Dark blue grey
            'background': '#0E1117',
            'text': '#FAFAFA',
            'grid': '#555555',
            # Color sequences for multi-series charts
            'categorical': ['#42A5F5', '#9575CD', '#4CAF50', '#FFC107', '#FF5722', '#26A69A', '#EC407A', '#AB47BC'],
            'sequential': ['#bbdefb', '#90caf9', '#64b5f6', '#42a5f5', '#2196f3', '#1e88e5', '#1976d2', '#1565c0', '#0d47a1'],
            'diverging': ['#ef5350', '#f44336', '#e53935', '#d32f2f', '#c62828', '#ffee58', '#ffeb3b', '#fdd835', '#66bb6a', '#4caf50', '#43a047']
        }
    else:
        # Light mode palette - slightly muted colors that work well on light backgrounds
        return {
            'primary': '#1976D2',  # Blue
            'secondary': '#512DA8',  # Purple
            'success': '#388E3C',  # Green
            'warning': '#F9A825',  # Yellow
            'danger': '#D32F2F',  # Red
            'info': '#0097A7',  # Cyan
            'light_accent': '#607D8B',  # Blue grey
            'dark_accent': '#37474F',  # Dark blue grey
            'background': '#FFFFFF',
            'text': '#212121',
            'grid': '#E0E0E0',
            # Color sequences for multi-series charts
            'categorical': ['#1976D2', '#673AB7', '#388E3C', '#FBC02D', '#E64A19', '#00897B', '#D81B60', '#8E24AA'],
            'sequential': ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#2196F3', '#1E88E5', '#1976D2', '#1565C0'],
            'diverging': ['#EF5350', '#E53935', '#D32F2F', '#C62828', '#B71C1C', '#FFEB3B', '#FDD835', '#FBC02D', '#4CAF50', '#388E3C', '#2E7D32']
        }

# Function to create consistent Plotly chart styling
def safe_add_vline(fig, x, **kwargs):
    """Safe method to add vertical lines to Plotly figures that works around add_vline issues"""
    try:
        # First attempt: Try the standard add_vline method
        fig.add_vline(x=x, **kwargs)
    except Exception as e:
        # Fallback: Add a scatter trace instead
        for i in range(len(fig.data)):
            try:
                y_range = fig.layout.yaxis.range
                if not y_range:
                    y_range = [0, 1]  # Default range if none is set
            except:
                y_range = [0, 1]  # Default fallback
                
            # Add a line as a scatter trace
            fig.add_trace(
                go.Scatter(
                    x=[x, x],
                    y=[y_range[0], y_range[1]],
                    mode='lines',
                    line=dict(
                        color=kwargs.get('line_color', 'red'),
                        width=kwargs.get('line_width', 1),
                        dash=kwargs.get('line_dash', 'solid')
                    ),
                    showlegend=False,
                    hoverinfo='none'
                )
            )
            break  # Just add one trace
    return fig

def safe_line_chart(data_frame, x, y, **kwargs):
    """Safe method to create line charts that works around px.line issues"""
    try:
        # First attempt: Try the standard px.line method
        fig = px.line(data_frame, x=x, y=y, **kwargs)
        return fig
    except ValueError as e:
        # Fallback: Create a manual line chart using go.Figure and go.Scatter
        fig = go.Figure()
        
        # Get the color sequence
        colors = get_color_palette()
        color_sequence = kwargs.get('color_discrete_sequence', colors['categorical'])
        
        # Check if we're using a color column
        if 'color' in kwargs:
            color_col = kwargs['color']
            # Make sure the color column exists before trying to access it
            if color_col in data_frame.columns:
                # Get unique values in the color column
                color_groups = data_frame[color_col].unique()
                
                # Add a trace for each color group
                for i, group in enumerate(color_groups):
                    group_data = data_frame[data_frame[color_col] == group]
                    color_idx = i % len(color_sequence)
                    
                    # Add scatter trace with lines and markers
                    fig.add_trace(go.Scatter(
                        x=group_data[x],
                        y=group_data[y],
                        mode='lines+markers' if kwargs.get('markers', False) else 'lines',
                        name=str(group),
                        line=dict(color=color_sequence[color_idx]),
                        marker=dict(color=color_sequence[color_idx])
                    ))
            else:
                # If the color column doesn't exist, create a simple line chart
                fig.add_trace(go.Scatter(
                    x=data_frame[x],
                    y=data_frame[y],
                    mode='lines+markers' if kwargs.get('markers', False) else 'lines',
                    line=dict(color=color_sequence[0]),
                    marker=dict(color=color_sequence[0])
                ))
        else:
            # Simple line chart without color groups
            fig.add_trace(go.Scatter(
                x=data_frame[x],
                y=data_frame[y],
                mode='lines+markers' if kwargs.get('markers', False) else 'lines',
                line=dict(color=color_sequence[0]),
                marker=dict(color=color_sequence[0])
            ))
        
        # Apply title if provided
        if 'title' in kwargs:
            fig.update_layout(title=kwargs['title'])
            
        # Apply proper labels
        x_label = kwargs.get('labels', {}).get(x, x) if 'labels' in kwargs else x
        y_label = kwargs.get('labels', {}).get(y, y) if 'labels' in kwargs else y
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title=y_label
        )
        
        return fig

def style_plotly_chart(fig, title=None, height=None, chart_type="default"):
    """Apply consistent styling to Plotly charts based on current theme"""
    # Import visualization enhancements 
    from visualization_enhancements import optimize_chart_layout, enhance_chart_for_dark_mode
    from visualization_enhancements import improve_sprint_burnup_chart, improve_task_distribution_chart
    from visualization_enhancements import improve_gantt_chart, improve_bubble_chart

    colors = get_color_palette()
    
    # Apply title if provided with improved positioning
    if title:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, color=colors['text']),
                y=0.98,  # Slightly lower from top
                x=0.5,   # Centered
                xanchor='center',
                yanchor='top'
            )
        )
        
    # Set height if provided, with minimum to prevent squashing
    if height:
        fig.update_layout(height=max(height, 350))
    else:
        fig.update_layout(height=400)  # Default height if none provided
    
    # Explicitly set the template to None first to avoid conflicts
    fig.update_layout(template=None)
    
    # Apply theme-based styling with improved margins
    fig.update_layout(
        font=dict(
            family="Helvetica, Arial, sans-serif",  # More readable font
            size=13,  # Slightly larger font
            color=colors['text']
        ),
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        title_font=dict(size=18, color=colors['text']),
        legend=dict(
            font=dict(color=colors['text'], size=12),
            bgcolor=colors['background'],
            bordercolor=colors['grid'],
            borderwidth=1
        ),
        # Increased margins to prevent label cropping
        margin=dict(l=70, r=50, t=80, b=80),
        hovermode="closest"
    )
    
    # Style axes with improved label positioning
    fig.update_xaxes(
        gridcolor=colors['grid'],
        linecolor=colors['grid'],
        zeroline=True,
        zerolinecolor=colors['grid'],
        tickfont=dict(color=colors['text'], size=11),
        title_font=dict(color=colors['text'], size=13),
        title_standoff=30,  # More space for axis title
        tickangle=0,  # Default horizontal ticks
        color=colors['text'],
        showline=True,
        showticklabels=True,
        mirror=True
    )
    
    fig.update_yaxes(
        gridcolor=colors['grid'],
        linecolor=colors['grid'],
        zeroline=True,
        zerolinecolor=colors['grid'],
        tickfont=dict(color=colors['text'], size=11),
        title_font=dict(color=colors['text'], size=13),
        title_standoff=30,  # More space for axis title
        color=colors['text'],
        showline=True,
        showticklabels=True,
        mirror=True
    )
    
    # Apply chart-specific optimizations
    fig = optimize_chart_layout(fig, chart_type=chart_type, colors=colors)
    
    # Apply specific improvements based on chart type
    if chart_type == "burnup":
        fig = improve_sprint_burnup_chart(fig, colors)
    elif chart_type == "task_distribution":
        fig = improve_task_distribution_chart(fig, colors)
    elif chart_type == "gantt":
        fig = improve_gantt_chart(fig, colors)
    elif chart_type == "bubble":
        fig = improve_bubble_chart(fig, colors)
    
    # Apply dark mode specific enhancements
    if 'theme' in st.session_state and st.session_state.theme == "dark":
        fig = enhance_chart_for_dark_mode(fig, colors)
    
    return fig

# Load data from file
if uploaded_file is not None:
    issues_df, skills_df, worklogs_df, leaves_df, tech_debt_df = load_data(uploaded_file)

# ---------- New Navigation Structure ----------
with st.sidebar:
    st.markdown("---")
    nav_options = [
        "📊 Dashboard",
        "🎯 Resource Management",
        "📆 Planning & Scheduling",
        "🚨 Risk Management",
        "🤖 Strateg-AIz"
    ]
    
    nav_selection = st.radio("Navigation", nav_options)
    
    # Check if navigation is set via recommendation links
    if st.session_state.get('sidebar_selection') is not None:
        nav_selection = st.session_state['sidebar_selection']
        # Reset after use
        st.session_state['sidebar_selection'] = None
        
    # Sub-navigation options based on main selection
    if nav_selection == "📊 Dashboard":
        st.markdown("### Dashboard Views")
        dashboard_view = st.radio(
            "",
            ["Project Overview", "Sprint Status", "Resource Allocation"],
            label_visibility="collapsed"
        )
    
    elif nav_selection == "🎯 Resource Management":
        st.markdown("### Resource Options")
        
        # Check if a specific resource tab was requested
        if st.session_state.get('resource_tab') is not None:
            if st.session_state['resource_tab'] == "AI Task Redistribution":
                # Map user-friendly name to radio button option
                default_index = 2  # "Task Redistribution (AI)" is at index 2
            else:
                default_index = 0
            # Reset after use
            st.session_state['resource_tab'] = None
        else:
            default_index = 0
            
        resource_view = st.radio(
            "",
            ["Team Workload", "Skill Distribution", "Task Redistribution (AI)"],
            index=default_index,
            label_visibility="collapsed"
        )
    
    elif nav_selection == "📆 Planning & Scheduling":
        st.markdown("### Planning Options")
        planning_view = st.radio(
            "",
            ["Sprint Planning", "Leave Impact Analysis", "Timeline Forecasting"],
            label_visibility="collapsed"
        )
    
    elif nav_selection == "🚨 Risk Management":
        st.markdown("### Risk Options")
        
        # Check for navigation to Daily Brief from recommendation links
        if st.session_state.get('risk_view') is not None:
            if st.session_state['risk_view'] == "Daily Brief":
                default_index = 0  # "Daily Brief" is at index 0
            elif st.session_state['risk_view'] == "Technical Debt":
                default_index = 1
            elif st.session_state['risk_view'] == "Risk Assessment":
                default_index = 2
            else:
                default_index = 0
            # Reset after use
            st.session_state['risk_view'] = None
        else:
            default_index = 0
            
        risk_view = st.radio(
            "",
            ["Daily Brief", "Technical Debt", "Risk Assessment"],
            index=default_index,
            label_visibility="collapsed"
        )
    
    # Quick actions panel
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Actions")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        qa_brief = st.button("📋 PM Brief")
        qa_balance = st.button("⚖️ Balance")
    with col2:
        qa_optimize = st.button("🔄 Optimize")
        qa_plan = st.button("🗓️ Plan")

# ---------- Reusable Standard Filter Component ----------
def standard_filter_section(expanded=True, section_id="default"):
    # Initialize filter state variables if not present
    if f"filter_project_{section_id}" not in st.session_state:
        st.session_state[f"filter_project_{section_id}"] = "All Projects"
    if f"filter_sprint_{section_id}" not in st.session_state:
        st.session_state[f"filter_sprint_{section_id}"] = "All Sprints"
    if f"filter_resource_{section_id}" not in st.session_state:
        st.session_state[f"filter_resource_{section_id}"] = "All Resources"
    
    with st.expander("Filters", expanded=expanded):
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            # Project/POD filter
            if issues_df is not None and 'Project' in issues_df.columns:
                projects = ["All Projects"] + sorted(issues_df['Project'].unique().tolist())
                selected_project = st.selectbox(
                    "Project/POD", 
                    projects, 
                    key=f"filter_project_{section_id}"
                )
            else:
                selected_project = "All Projects"
                st.selectbox("Project/POD", ["All Projects"], key=f"filter_project_empty_{section_id}")
        
        with filter_col2:
            # Sprint filter
            if issues_df is not None and 'Sprint' in issues_df.columns:
                sprints = ["All Sprints"] + sorted(issues_df['Sprint'].dropna().unique().tolist())
                selected_sprint = st.selectbox(
                    "Sprint", 
                    sprints, 
                    key=f"filter_sprint_{section_id}"
                )
            else:
                selected_sprint = "All Sprints"
                st.selectbox("Sprint", ["All Sprints"], key=f"filter_sprint_empty_{section_id}")
        
        with filter_col3:
            # Resource filter
            resources = ["All Resources"]
            if worklogs_df is not None and 'Resource' in worklogs_df.columns:
                resources += sorted(worklogs_df['Resource'].unique().tolist())
            elif skills_df is not None and 'Resource' in skills_df.columns:
                resources += sorted(skills_df['Resource'].unique().tolist())
            selected_resource = st.selectbox(
                "Resource", 
                resources, 
                key=f"filter_resource_{section_id}"
            )
        
        # Print debug info about selections
        print(f"Current filter selections for {section_id}:")
        print(f"  Project: {selected_project}")
        print(f"  Sprint: {selected_sprint}")
        print(f"  Resource: {selected_resource}")
        
        # Return selected filters
        return {
            "project": selected_project,
            "sprint": selected_sprint,
            "resource": selected_resource
        }

# Apply selected filters to dataframes
def apply_filters(filters):
    # Start with copies of original dataframes
    filtered_issues_df = issues_df.copy() if issues_df is not None else None
    filtered_worklogs_df = worklogs_df.copy() if worklogs_df is not None else None
    filtered_skills_df = skills_df.copy() if skills_df is not None else None
    filtered_leaves_df = leaves_df.copy() if leaves_df is not None else None
    
    # Debug information about filters and dataframes
    print(f"Applying filters: {filters}")
    print(f"Original issues_df shape: {filtered_issues_df.shape if filtered_issues_df is not None else None}")
    
    # Apply Project filter
    if filtered_issues_df is not None and filters["project"] != "All Projects":
        print(f"Filtering by project: {filters['project']}")
        print(f"Project values in dataframe: {filtered_issues_df['Project'].unique()}")
        
        # Check column names for debugging
        print(f"Columns in issues_df: {filtered_issues_df.columns.tolist()}")
        print(f"Columns in worklogs_df: {filtered_worklogs_df.columns.tolist() if filtered_worklogs_df is not None else 'None'}")
        
        filtered_issues_df = filtered_issues_df[filtered_issues_df['Project'] == filters["project"]]
        print(f"After project filter - issues_df shape: {filtered_issues_df.shape}")
        
        # Filter related worklogs - check whether to use 'Issue Key' or 'Issue key'
        if filtered_worklogs_df is not None:
            # Determine the correct column name
            issue_key_col = None
            if 'Issue Key' in filtered_worklogs_df.columns:
                issue_key_col = 'Issue Key'
            elif 'Issue key' in filtered_worklogs_df.columns:
                issue_key_col = 'Issue key'
                
            if issue_key_col:
                # Determine the correct column name in issues_df
                issues_key_col = None
                if 'Issue Key' in filtered_issues_df.columns:
                    issues_key_col = 'Issue Key'
                elif 'Issue key' in filtered_issues_df.columns:
                    issues_key_col = 'Issue key'
                
                if issues_key_col:
                    filtered_issue_keys = filtered_issues_df[issues_key_col].unique()
                    filtered_worklogs_df = filtered_worklogs_df[filtered_worklogs_df[issue_key_col].isin(filtered_issue_keys)]
                    print(f"After filtering worklogs by {issues_key_col} - worklogs_df shape: {filtered_worklogs_df.shape}")
                else:
                    print("Could not find Issue Key column in issues_df")
    
    # Apply Sprint filter
    if filtered_issues_df is not None and 'Sprint' in filtered_issues_df.columns and filters["sprint"] != "All Sprints":
        print(f"Filtering by sprint: {filters['sprint']}")
        print(f"Sprint values in dataframe: {filtered_issues_df['Sprint'].unique()}")
        filtered_issues_df = filtered_issues_df[filtered_issues_df['Sprint'] == filters["sprint"]]
        print(f"After sprint filter - issues_df shape: {filtered_issues_df.shape}")
        
        # Filter related worklogs - check whether to use 'Issue Key' or 'Issue key'
        if filtered_worklogs_df is not None:
            # Determine the correct column name
            issue_key_col = None
            if 'Issue Key' in filtered_worklogs_df.columns:
                issue_key_col = 'Issue Key'
            elif 'Issue key' in filtered_worklogs_df.columns:
                issue_key_col = 'Issue key'
                
            if issue_key_col:
                # Determine the correct column name in issues_df
                issues_key_col = None
                if 'Issue Key' in filtered_issues_df.columns:
                    issues_key_col = 'Issue Key'
                elif 'Issue key' in filtered_issues_df.columns:
                    issues_key_col = 'Issue key'
                
                if issues_key_col:
                    filtered_issue_keys = filtered_issues_df[issues_key_col].unique()
                    filtered_worklogs_df = filtered_worklogs_df[filtered_worklogs_df[issue_key_col].isin(filtered_issue_keys)]
                    print(f"After filtering worklogs by sprint - worklogs_df shape: {filtered_worklogs_df.shape}")
                else:
                    print("Could not find Issue Key column in issues_df")
    
    # Apply Resource filter
    if filters["resource"] != "All Resources":
        print(f"Filtering by resource: {filters['resource']}")
        
        # Resources might be called "Assignee" in the issues dataframe
        if filtered_issues_df is not None and 'Assignee' in filtered_issues_df.columns:
            print(f"Assignee values in dataframe: {filtered_issues_df['Assignee'].unique()}")
            filtered_issues_df = filtered_issues_df[filtered_issues_df['Assignee'] == filters["resource"]]
            print(f"After resource filter (assignee) - issues_df shape: {filtered_issues_df.shape}")
        
        # Filter worklogs by resource
        if filtered_worklogs_df is not None:
            resource_col = None
            if 'Resource' in filtered_worklogs_df.columns:
                resource_col = 'Resource'
            elif 'User' in filtered_worklogs_df.columns:  # Alternative column name
                resource_col = 'User'
                
            if resource_col:
                filtered_worklogs_df = filtered_worklogs_df[filtered_worklogs_df[resource_col] == filters["resource"]]
                print(f"After filtering worklogs by resource - worklogs_df shape: {filtered_worklogs_df.shape}")
        
        # Filter skills by resource
        if filtered_skills_df is not None:
            resource_col = None
            if 'Resource' in filtered_skills_df.columns:
                resource_col = 'Resource'
            elif 'Name' in filtered_skills_df.columns:  # Alternative column name
                resource_col = 'Name'
                
            if resource_col:
                filtered_skills_df = filtered_skills_df[filtered_skills_df[resource_col] == filters["resource"]]
                print(f"After filtering skills by resource - skills_df shape: {filtered_skills_df.shape}")
        
        # Filter leaves by resource
        if filtered_leaves_df is not None:
            resource_col = None
            if 'Resource' in filtered_leaves_df.columns:
                resource_col = 'Resource'
            elif 'User' in filtered_leaves_df.columns:  # Alternative column name
                resource_col = 'User'
                
            if resource_col:
                filtered_leaves_df = filtered_leaves_df[filtered_leaves_df[resource_col] == filters["resource"]]
                print(f"After filtering leaves by resource - leaves_df shape: {filtered_leaves_df.shape}")
    
    # Final debug information
    print(f"Final filtered issues_df shape: {filtered_issues_df.shape if filtered_issues_df is not None else None}")
    
    return filtered_issues_df, filtered_worklogs_df, filtered_skills_df, filtered_leaves_df

# ---------- Implement Visual Components ----------

# Now let's implement the different sections of the application
# I'll follow the mockup structure

# ---------- 1. DASHBOARD SECTION ----------
if nav_selection == "📊 Dashboard":
    # Header section
    st.title("📊 Dashboard")
    st.markdown(f"## {dashboard_view}")
    st.markdown("Comprehensive view of project status, resources, and milestones.")
    
    # Standard filter section
    filters = standard_filter_section(section_id="dashboard")
    
    # Apply filters
    filtered_issues_df, filtered_worklogs_df, filtered_skills_df, filtered_leaves_df = apply_filters(filters)
    
    # Dashboard content based on selected view
    if dashboard_view == "Project Overview":
        # Create tabs for different project views
        tab1, tab2, tab3 = st.tabs(["Key Metrics", "Status & Progress", "Team Performance"])
        
        with tab1:
            if filtered_issues_df is not None and not filtered_issues_df.empty:
                # Create metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                # Calculate metrics
                total_tasks = len(filtered_issues_df)
                completed_tasks = len(filtered_issues_df[filtered_issues_df['Status'] == 'Done'])
                at_risk_tasks = len(filtered_issues_df[filtered_issues_df['Priority'].isin(['High', 'Highest'])])
                
                # Calculate overdue tasks
                filtered_issues_df['Due Date'] = pd.to_datetime(filtered_issues_df['Due Date'], errors='coerce')
                today = pd.to_datetime("today")
                overdue_tasks = len(filtered_issues_df[
                    (filtered_issues_df['Due Date'] < today) & 
                    (filtered_issues_df['Status'] != 'Done')
                ])
                
                # Display metrics
                with col1:
                    st.metric("Total Tasks", total_tasks)
                with col2:
                    st.metric("Completed", completed_tasks, f"{int(completed_tasks/total_tasks*100)}%" if total_tasks > 0 else "0%")
                with col3:
                    st.metric("At Risk", at_risk_tasks)
                with col4:
                    st.metric("Overdue", overdue_tasks, delta=None if overdue_tasks == 0 else f"{overdue_tasks} task{'s' if overdue_tasks > 1 else ''} past due")
                
                # Status distribution chart
                st.subheader("Task Status Distribution")
                status_counts = filtered_issues_df['Status'].value_counts().reset_index()
                status_counts.columns = ['Status', 'Count']
                
                # Get current color palette
                colors = get_color_palette()
                
                # Create bar chart with enhanced styling for dark mode compatibility
                fig = go.Figure()
                
                # Manually add bars for better control
                for i, row in status_counts.iterrows():
                    fig.add_trace(go.Bar(
                        x=[row['Status']],
                        y=[row['Count']],
                        name=row['Status'],
                        text=[row['Count']],
                        textposition='outside',
                        marker_color=colors['categorical'][i % len(colors['categorical'])],
                        hovertemplate='<b>%{x}</b><br>Tasks: %{y}<extra></extra>',
                        marker_line=dict(width=0.5, color=colors['background'])
                    ))
                
                # Update layout with dark mode settings
                fig.update_layout(
                    title="Task Distribution by Status",
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    font=dict(color=colors['text']),
                    height=350,
                    margin=dict(l=10, r=10, t=40, b=0),
                    legend=dict(font=dict(color=colors['text']))
                )
                
                # Style axes explicitly
                fig.update_xaxes(
                    title="Task Status",
                    gridcolor=colors['grid'],
                    linecolor=colors['grid'],
                    tickfont=dict(color=colors['text']),
                    title_font=dict(color=colors['text'])
                )
                
                fig.update_yaxes(
                    title="Number of Tasks",
                    gridcolor=colors['grid'],
                    linecolor=colors['grid'],
                    tickfont=dict(color=colors['text']),
                    title_font=dict(color=colors['text'])
                )
                
                # Add subtle grid lines only on the y-axis
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=colors['grid'])
                fig.update_xaxes(showgrid=False)
                
                # Create a unique key based on the filter selections to prevent duplicate keys
                chart_key = f"plotly_status_{filters['project']}_{filters['sprint']}_{filters['resource']}".replace(" ", "_")
                st.plotly_chart(fig, use_container_width=True, key=chart_key)
                
                # Priority distribution
                st.subheader("Task Priority Distribution")
                if 'Priority' in filtered_issues_df.columns:
                    priority_counts = filtered_issues_df['Priority'].value_counts().reset_index()
                    priority_counts.columns = ['Priority', 'Count']
                    
                    # Define priority order for consistent display
                    priority_order = {"Highest": 1, "High": 2, "Medium": 3, "Low": 4, "Lowest": 5}
                    priority_counts['Order'] = priority_counts['Priority'].map(priority_order)
                    priority_counts = priority_counts.sort_values('Order')
                    
                    # Create pie chart with enhanced styling
                    colors = get_color_palette()
                    
                    # Create a custom color map that works well in both light and dark modes
                    # and follows a logical gradient from red (highest) to green (lowest)
                    if st.session_state.theme == 'dark':
                        color_map = {
                            'Highest': '#F44336',  # Bright red
                            'High': '#FF9800',    # Orange
                            'Medium': '#FFC107',  # Amber
                            'Low': '#8BC34A',     # Light green
                            'Lowest': '#4CAF50',  # Green
                            'Critical': '#D32F2F', # Darker red for critical
                            'Blocker': '#C62828',  # Even darker red for blockers
                            'Major': '#F57C00',    # Dark orange for major
                            'Minor': '#7CB342',    # Lighter green for minor
                            'Trivial': '#4CAF50'   # Green for trivial
                        }
                    else:
                        color_map = {
                            'Highest': '#E53935',  # Red 
                            'High': '#FB8C00',    # Orange
                            'Medium': '#FDD835',  # Yellow
                            'Low': '#7CB342',     # Light green
                            'Lowest': '#43A047',  # Green
                            'Critical': '#C62828', # Darker red for critical
                            'Blocker': '#B71C1C',  # Even darker red for blockers
                            'Major': '#EF6C00',    # Dark orange for major
                            'Minor': '#689F38',    # Lighter green for minor
                            'Trivial': '#388E3C'   # Green for trivial
                        }
                    
                    # Create pie chart with direct styling for dark mode
                    fig = go.Figure()
                    
                    # Add pie trace
                    fig.add_trace(go.Pie(
                        labels=priority_counts['Priority'],
                        values=priority_counts['Count'],
                        hole=0.4,  # Create a donut chart for more modern look
                        marker=dict(
                            colors=[color_map.get(p, colors['primary']) for p in priority_counts['Priority']],
                            line=dict(color=colors['background'], width=2)
                        ),
                        textposition='inside',
                        textinfo='percent+label',
                        hovertemplate='<b>%{label}</b><br>Tasks: %{value} (%{percent})<extra></extra>'
                    ))
                    
                    # Update layout with dark mode settings
                    fig.update_layout(
                        title="Task Distribution by Priority",
                        height=350,
                        plot_bgcolor=colors['background'],
                        paper_bgcolor=colors['background'],
                        font=dict(color=colors['text']),
                        margin=dict(l=10, r=10, t=40, b=0)
                    )
                    
                    # Additional customizations
                    fig.update_layout(
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.2,
                            xanchor="center",
                            x=0.5
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key="plotly_1062c0dbdc9c")
                
            else:
                st.warning("No data available with current filters.")
        
        with tab2:
            if filtered_issues_df is not None and not filtered_issues_df.empty:
                # Sprint Burnup Chart
                st.subheader("Sprint Progress")
                
                # Prepare burnup chart data
                burnup_df = filtered_issues_df.copy()
                burnup_df['Start Date'] = pd.to_datetime(burnup_df['Start Date'], errors='coerce')
                burnup_df['Due Date'] = pd.to_datetime(burnup_df['Due Date'], errors='coerce')
                
                # Skip if no date data
                if burnup_df['Start Date'].isna().all() or burnup_df['Due Date'].isna().all():
                    st.warning("Start Date or Due Date missing. Cannot build burnup chart.")
                else:
                    # Get current date for comparison
                    current_date = pd.Timestamp(datetime.now())
                    
                    # Create inferred resolution date column
                    burnup_df['Inferred Resolution Date'] = None
                    
                    # For done tasks, use due date as resolution date
                    mask_done = (burnup_df['Status'] == 'Done')
                    burnup_df.loc[mask_done, 'Inferred Resolution Date'] = burnup_df.loc[mask_done, 'Due Date']
                    
                    # If inferred date is in future, use today instead
                    mask_future = (burnup_df['Inferred Resolution Date'] > current_date) | (burnup_df['Inferred Resolution Date'].isna())
                    burnup_df.loc[mask_done & mask_future, 'Inferred Resolution Date'] = current_date
                    
                    # Convert to datetime format
                    burnup_df['Inferred Resolution Date'] = pd.to_datetime(burnup_df['Inferred Resolution Date'], errors='coerce')
                    
                    if not burnup_df['Start Date'].isna().all():
                        # Create date range from earliest start date to latest due date (or today if all past)
                        latest_date = max(burnup_df['Due Date'].max(), current_date)
                        date_range = pd.date_range(start=burnup_df['Start Date'].min(), end=latest_date)
                        
                        # Create dataframe for burnup chart
                        burnup_data = pd.DataFrame({'Date': date_range})
                        
                        # Calculate completed story points by date
                        burnup_data['Completed'] = burnup_data['Date'].apply(
                            lambda d: burnup_df[(burnup_df['Status'] == 'Done') & 
                                                (burnup_df['Inferred Resolution Date'] <= d)]['Story Points'].sum()
                        )
                        
                        # Total scope line
                        burnup_data['Total Scope'] = burnup_df['Story Points'].sum()
                        
                        # Add ideal burnup line
                        start_date = burnup_data['Date'].min()
                        end_date = burnup_data['Date'].max()
                        total_days = (end_date - start_date).days
                        
                        if total_days > 0:
                            total_scope = burnup_data['Total Scope'].iloc[0]
                            
                            # Calculate ideal burnup for each day
                            burnup_data['Ideal'] = burnup_data['Date'].apply(
                                lambda d: min(total_scope, total_scope * ((d - start_date).days / total_days))
                            )
                        else:
                            burnup_data['Ideal'] = burnup_data['Total Scope']
                        
                        # Get color palette for consistent styling
                        colors = get_color_palette()
                        
                        # Create plotly figure with improved styling
                        fig = go.Figure()
                        
                        # Actual completed line - using primary color
                        fig.add_trace(go.Scatter(
                            x=burnup_data['Date'], 
                            y=burnup_data['Completed'], 
                            mode='lines+markers', 
                            name='Completed',
                            line=dict(color=colors['success'], width=3),
                            marker=dict(size=8, symbol='circle', line=dict(width=1, color=colors['background'])),
                            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Completed: <b>%{y:.1f}</b> points<extra></extra>'
                        ))
                        
                        # Ideal burnup line - using warning color with transparent fill
                        fig.add_trace(go.Scatter(
                            x=burnup_data['Date'], 
                            y=burnup_data['Ideal'], 
                            mode='lines', 
                            name='Ideal Progress',
                            line=dict(dash='dot', color=colors['warning'], width=2),
                            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Ideal: <b>%{y:.1f}</b> points<extra></extra>'
                        ))
                        
                        # Total scope line - using danger color
                        fig.add_trace(go.Scatter(
                            x=burnup_data['Date'], 
                            y=[burnup_data['Total Scope'].iloc[0]]*len(burnup_data),
                            mode='lines', 
                            name='Total Scope', 
                            line=dict(dash='dash', color=colors['danger'], width=2),
                            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Total: <b>%{y:.1f}</b> points<extra></extra>'
                        ))
                        
                        # Apply consistent styling with burnup-specific enhancements
                        sprint_title = f'Sprint Burnup Chart{" for " + filters["sprint"] if filters["sprint"] != "All Sprints" else ""}'
                        fig = style_plotly_chart(fig, title=sprint_title, height=400, chart_type="burnup")
                        
                        # Use fix_axis_labels from visualization_enhancements to improve label positioning
                        from visualization_enhancements import fix_axis_labels
                        fig = fix_axis_labels(fig, x_title='Date', y_title='Story Points', colors=colors)
                        
                        # Add today marker with improved visibility
                        current_completed = burnup_data[burnup_data['Date'] <= current_date]['Completed'].iloc[-1] if not burnup_data[burnup_data['Date'] <= current_date].empty else 0
                        today_str = current_date.strftime('%Y-%m-%d')
                        
                        fig.add_annotation(
                            x=today_str,
                            y=current_completed,
                            text="Today",
                            font=dict(color=colors['text'], size=12, family="sans-serif"),
                            bgcolor=colors['primary'],
                            bordercolor=colors['primary'],
                            borderwidth=2,
                            borderpad=4,
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor=colors['primary'],
                            ax=0,
                            ay=-40
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key="plotly_7eaef315bd5c")
                
                # Traffic Light Matrix
                st.subheader("🚦 Traffic Light Matrix - Task Status")
                today = pd.to_datetime("today").normalize()
                filtered_issues_df['Due Date'] = pd.to_datetime(filtered_issues_df['Due Date'], errors='coerce')
                
                summary = filtered_issues_df.groupby('Assignee').agg(
                    total_tasks=('Issue Key', 'count'),
                    overdue_tasks=('Due Date', lambda d: (d < today).sum())
                ).reset_index()
                
                summary['Status'] = summary.apply(
                    lambda row: '🟢' if row['overdue_tasks'] == 0 else (
                        '🟠' if row['overdue_tasks'] < row['total_tasks'] * 0.5 else '🔴'
                    ), axis=1
                )
                
                # Add color-coding to the dataframe
                summary['Status Color'] = summary['Status'].map({'🟢': 'green', '🟠': 'orange', '🔴': 'red'})
                
                # Display as a styled DataFrame
                st.dataframe(summary[['Assignee', 'total_tasks', 'overdue_tasks', 'Status']], use_container_width=True)
            else:
                st.warning("No data available with current filters.")
        
        with tab3:
            if filtered_issues_df is not None and filtered_worklogs_df is not None:
                # Team Performance Metrics
                st.subheader("Team Performance")
                
                # Create two columns
                perf_col1, perf_col2 = st.columns(2)
                
                with perf_col1:
                    # Team Workload bar chart
                    st.subheader("Team Workload")
                    if 'Resource' in filtered_worklogs_df.columns and 'Time Spent (hrs)' in filtered_worklogs_df.columns:
                        workload = filtered_worklogs_df.groupby('Resource')['Time Spent (hrs)'].sum().sort_values(ascending=False)
                        if not workload.empty:
                            # Get color palette for consistent styling
                            colors = get_color_palette()
                            
                            # Create enhanced bar chart for workload
                            fig = px.bar(
                                x=workload.index, 
                                y=workload.values,
                                color_discrete_sequence=[colors['primary']],
                                opacity=0.85,
                                text=workload.values.round(1),
                                template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark'
                            )
                            
                            # Add hover effects
                            fig.update_traces(
                                textposition='outside',
                                hovertemplate='<b>%{x}</b><br>Hours: <b>%{y:.1f}</b><extra></extra>',
                                marker=dict(line=dict(width=0.5, color=colors['background']))
                            )
                            
                            # Apply consistent styling
                            fig = style_plotly_chart(fig, title="Resource Utilization (Hours)", height=350)
                            
                            # Additional customizations
                            fig.update_layout(
                                xaxis_title="Resource", 
                                yaxis_title="Hours",
                                xaxis=dict(tickangle=-45)
                            )
                            
                            # Add subtle grid lines only on the y-axis
                            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=colors['grid'])
                            fig.update_xaxes(showgrid=False)
                            
                            st.plotly_chart(fig, use_container_width=True, key="plotly_b0270a8601f8")
                        else:
                            st.info("No worklog data available with current filters.")
                    else:
                        st.warning("Worklog data missing required columns.")
                
                with perf_col2:
                    # Bubble Chart (Overload vs. Velocity)
                    st.subheader("Overload vs. Velocity")
                    if 'Resource' in filtered_worklogs_df.columns and 'Time Spent (hrs)' in filtered_worklogs_df.columns:
                        if 'Story Points' in filtered_issues_df.columns and 'Assignee' in filtered_issues_df.columns:
                            filtered_worklogs_df['Date'] = pd.to_datetime(filtered_worklogs_df['Date'], errors='coerce')
                            filtered_worklogs_df['Week'] = filtered_worklogs_df['Date'].dt.strftime('%Y-%U')
                            actuals = filtered_worklogs_df.groupby(['Week', 'Resource'])['Time Spent (hrs)'].sum().reset_index()
                            
                            velocity = filtered_issues_df.groupby('Assignee')['Story Points'].sum().reset_index()
                            velocity.columns = ['Resource', 'Story Points']
                            merged = pd.merge(actuals, velocity, on='Resource', how='left')
                            merged = merged.dropna()
                            
                            if not merged.empty:
                                # Get color palette for consistent styling
                                colors = get_color_palette()
                                
                                # Create enhanced bubble chart
                                fig = px.scatter(
                                    merged,
                                    x='Story Points',
                                    y='Time Spent (hrs)',
                                    size='Time Spent (hrs)',
                                    color='Resource',
                                    hover_name='Resource',
                                    opacity=0.85,
                                    template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark',
                                    color_discrete_sequence=colors['categorical']
                                )
                                
                                # Enhance hover information
                                fig.update_traces(
                                    marker=dict(line=dict(width=1, color=colors['background'])),
                                    hovertemplate='<b>%{hovertext}</b><br>Story Points: <b>%{x}</b><br>Hours: <b>%{y:.1f}</b><br>Size: <b>%{marker.size:.1f}</b> hrs<extra></extra>'
                                )
                                
                                # Apply consistent styling with bubble chart-specific enhancements
                                fig = style_plotly_chart(fig, title='Workload vs. Velocity Analysis', height=350, chart_type="bubble")
                                
                                # Use fix_axis_labels from visualization_enhancements to improve label positioning
                                from visualization_enhancements import fix_axis_labels
                                fig = fix_axis_labels(fig, x_title='Story Points Delivered', y_title='Hours Logged', colors=colors)
                                
                                # Add reference line for average ratio
                                if len(merged) > 0:
                                    avg_ratio = merged['Time Spent (hrs)'].sum() / merged['Story Points'].sum() if merged['Story Points'].sum() > 0 else 0
                                    x_range = [0, merged['Story Points'].max() * 1.1]
                                    y_range = [0, merged['Story Points'].max() * avg_ratio * 1.1]
                                    
                                    fig.add_trace(go.Scatter(
                                        x=x_range,
                                        y=[x * avg_ratio for x in x_range],
                                        mode='lines',
                                        name=f'Avg: {avg_ratio:.1f} hrs/pt',
                                        line=dict(color=colors['warning'], width=2, dash='dot'),
                                        hovertemplate='Average: <b>%{y:.1f}</b> hrs for <b>%{x}</b> points<extra></extra>'
                                    ))
                                    
                                    # Set axis ranges
                                    fig.update_xaxes(range=x_range)
                                    fig.update_yaxes(range=y_range)
                                
                                st.plotly_chart(fig, use_container_width=True, key="plotly_81ae55c62df1")
                            else:
                                st.info("Insufficient data for bubble chart with current filters.")
                        else:
                            st.warning("Issues data missing required columns.")
                    else:
                        st.warning("Worklog data missing required columns.")
            else:
                st.warning("Missing worklog or issues data.")
    
    elif dashboard_view == "Sprint Status":
        # Create tabs for different sprint views
        tab1, tab2 = st.tabs(["Timeline", "Sprint Details"])
        
        with tab1:
            # Gantt Chart
            st.subheader("📅 Gantt Chart - Timeline by Assignee")
            if filtered_issues_df is not None and not filtered_issues_df.empty:
                filtered_issues_df['Start Date'] = pd.to_datetime(filtered_issues_df['Start Date'], errors='coerce')
                filtered_issues_df['Due Date'] = pd.to_datetime(filtered_issues_df['Due Date'], errors='coerce')
                gantt_data = filtered_issues_df.dropna(subset=['Start Date', 'Due Date'])
                
                if not gantt_data.empty:
                    # Get color palette for consistent styling
                    colors = get_color_palette()
                    
                    # Create enhanced Gantt chart with improved styling
                    fig = px.timeline(
                        gantt_data,
                        x_start="Start Date",
                        x_end="Due Date",
                        y="Assignee",
                        color="Project",
                        hover_name="Summary",
                        hover_data=["Issue Key", "Status", "Priority"],
                        color_discrete_sequence=colors['categorical'],
                        opacity=0.85,
                        template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark'
                    )
                    
                    # Custom hover template with cleaner formatting
                    fig.update_traces(
                        hovertemplate='<b>%{hovertext}</b><br>' +
                                    'Issue: %{customdata[0]}<br>' +
                                    'Status: %{customdata[1]}<br>' +
                                    'Priority: %{customdata[2]}<br>' +
                                    'Start: %{x[0]|%Y-%m-%d}<br>' +
                                    'Due: %{x[1]|%Y-%m-%d}<extra></extra>',
                        marker=dict(line=dict(width=1, color=colors['background']))
                    )
                    
                    # Apply consistent styling with Gantt-specific enhancements
                    title = f"Timeline for {filters['project'] if filters['project'] != 'All Projects' else 'All Projects'}{' - ' + filters['sprint'] if filters['sprint'] != 'All Sprints' else ''}"
                    fig = style_plotly_chart(fig, title=title, height=400, chart_type="gantt")
                    
                    # Use fix_axis_labels from visualization_enhancements to improve label positioning
                    from visualization_enhancements import fix_axis_labels
                    fig = fix_axis_labels(fig, x_title='Timeline', y_title='Team Member', colors=colors)
                    
                    # Add date selector for timeline
                    today = pd.Timestamp(datetime.now()).normalize()
                    fig.update_layout(
                        xaxis=dict(
                            tickformat="%b %d",
                            rangeselector=dict(
                                buttons=list([
                                    dict(count=1, label="1m", step="month", stepmode="backward"),
                                    dict(count=3, label="3m", step="month", stepmode="backward"),
                                    dict(step="all")
                                ]),
                                bgcolor=colors['background'],
                                bordercolor=colors['grid'],
                                borderwidth=1,
                                font=dict(color=colors['text'])
                            )
                        ),
                        yaxis=dict(
                            autorange="reversed"
                        )
                    )
                    
                    # Add a vertical line for today
                    # Use our safe add_vline method to avoid Plotly errors
                    safe_add_vline(fig,
                        x=today, 
                        line_width=2, 
                        line_dash="dash", 
                        line_color=colors['primary'],
                        annotation_text="Today",
                        annotation_position="top right"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key="plotly_141c0ffb59b0")
                else:
                    st.warning("No tasks with valid start and due dates available with current filters.")
            else:
                st.warning("No issue data available with current filters.")
        
        with tab2:
            # Sprint Details
            st.subheader("Sprint Details and Progress")
            if filtered_issues_df is not None and not filtered_issues_df.empty:
                # Sprint Summary
                if filters["sprint"] != "All Sprints":
                    sprint_df = filtered_issues_df[filtered_issues_df['Sprint'] == filters["sprint"]]
                    
                    # Create metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Calculate metrics
                    total_tasks = len(sprint_df)
                    completed_tasks = len(sprint_df[sprint_df['Status'] == 'Done'])
                    completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
                    total_story_points = sprint_df['Story Points'].sum()
                    completed_points = sprint_df[sprint_df['Status'] == 'Done']['Story Points'].sum()
                    point_completion_rate = completed_points / total_story_points if total_story_points > 0 else 0
                    
                    # Display metrics
                    with col1:
                        st.metric("Total Tasks", total_tasks)
                    with col2:
                        st.metric("Completed Tasks", f"{completed_tasks} ({int(completion_rate*100)}%)")
                    with col3:
                        st.metric("Total Points", total_story_points)
                    with col4:
                        st.metric("Completed Points", f"{completed_points} ({int(point_completion_rate*100)}%)")
                    
                    # Status breakdown
                    st.subheader("Status Breakdown")
                    status_counts = sprint_df['Status'].value_counts().reset_index()
                    status_counts.columns = ['Status', 'Count']
                    
                    # Get color palette for consistent styling
                    colors = get_color_palette()
                    
                    # Create enhanced status breakdown chart
                    fig = px.bar(
                        status_counts, 
                        x='Status', 
                        y='Count', 
                        color='Status',
                        color_discrete_sequence=colors['categorical'],
                        text='Count',
                        opacity=0.85,
                        template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark'
                    )
                    
                    # Add hover effects and text labels
                    fig.update_traces(
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Tasks: <b>%{y}</b><extra></extra>',
                        marker=dict(line=dict(width=0.5, color=colors['background']))
                    )
                    
                    # Apply consistent styling
                    fig = style_plotly_chart(fig, title="Sprint Tasks by Status", height=350)
                    
                    # Add grid lines only on y-axis
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=colors['grid'])
                    fig.update_xaxes(showgrid=False)
                    
                    st.plotly_chart(fig, use_container_width=True, key="plotly_7e339947194e")
                    
                    # Task List
                    st.subheader("Sprint Tasks")
                    st.dataframe(sprint_df[['Issue Key', 'Summary', 'Assignee', 'Status', 'Priority', 'Story Points']], use_container_width=True)
                else:
                    st.info("Please select a specific sprint from the filters to view detailed sprint information.")
            else:
                st.warning("No data available for this sprint.")
    
    elif dashboard_view == "Resource Allocation":
        # Create tabs for resource views
        tab1, tab2 = st.tabs(["Workload Distribution", "Skill Coverage"])
        
        with tab1:
            # Resource Workload
            st.subheader("Resource Workload")
            if filtered_worklogs_df is not None and not filtered_worklogs_df.empty:
                # Team Workload stacked bar chart
                if 'Resource' in filtered_worklogs_df.columns and 'Date' in filtered_worklogs_df.columns and 'Time Spent (hrs)' in filtered_worklogs_df.columns:
                    filtered_worklogs_df['Date'] = pd.to_datetime(filtered_worklogs_df['Date'], errors='coerce')
                    filtered_worklogs_df['Week'] = filtered_worklogs_df['Date'].dt.strftime('%Y-%U')
                    grouped = filtered_worklogs_df.groupby(['Week', 'Resource'])['Time Spent (hrs)'].sum().reset_index()
                    
                    if not grouped.empty:
                        # Sort weeks chronologically
                        weeks = sorted(grouped['Week'].unique())
                        grouped['Week_Sorted'] = pd.Categorical(grouped['Week'], categories=weeks, ordered=True)
                        grouped = grouped.sort_values('Week_Sorted')
                        
                        # Get color palette for consistent styling
                        colors = get_color_palette()
                        
                        # Create enhanced stacked bar chart for resource utilization
                        fig = px.bar(
                            grouped,
                            x='Week',
                            y='Time Spent (hrs)',
                            color='Resource',
                            labels={'Week': 'Week', 'Time Spent (hrs)': 'Hours Worked', 'Resource': 'Team Member'},
                            color_discrete_sequence=colors['categorical'],
                            template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark',
                            text=grouped.groupby('Week')['Time Spent (hrs)'].transform('sum').round(1)
                        )
                        
                        # Enhance hover information
                        fig.update_traces(
                            textposition='outside',
                            hovertemplate='<b>%{x}</b><br>Resource: <b>%{customdata[0]}</b><br>Hours: <b>%{y:.1f}</b><extra></extra>',
                            customdata=np.column_stack([grouped['Resource']])
                        )
                        
                        # Apply consistent styling
                        fig = style_plotly_chart(fig, title='Resource Utilization by Week', height=400)
                        
                        # Update layout with stacked bar mode and improved axis labels
                        fig.update_layout(
                            barmode='stack', 
                            xaxis_title='Week', 
                            yaxis_title='Hours Worked',
                            xaxis=dict(
                                tickangle=-45,
                                type='category'
                            ),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=-0.3,
                                xanchor="center",
                                x=0.5
                            )
                        )
                        
                        # Add reference line for 40-hour workweek
                        if len(grouped) > 0:
                            team_size = len(grouped['Resource'].unique())
                            if team_size > 0:
                                fig.add_shape(
                                    type="line",
                                    x0=grouped['Week'].min(), 
                                    x1=grouped['Week'].max(),
                                    y0=40 * team_size, 
                                    y1=40 * team_size,
                                    line=dict(color=colors['warning'], width=2, dash="dash")
                                )
                                
                                fig.add_annotation(
                                    x=grouped['Week'].max(), 
                                    y=40 * team_size,
                                    text=f"Capacity ({team_size} × 40hrs)",
                                    showarrow=False,
                                    yshift=10,
                                    font=dict(color=colors['warning'])
                                )
                        
                        st.plotly_chart(fig, use_container_width=True, key="plotly_fb769d98f52f")
                    else:
                        st.info("No worklog data with current filters.")
                else:
                    st.warning("Worklog data missing required columns.")
                
                # Calendar Heat Map
                st.subheader("Calendar Heatmap - Resource Utilization")
                if 'Resource' in filtered_worklogs_df.columns and 'Date' in filtered_worklogs_df.columns:
                    # Convert to datetime
                    filtered_worklogs_df['Date'] = pd.to_datetime(filtered_worklogs_df['Date'], errors='coerce')
                    
                    # Extract day and week information
                    filtered_worklogs_df['Day'] = filtered_worklogs_df['Date'].dt.day_name()
                    filtered_worklogs_df['Week'] = filtered_worklogs_df['Date'].dt.strftime('%Y-%U')
                    
                    # Order days correctly
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    
                    # Aggregate data by day, week, and resource
                    heatmap_data = filtered_worklogs_df.groupby([filtered_worklogs_df['Week'], filtered_worklogs_df['Day'], 'Resource'])['Time Spent (hrs)'].sum().reset_index()
                    
                    # Pivot the data for the heatmap
                    if not heatmap_data.empty:
                        # Get the most recent weeks (up to 8) for better visualization
                        recent_weeks = sorted(heatmap_data['Week'].unique())[-8:]
                        recent_data = heatmap_data[heatmap_data['Week'].isin(recent_weeks)]
                        
                        # Order the days correctly
                        recent_data['Day'] = pd.Categorical(recent_data['Day'], categories=day_order, ordered=True)
                        recent_data = recent_data.sort_values(['Week', 'Day'])
                        
                        # Create a pivot table with days as rows and weeks as columns
                        pivot_data = recent_data.pivot_table(
                            index='Resource',
                            columns=['Week', 'Day'],
                            values='Time Spent (hrs)',
                            aggfunc='sum',
                            fill_value=0
                        )
                        
                        # Flatten the MultiIndex for better display
                        pivot_data.columns = [f"{week} {day[:3]}" for week, day in pivot_data.columns]
                        
                        # Display the heatmap
                        st.dataframe(
                            pivot_data.style.background_gradient(cmap='YlGnBu', axis=None),
                            use_container_width=True
                        )
                        
                        # Get color palette for consistent styling
                        colors = get_color_palette()
                        
                        # Create enhanced visual heatmap
                        fig = px.imshow(
                            pivot_data.values,
                            labels=dict(x="Day", y="Resource", color="Hours"),
                            x=pivot_data.columns,
                            y=pivot_data.index,
                            color_continuous_scale=colors['sequential'],
                            template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark'
                        )
                        
                        # Enhanced hover info and styling
                        fig.update_traces(
                            hovertemplate="<b>%{y}</b><br>" +
                                         "%{x}<br>" +
                                         "Hours: <b>%{z:.1f}</b><extra></extra>"
                        )
                        
                        # Apply consistent styling
                        fig = style_plotly_chart(fig, title="Workload Intensity Heatmap", height=400)
                        
                        # Enhance layout
                        fig.update_layout(
                            xaxis=dict(side="top", tickangle=-30),
                            yaxis=dict(title="Team Member"),
                            coloraxis_colorbar=dict(
                                title="Hours",
                                thicknessmode="pixels", thickness=20,
                                lenmode="pixels", len=300,
                                yanchor="top", y=1,
                                ticks="outside",
                                dtick=4
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key="plotly_ffc04150db42")
                    else:
                        st.info("No date-based worklog data available with current filters.")
                else:
                    st.warning("Worklog data missing required columns for heatmap.")
            else:
                st.warning("No worklog data available with current filters.")
        
        with tab2:
            # Skill Distribution
            st.subheader("Skill Distribution")
            if filtered_skills_df is not None and not filtered_skills_df.empty:
                # Standardize columns
                if 'Resource' in filtered_skills_df.columns and 'Name' not in filtered_skills_df.columns:
                    filtered_skills_df = filtered_skills_df.rename(columns={'Resource': 'Name'})
                
                if 'Name' in filtered_skills_df.columns and 'Skillset' in filtered_skills_df.columns:
                    # Get unique skillsets
                    skillsets = filtered_skills_df['Skillset'].unique()
                    
                    # Skill Distribution Overview
                    skill_counts = filtered_skills_df['Skillset'].value_counts().reset_index()
                    skill_counts.columns = ['Skill', 'Count']
                    
                    # Get color palette for consistent styling
                    colors = get_color_palette()
                    
                    # Create enhanced skill distribution bar chart
                    fig = px.bar(
                        skill_counts,
                        x='Skill',
                        y='Count',
                        color='Skill',
                        color_discrete_sequence=colors['categorical'],
                        text='Count',
                        opacity=0.85,
                        template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark'
                    )
                    
                    # Enhance hover information and add text labels
                    fig.update_traces(
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Team members: <b>%{y}</b><extra></extra>',
                        marker=dict(line=dict(width=0.5, color=colors['background']))
                    )
                    
                    # Apply consistent styling
                    fig = style_plotly_chart(fig, title="Team Skill Distribution", height=350)
                    
                    # Add grid lines only on y-axis
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=colors['grid'])
                    fig.update_xaxes(showgrid=False)
                    
                    st.plotly_chart(fig, use_container_width=True, key="plotly_e6be6e48248e")
                    
                    # Radar Chart for Each Skill Category
                    for skill_category in skillsets:
                        st.subheader(f"{skill_category} Skills Distribution")
                        skill_data = filtered_skills_df[filtered_skills_df['Skillset'] == skill_category]
                        
                        if not skill_data.empty:
                            # For all resources, count number with each skill
                            team_skill = skill_data['Name'].value_counts().reset_index()
                            team_skill.columns = ['Resource', 'Skill Count']
                            
                            # Get color palette for consistent styling
                            colors = get_color_palette()
                            
                            # Create enhanced bar chart for skill category distribution
                            fig = px.bar(
                                team_skill,
                                x='Resource',
                                y='Skill Count',
                                color='Resource',
                                color_discrete_sequence=colors['categorical'],
                                text='Skill Count',
                                opacity=0.85,
                                template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark'
                            )
                            
                            # Add hover effects and text labels
                            fig.update_traces(
                                textposition='outside',
                                hovertemplate='<b>%{x}</b><br>Skills: <b>%{y}</b><extra></extra>',
                                marker=dict(line=dict(width=0.5, color=colors['background']))
                            )
                            
                            # Apply consistent styling
                            fig = style_plotly_chart(fig, title=f"Resources with {skill_category} Skills", height=350)
                            
                            # Add grid lines only on y-axis
                            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=colors['grid'], title="Number of Skills")
                            fig.update_xaxes(showgrid=False, title="Team Member")
                            
                            st.plotly_chart(fig, use_container_width=True, key="plotly_9d7d1c291fcc")
                    
                    # Treemap of Resource Skills
                    st.subheader("Skill Distribution Treemap")
                    
                    # Get color palette for consistent styling
                    colors = get_color_palette()
                    
                    # Create enhanced treemap visualization
                    fig = px.treemap(
                        filtered_skills_df,
                        path=['Skillset', 'Name'],
                        color_discrete_sequence=colors['categorical'],
                        template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark'
                    )
                    
                    # Enhance hover information
                    fig.update_traces(
                        hovertemplate='<b>%{label}</b><br>Value: %{value}<extra></extra>',
                        marker=dict(cornerradius=5)
                    )
                    
                    # Apply consistent styling
                    fig = style_plotly_chart(fig, title="Team Skills Composition", height=500)
                    
                    # Additional styling for treemap
                    fig.update_layout(
                        margin=dict(t=50, l=25, r=25, b=25),
                        treemapcolorway=colors['categorical']
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key="plotly_2eca974a2dc7")
                else:
                    st.warning("Skills data missing required columns for visualization.")
            else:
                st.warning("No skills data available with current filters.")


# ---------- Define PM Daily Brief Function ----------
def pm_daily_brief():
    st.title("📝 Project Manager Daily Brief")
    if issues_df is None:
        st.warning("Please upload a valid JIRA Excel file.")
        return
        
    # Track whether we need to show highlighted sections
    show_overdue = st.session_state.get('show_overdue', False)
    show_due_soon = st.session_state.get('show_due_soon', False)
    
    # Reset flags after use
    if show_overdue:
        st.session_state['show_overdue'] = False
        st.info("📌 Showing overdue tasks as requested")
        
    if show_due_soon:
        st.session_state['show_due_soon'] = False
        st.info("📌 Showing tasks due this week as requested")

    today = pd.to_datetime("today").normalize()
    issues_df['Start Date'] = pd.to_datetime(issues_df['Start Date'], errors='coerce')
    issues_df['Due Date'] = pd.to_datetime(issues_df['Due Date'], errors='coerce')

    unassigned = issues_df[issues_df['Assignee'].isna()]
    due_soon = issues_df[issues_df['Due Date'].between(today, today + pd.Timedelta(days=7), inclusive='both')]
    stuck = issues_df[(issues_df['Status'] == 'In Progress') & ((today - issues_df['Start Date']).dt.days > 7)]
    missing_est = issues_df[issues_df['Original Estimate (days)'].isna() | issues_df['Story Points'].isna()]
    overdue = issues_df[issues_df['Due Date'] < today]

    st.subheader("🔧 Action Required")
    if not unassigned.empty: st.markdown("**🔲 Unassigned Tasks**"); st.dataframe(unassigned)
    
    # Add highlight box around due soon tasks if selected
    if show_due_soon and not due_soon.empty:
        with st.container():
            st.markdown("---")
            st.markdown("### 🔍 HIGHLIGHTED SECTION")
            st.markdown("**🗓 Tasks Due This Week**")
            st.dataframe(due_soon, use_container_width=True)
            st.markdown("---")
    elif not due_soon.empty:
        st.markdown("**🗓 Tasks Due This Week**")
        st.dataframe(due_soon)
        
    if not stuck.empty: st.markdown("**🔄 Stuck Tasks (In Progress > 7 days)**"); st.dataframe(stuck)

    st.subheader("🚨 Alerts & Notifications")
    if not missing_est.empty: st.markdown("**⚠️ Missing Estimates**"); st.dataframe(missing_est)
    
    # Add highlight box around overdue tasks if selected
    if show_overdue and not overdue.empty:
        with st.container():
            st.markdown("---")
            st.markdown("### 🔍 HIGHLIGHTED SECTION")
            st.markdown("**⏰ Overdue Tasks**")
            st.dataframe(overdue, use_container_width=True)
            st.markdown("---")
    elif not overdue.empty:
        st.markdown("**⏰ Overdue Tasks**")
        st.dataframe(overdue)

    st.subheader("🤖 Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Reassign unassigned tasks", key="btn_reassign"):
            st.info("This feature would automatically redistribute unassigned tasks to team members based on workload and skills. Integration with JIRA API required for full implementation.")
            
    with col2:
        if st.button("Alert overdue assignees", key="btn_alert"):
            st.info("This feature would send MS Teams notifications to team members with overdue tasks. MS Teams integration required for full implementation.")
            
    with col3:
        if st.button("Review due this week", key="btn_review"):
            st.info("This feature would schedule an MS Outlook calendar event to review upcoming tasks. MS Outlook integration required for full implementation.")

    brief = f"""
    === PROJECT MANAGER DAILY BRIEF ===
    - {len(unassigned)} unassigned tasks
    - {len(due_soon)} tasks due this week
    - {len(stuck)} tasks in progress > 7 days
    - {len(missing_est)} tasks missing estimates
    - {len(overdue)} overdue tasks
    """
    st.download_button("📄 Download Brief as TXT", brief, file_name="PM_Daily_Brief.txt")

# ---------- Define Strateg-AIz Assistant Function ----------
def ai_pm_buddy_assistant():
    st.title("🤖 Strateg-AIz")
    
    # Reference global variables
    global issues_df, skills_df, worklogs_df, leaves_df
    
    # Set up tabs for different Strateg-AIz features
    ai_tabs = st.tabs(["Ask PM Buddy", "Smart PM Brief", "What-if Simulation", "Buddy Brain", "Doc Dock"])
    
    # Use the global OpenAI client
    client = openai_client
    
    # Display a friendly message if the OpenAI client is not available
    if client is None:
        st.warning("⚠️ OpenAI API key is not configured or is invalid. The AI Assistant will operate in limited functionality mode.")
        st.markdown("""To enable all AI features, please set up your OpenAI API key:  
        1. Get an API key from [OpenAI Platform](https://platform.openai.com/api-keys)  
        2. Add it to your Streamlit secrets  
        3. Restart the application""")
        
        # Continue with limited functionality - non-AI visualizations will still work
    
    # ---------- Summarize Data ----------
    try:
        issues_summary = issues_df.describe(include='all').to_string() if issues_df is not None else ""
        worklog_summary = worklogs_df.groupby('Resource')['Time Spent (hrs)'].sum().to_string() if worklogs_df is not None else ""
        skill_distribution = skills_df['Skillset'].value_counts().to_string() if skills_df is not None else ""
        leave_summary = leaves_df['Resource'].value_counts().to_string() if leaves_df is not None else ""
        
        # ---------- Ask PM Buddy Tab ----------
        with ai_tabs[0]:
            st.subheader("📋 Ask PM Buddy")
            st.markdown("""
            Ask any project management question. The AI PM Buddy will analyze your JIRA data and provide insights.
            """)
            
            # Create a structured guided question interface
            st.markdown("### Guided Question Builder")
            
            # Two column layout for better organization
            config_col1, config_col2 = st.columns([1, 1])
            
            with config_col1:
                # Role-based responses selection
                pm_roles = [
                    "Project Manager",
                    "Scrum Master",
                    "Product Owner",
                    "Team Lead",
                    "Resource Manager",
                    "Executive Sponsor",
                    "Technical Lead"
                ]
                selected_role = st.selectbox("👤 Select perspective:", pm_roles, index=0)
                
                # Question category to guide question building
                question_categories = [
                    "Resource Analysis",
                    "Timeline Assessment", 
                    "Risk Management",
                    "Sprint Planning",
                    "Team Performance",
                    "Quality Metrics",
                    "Stakeholder Communication",
                    "Custom Question"
                ]
                selected_category = st.selectbox("🔍 Question category:", question_categories)
            
            with config_col2:
                # Time period selection for context
                time_periods = [
                    "Current Sprint",
                    "Next Sprint",
                    "Current Month",
                    "Next Month",
                    "Current Quarter",
                    "Next Quarter",
                    "Entire Project"
                ]
                selected_timeframe = st.selectbox("⏱ Time period:", time_periods)
                
                # Resource focus if applicable
                if issues_df is not None and 'Assignee' in issues_df.columns:
                    resources = ["All Resources"] + sorted(issues_df['Assignee'].dropna().unique().tolist())
                else:
                    resources = ["All Resources", "Team A", "Team B", "Development", "QA", "Design"]
                
                selected_resource = st.selectbox("👥 Resource focus:", resources)
            
            # Role-specific question suggestions based on perspective
            role_questions = {
                "Project Manager": [
                    "What is the overall health of the project?",
                    "Which risks need immediate mitigation?",
                    "How balanced is the workload across the team?",
                    "Which tasks are at risk of missing their deadlines?",
                    "What is our budget burn rate compared to project progress?",
                    "Who needs help with their current assignments?"
                ],
                "Scrum Master": [
                    "What impediments are blocking the team?",
                    "How is the team's velocity trending?",
                    "Which ceremonies need improvement?",
                    "Are there bottlenecks in our development process?",
                    "How can we improve our sprint retrospective process?",
                    "Which team members need coaching support?"
                ],
                "Product Owner": [
                    "Which features will deliver the most customer value?",
                    "What is the projected delivery date for the MVP?",
                    "How should we prioritize the backlog for maximum ROI?",
                    "What customer needs are not being addressed in the current sprint?",
                    "Which features are at risk based on technical complexity?",
                    "How is our feature completion rate trending?"
                ],
                "Team Lead": [
                    "How is my team performing compared to previous sprints?",
                    "Which team members need additional support?",
                    "What skill gaps exist in my team?",
                    "How can I better distribute work among team members?",
                    "What technical challenges are slowing down my team?",
                    "How can I improve team collaboration?"
                ],
                "Resource Manager": [
                    "Who is over-allocated and needs workload reduction?",
                    "Which projects need additional resources?",
                    "How can we optimize resource allocation across projects?",
                    "What is the impact of upcoming leave schedules?",
                    "Which resources have specialized skills that create bottlenecks?",
                    "How can we balance project staffing needs?"
                ],
                "Executive Sponsor": [
                    "What is the overall ROI projection for this project?",
                    "Are we on track to meet our strategic objectives?",
                    "What major risks should I be aware of?",
                    "How does this project compare to others in the portfolio?",
                    "What resource investments would yield the highest returns?",
                    "Are there any escalations that need my attention?"
                ],
                "Technical Lead": [
                    "What is our current technical debt situation?",
                    "Which architectural decisions need review?",
                    "Are there quality issues that need addressing?",
                    "How can we improve our development infrastructure?",
                    "What emerging technical challenges should we prepare for?",
                    "Which technical skills do we need to strengthen in the team?"
                ]
            }
            
            # Dynamic question suggestions based on category
            category_questions = {
                "Resource Analysis": [
                    "What are the current resource overload risks and how to mitigate them?",
                    "Which team members are underutilized and how can we better assign tasks to them?",
                    "What is the recommended reallocation strategy to optimize team utilization?",
                    "How does current resource utilization compare to previous sprints?"
                ],
                "Timeline Assessment": [
                    "What is the forecast for project completion based on current progress?",
                    "Which milestones are at risk of delay and why?",
                    "How might our timeline change if we added more resources?",
                    "What critical path dependencies might impact our timeline?"
                ],
                "Risk Management": [
                    "What are the top 3 project risks right now?",
                    "What contingency plans should we have for identified risks?",
                    "How will upcoming resource leaves impact project delivery?",
                    "What risk mitigation strategies do you recommend prioritizing?"
                ],
                "Sprint Planning": [
                    "How should we reallocate tasks to meet sprint deadlines?",
                    "What's the optimal distribution of story points for the next sprint?",
                    "Which backlog items should we prioritize for the next sprint?",
                    "How can we balance technical debt reduction with new feature development?"
                ],
                "Team Performance": [
                    "How can we increase velocity without increasing burnout risk?",
                    "Which team members are performing above/below average and why?",
                    "What skills should the team develop to improve overall performance?",
                    "How does our cycle time compare to industry benchmarks?"
                ],
                "Quality Metrics": [
                    "What is our current defect density and how can we improve it?",
                    "Are there any areas with higher than average technical debt?",
                    "How has our code quality changed over recent sprints?",
                    "What testing coverage improvements should we prioritize?"
                ],
                "Stakeholder Communication": [
                    "What key updates should be communicated to stakeholders this week?",
                    "How should we address stakeholder concerns about current progress?",
                    "What visualization would best communicate our current status to executives?",
                    "What expectations should we set with stakeholders given current metrics?"
                ],
                "Custom Question": [""]
            }
            
            # Show "Questions by Role" and "Questions by Category" tabs
            question_tabs = st.tabs(["Questions by Role", "Questions by Category"])
            
            with question_tabs[0]:  # Role-based questions tab
                if selected_role in role_questions:
                    # Get questions specific to the selected role
                    role_specific_questions = role_questions[selected_role]
                    selected_role_question = st.selectbox(
                        "📝 Suggested questions for " + selected_role + ":", 
                        ["-- Select a suggested question --"] + role_specific_questions,
                        key="role_question_selector"
                    )
                    
                    if selected_role_question != "-- Select a suggested question --":
                        # Customize question with selected timeframe and resource if needed
                        if selected_resource != "All Resources" and "team" in selected_role_question.lower():
                            customized_question = selected_role_question.replace("the team", selected_resource).replace("my team", selected_resource)
                        else:
                            customized_question = selected_role_question
                        
                        if "current" in customized_question.lower() and selected_timeframe != "Current Sprint":
                            customized_question = customized_question.replace("current", selected_timeframe.lower())
                        
                        st.session_state["selected_question"] = customized_question
                else:
                    st.info(f"No specific questions available for {selected_role}. Please select a different role or use the category-based questions.")
            
            with question_tabs[1]:  # Category-based questions tab
                if selected_category != "Custom Question":
                    suggested_questions = category_questions.get(selected_category, [])
                    selected_question = st.selectbox(
                        "📝 Suggested questions for " + selected_category + ":", 
                        ["-- Select a suggested question --"] + suggested_questions,
                        key="category_question_selector"
                    )
                    
                    if selected_question != "-- Select a suggested question --":
                        # Customize question with selected timeframe and resource if needed
                        if selected_resource != "All Resources" and "resource" in selected_question.lower():
                            customized_question = selected_question.replace("team members", selected_resource).replace("resources", selected_resource)
                        else:
                            customized_question = selected_question
                        
                        if "current" in customized_question.lower() and selected_timeframe != "Current Sprint":
                            customized_question = customized_question.replace("current", selected_timeframe.lower())
                        
                        st.session_state["selected_question"] = customized_question
                else:
                    st.info("Enter your custom question in the text area below.")
                    st.session_state["selected_question"] = ""
            
            # Get the final question text
            default_text = st.session_state.get("selected_question", "What are the key risks in current sprint and how can they be mitigated?")
            user_question = st.text_area("📝 Your final question:", value=default_text, height=100)
            
            # Add context display
            query_context = f"*Question will be answered from the perspective of a **{selected_role}**, focusing on **{selected_resource}** during the **{selected_timeframe}**.*"
            st.markdown(query_context)
            
            # Execute button with prominent styling
            submit_button = st.button("🚀 Ask AI PM Buddy", use_container_width=True)
                
            if submit_button and user_question:
                with st.spinner("PM Buddy is thinking..."):
                    # Only proceed if we have data
                    if issues_df is None or worklogs_df is None:
                        st.error("Please upload JIRA data first.")
                    else:
                        # Generate system prompt with data summaries
                        system_prompt = f"""You are an AI assistant specialized in analyzing JIRA project data and providing actionable insights from a {selected_role}'s perspective.  
                        Use the following data summaries to inform your responses:
                        
                        ISSUES SUMMARY:
                        {issues_summary}
                        
                        WORKLOG SUMMARY:
                        {worklog_summary}
                        
                        SKILL DISTRIBUTION:
                        {skill_distribution}
                        
                        LEAVE SUMMARY:
                        {leave_summary}
                        
                        Answer the user's question based on this data and provide specific, actionable insights. If you cannot answer based on the provided data, clearly state what information would be needed. Format your response with markdown for better readability.
                        """
                        
                        try:
                            st.markdown("### PM Buddy's Response:")
                            
                            # Placeholder for API response (simulated for now)
                            if client is None:
                                response = f"""It appears that your dataset contains information about project issues, resource workloads, skill distribution, and team member availability.
                                
                                Based on my analysis of your data, here are some insights related to your question about \"{user_question}\":
                                
                                *Please note: To get actual AI-generated insights, please configure your OpenAI API key in the Streamlit secrets.*
                                """
                            else:
                                # Optimize prompts to reduce token usage if needed
                                optimized_system_prompt = optimize_prompt(system_prompt)
                                
                                # Actual OpenAI API call
                                api_response = client.chat.completions.create(
                                    model="gpt-4o", # the newest OpenAI model (released May 13, 2024)
                                    messages=[
                                        {"role": "system", "content": optimized_system_prompt},
                                        {"role": "user", "content": user_question}
                                    ],
                                    temperature=0.7
                                )
                                
                                # Extract the content from the response
                                response = api_response.choices[0].message.content
                                
                                # Track token usage
                                token_manager.track_usage(api_response)
                            
                            # Display the response
                            st.markdown(response)
                            
                            # Save to conversation history
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            if 'chat_session' not in st.session_state:
                                st.session_state['chat_session'] = []
                            
                            st.session_state['chat_session'].append({
                                "question": user_question,
                                "answer": response,
                                "timestamp": timestamp,
                                "role": selected_role
                            })
                            
                        except Exception as e:
                            st.error(f"Error generating response: {e}")
                            st.info("To use the AI PM Buddy feature, ensure you have a valid OpenAI API key configured in your Streamlit secrets.")
        
        # ---------- Smart PM Brief Tab ----------
        with ai_tabs[1]:
            st.subheader("Smart PM Brief")
            st.markdown("""
            Generate AI-powered project management briefs to share with your team or stakeholders.
            """)
            
            # Brief types
            brief_type = st.selectbox(
                "Brief Type",
                ["Daily Status Update", "Weekly Progress Report", "Risk Assessment", "Resource Allocation Summary", "Sprint Readiness", "Executive Summary"]
            )
            
            # Audience selection
            audience = st.selectbox(
                "Target Audience",
                ["Team Members", "Project Stakeholders", "Senior Management", "Clients"]
            )
            
            # Level of detail
            detail_level = st.select_slider(
                "Level of Detail",
                options=["Concise", "Moderate", "Detailed"]
            )
            
            # Generate button
            if st.button("Generate Smart Brief"):
                with st.spinner("Generating smart brief..."):
                    # Only proceed if we have data
                    if issues_df is None or worklogs_df is None:
                        st.error("Please upload JIRA data first.")
                    else:
                        try:
                            # Create content for the brief based on our data
                            today = pd.to_datetime("today").normalize()
                            issues_df['Start Date'] = pd.to_datetime(issues_df['Start Date'], errors='coerce')
                            issues_df['Due Date'] = pd.to_datetime(issues_df['Due Date'], errors='coerce')
                            
                            # Basic metrics for all brief types
                            metrics = {
                                "total_issues": len(issues_df) if issues_df is not None else 0,
                                "open_issues": len(issues_df[issues_df['Status'] != 'Done']) if issues_df is not None else 0,
                                "issues_due_this_week": len(issues_df[issues_df['Due Date'].between(today, today + pd.Timedelta(days=7), inclusive='both')]) if issues_df is not None else 0,
                                "overdue_issues": len(issues_df[(issues_df['Due Date'] < today) & (issues_df['Status'] != 'Done')]) if issues_df is not None else 0
                            }
                            
                            # System prompt based on brief type
                            system_prompt = f"""You are an AI assistant that generates project management briefs based on JIRA data.  
                            Generate a {detail_level.lower()} {brief_type} for {audience}.
                            
                            Use the following metrics and data summaries:
                            - Total Issues: {metrics['total_issues']}
                            - Open Issues: {metrics['open_issues']}
                            - Issues Due This Week: {metrics['issues_due_this_week']}
                            - Overdue Issues: {metrics['overdue_issues']}
                            
                            ISSUES SUMMARY:
                            {issues_summary}
                            
                            WORKLOG SUMMARY:
                            {worklog_summary}
                            
                            Format the brief appropriately for a {brief_type} with markdown. Include relevant metrics, insights, and recommendations based on the data. If relevant, suggest action items.
                            """
                            
                            # Display the brief
                            st.markdown("### Generated Brief")
                            
                            # Placeholder for API response (simulated for now)
                            if client is None:
                                brief_content = f"""# {brief_type}
                                **For:** {audience}  
                                **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
                                
                                ## Project Status
                                - **Total Issues:** {metrics['total_issues']}
                                - **Open Issues:** {metrics['open_issues']}
                                - **Issues Due This Week:** {metrics['issues_due_this_week']}
                                - **Overdue Issues:** {metrics['overdue_issues']}
                                
                                *Please note: To get actual AI-generated briefs, please configure your OpenAI API key in the Streamlit secrets.*
                                """
                            else:
                                # Optimize prompts if needed
                                optimized_system_prompt = optimize_prompt(system_prompt)
                                
                                # Actual OpenAI API call
                                api_response = client.chat.completions.create(
                                    model="gpt-4o", # the newest OpenAI model (released May 13, 2024)
                                    messages=[
                                        {"role": "system", "content": optimized_system_prompt},
                                        {"role": "user", "content": f"Generate a {brief_type} for {audience}."}
                                    ],
                                    temperature=0.7
                                )
                                
                                # Extract the content from the response
                                brief_content = api_response.choices[0].message.content
                                
                                # Track token usage
                                token_manager.track_usage(api_response)
                            
                            st.markdown(brief_content)
                            
                            # Save to briefs history
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            if 'generated_briefs' not in st.session_state:
                                st.session_state['generated_briefs'] = []
                            
                            st.session_state['generated_briefs'].append({
                                "type": brief_type,
                                "audience": audience,
                                "content": brief_content,
                                "timestamp": timestamp
                            })
                            
                            # Add download button for the brief
                            st.download_button(
                                label="📥 Download Brief",
                                data=brief_content,
                                file_name=f"PM_Brief_{brief_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                mime="text/markdown"
                            )
                            
                            # Add option to generate PDF
                            if st.button("Generate PDF"):
                                try:
                                    # Create PDF
                                    class PDF(FPDF):
                                        def header(self):
                                            self.set_font('Arial', 'B', 12)
                                            self.cell(0, 10, brief_type, 0, 1, 'C')
                                            self.ln(10)
                                            
                                        def footer(self):
                                            self.set_y(-15)
                                            self.set_font('Arial', 'I', 8)
                                            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
                                    
                                    pdf = PDF()
                                    pdf.add_page()
                                    pdf.set_font('Arial', '', 11)
                                    pdf.multi_cell(0, 10, brief_content.replace('#', '').replace('*', ''))
                                    
                                    # Get PDF content as bytes
                                    pdf_content = pdf.output(dest='S').encode('latin-1')
                                    
                                    # Create download button for PDF
                                    st.download_button(
                                        label="📥 Download PDF",
                                        data=pdf_content,
                                        file_name=f"PM_Brief_{brief_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                        mime="application/pdf"
                                    )
                                except Exception as e:
                                    st.error(f"Error generating PDF: {e}")
                                    
                        except Exception as e:
                            st.error(f"Error generating brief: {e}")
                            st.info("To use the Smart PM Brief feature, ensure you have a valid OpenAI API key configured in your Streamlit secrets.")
        
        # ---------- What-if Simulation Tab ----------
        with ai_tabs[2]:
            st.subheader("What-if Simulation")
            st.markdown("""
            Simulate the impact of changes to team composition, leave schedules, or project priorities.
            """)
            
            # Simulation type selection
            simulation_type = st.selectbox(
                "Simulation Type",
                ["Resource Absence", "Project Deadline Change", "Team Composition Change", "Priority Shift", "Budget Change"]
            )
            
            # Create appropriate UI elements based on simulation type
            if simulation_type == "Resource Absence":
                if leaves_df is not None and 'Resource' in leaves_df.columns:
                    available_resources = sorted(leaves_df['Resource'].unique().tolist())
                elif skills_df is not None and 'Resource' in skills_df.columns:
                    available_resources = sorted(skills_df['Resource'].unique().tolist())
                elif worklogs_df is not None and 'Resource' in worklogs_df.columns:
                    available_resources = sorted(worklogs_df['Resource'].unique().tolist())
                elif issues_df is not None and 'Assignee' in issues_df.columns:
                    available_resources = sorted(issues_df['Assignee'].dropna().unique().tolist())
                else:
                    available_resources = ["No resources found in data"]
                
                # Create a resource selection interface with roles displayed
                st.markdown("### Select Resources for Absence Simulation")
                
                # Get resource roles from skills dataframe
                resource_roles = {}
                if skills_df is not None and 'Resource' in skills_df.columns and 'Skillset' in skills_df.columns:
                    for _, row in skills_df.iterrows():
                        resource_roles[row['Resource']] = row['Skillset']
                
                # Create a dataframe for resource selection with roles
                selection_data = []
                for resource in available_resources:
                    role = resource_roles.get(resource, "Unknown")
                    selection_data.append({"Resource": resource, "Role": role})
                
                selection_df = pd.DataFrame(selection_data)
                
                # Display the resource selection table
                st.dataframe(selection_df, use_container_width=True)
                
                # Allow multi-selection of resources
                selected_resources = st.multiselect("Select Resources for Absence", available_resources)
                
                # Date selection for absence period
                today = datetime.now().date()
                col1, col2 = st.columns(2)
                with col1:
                    absence_start = st.date_input("Absence Start Date", value=today)
                with col2:
                    absence_end = st.date_input("Absence End Date", value=today + timedelta(days=5))
                    
                # Calculate absence duration in days
                absence_duration = (absence_end - absence_start).days
                if absence_duration < 0:
                    st.error("End date must be after start date!")
                else:
                    st.info(f"Selected absence period: {absence_duration} days")
                    
                # Display roles of selected resources
                if selected_resources:
                    st.markdown("### Selected Resources and Roles:")
                    selected_data = []
                    for resource in selected_resources:
                        role = resource_roles.get(resource, "Unknown")
                        selected_data.append({"Resource": resource, "Role": role})
                    
                    selected_df = pd.DataFrame(selected_data)
                    st.dataframe(selected_df, use_container_width=True)
                
                if st.button("Run Simulation"):
                    with st.spinner("Running simulation..."):
                        # Only proceed if resources are selected
                        if not selected_resources:
                            st.error("Please select at least one resource.")
                        else:
                            try:
                                st.markdown("### Simulation Results")
                                st.markdown(f"**Simulating absence of:** {', '.join(selected_resources)}")
                                st.markdown(f"**Duration:** {absence_duration} days")
                                
                                # Placeholder for actual simulation results
                                if client is None:
                                    simulation_result = f"""## Impact Assessment
                                    
                                    **Timeline Impact:**
                                    - Estimated project delay: ~{absence_duration//2} days
                                    - {len(selected_resources)*3} tasks will be affected
                                    
                                    **Resource Allocation:**
                                    - Workload redistribution needed for {len(selected_resources)*3} tasks
                                    - Key skill gaps: Documentation, Testing
                                    
                                    **Recommendations:**
                                    - Consider postponing deliverable deadlines by 1 week
                                    - Temporarily reassign critical tasks to Team Members A and B
                                    - Prioritize tasks XYZ-123, ABC-456 during the absence period
                                    
                                    *Please note: To get actual AI-generated simulations, please configure your OpenAI API key in the Streamlit secrets.*
                                    """
                                else:
                                    # Prepare data for simulation
                                    affected_tasks = issues_df[issues_df['Assignee'].isin(selected_resources) & (issues_df['Status'] != 'Done')]
                                    affected_count = len(affected_tasks)
                                    
                                    # Create prompt for simulation
                                    system_prompt = f"""You are an AI assistant that simulates the impact of resource absence on project timelines and team workload.
                                    
                                    SIMULATION PARAMETERS:
                                    - Resources: {', '.join(selected_resources)}
                                    - Absence Duration: {absence_duration} days
                                    - Affected Tasks: {affected_count}
                                    
                                    AFFECTED TASK DATA:
                                    {affected_tasks.to_string() if not affected_tasks.empty else 'No tasks affected'}
                                    
                                    TEAM WORKLOAD:
                                    {worklog_summary}
                                    
                                    SKILL DISTRIBUTION:
                                    {skill_distribution}
                                    
                                    Generate a detailed impact assessment that includes:
                                    1. Timeline Impact - How will this affect project deadlines?
                                    2. Resource Allocation - How should work be redistributed?
                                    3. Specific Recommendations - What actions should be taken to mitigate impact?
                                    
                                    Format your response with markdown for better readability. Be specific about task IDs, team members, and dates where possible.
                                    """
                                    
                                    # Optimize prompts if needed
                                    optimized_system_prompt = optimize_prompt(system_prompt)
                                    
                                    # OpenAI API call
                                    api_response = client.chat.completions.create(
                                        model="gpt-4o", # the newest OpenAI model (released May 13, 2024)
                                        messages=[
                                            {"role": "system", "content": optimized_system_prompt},
                                            {"role": "user", "content": f"Simulate the impact of {', '.join(selected_resources)} being absent for {absence_duration} days."}
                                        ],
                                        temperature=0.7
                                    )
                                    
                                    # Extract the content from the response
                                    simulation_result = api_response.choices[0].message.content
                                    
                                    # Track token usage
                                    token_manager.track_usage(api_response)
                                
                                st.markdown(simulation_result)
                                
                                # Save to simulation history
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                if 'simulation_history' not in st.session_state:
                                    st.session_state['simulation_history'] = []
                                
                                st.session_state['simulation_history'].append({
                                    "type": simulation_type,
                                    "resources": selected_resources,
                                    "duration": f"{absence_duration} days",
                                    "result": simulation_result,
                                    "timestamp": timestamp
                                })
                                
                                # Add download button for the simulation results
                                st.download_button(
                                    label="📥 Download Simulation Report",
                                    data=f"# {simulation_type} Simulation\nResources: {', '.join(selected_resources)}\nDuration: {absence_duration} days\n\n{simulation_result}",
                                    file_name=f"Simulation_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    mime="text/plain"
                                )
                                
                            except Exception as e:
                                st.error(f"Error running simulation: {e}")
                                st.info("To use the simulation feature, ensure you have valid data and OpenAI API key configured.")
            
            elif simulation_type == "Project Deadline Change":
                st.markdown("### Project Deadline Change Simulation")
                st.markdown("Analyze the impact of changes to project deadlines and delivery dates.")
                
                # Get project keys and names
                if issues_df is not None and 'Project' in issues_df.columns:
                    project_keys = sorted(issues_df['Project'].unique().tolist())
                else:
                    project_keys = ["No projects found"]
                
                # Project and date selection
                selected_project = st.selectbox("Select Project", project_keys)
                
                # Current project deadline
                if issues_df is not None and selected_project in project_keys:
                    project_tasks = issues_df[issues_df['Project'] == selected_project]
                    if 'Due Date' in project_tasks.columns:
                        project_due_dates = project_tasks['Due Date'].dropna()
                        if len(project_due_dates) > 0:
                            latest_due_date = project_due_dates.max()
                            st.info(f"Current latest project deadline: {latest_due_date.strftime('%Y-%m-%d')}")
                        else:
                            latest_due_date = datetime.now().date() + timedelta(days=30)
                            st.warning("No due dates found for this project. Using default date.")
                    else:
                        latest_due_date = datetime.now().date() + timedelta(days=30)
                        st.warning("No due date information found for this project.")
                else:
                    latest_due_date = datetime.now().date() + timedelta(days=30)
                
                # Date selection for new deadline
                new_deadline = st.date_input("New Project Deadline", 
                                         value=latest_due_date.date() if isinstance(latest_due_date, datetime) else datetime.now().date() + timedelta(days=30))
                
                # Calculate the shift in days
                if isinstance(latest_due_date, datetime):
                    deadline_shift = (new_deadline - latest_due_date.date()).days
                    shift_type = "extension" if deadline_shift > 0 else "reduction"
                    st.info(f"Deadline {shift_type} of {abs(deadline_shift)} days")
                
                # Simulation settings
                st.markdown("### Simulation Settings")
                adjust_resources = st.checkbox("Adjust team resources", value=False)
                prioritize_critical = st.checkbox("Prioritize critical path tasks", value=True)
                reschedule_tasks = st.checkbox("Automatically reschedule dependent tasks", value=True)
                
                if st.button("Run Deadline Simulation"):
                    with st.spinner("Running simulation..."):
                        try:
                            st.markdown("### Simulation Results")
                            
                            if client is None:
                                # Generate enhanced placeholder simulation results if OpenAI client is not available
                                shift_description = f"{abs(deadline_shift)} day {shift_type}" if 'deadline_shift' in locals() else "deadline change"
                                
                                # Prepare task details for display
                                affected_tasks_details = ""
                                rescheduled_tasks_details = ""
                                critical_tasks_details = ""
                                
                                if 'project_tasks' in locals() and not project_tasks.empty and 'task_details' in locals() and task_details:
                                    # Create a list of affected tasks
                                    affected_tasks_details = "\n| Task ID | Summary | Current Due Date | New Due Date |\n| --- | --- | --- | --- |\n"
                                    
                                    for task in task_details[:5]:  # Limit to first 5 tasks for clarity
                                        task_id = task["key"]
                                        summary = task["summary"]
                                        current_date = task["due_date"]
                                        
                                        # Calculate new date based on deadline shift
                                        if 'deadline_shift' in locals() and current_date and current_date != "None" and current_date != "nan":
                                            try:
                                                current_dt = datetime.strptime(current_date, '%Y-%m-%d')
                                                new_date = (current_dt + timedelta(days=deadline_shift)).strftime('%Y-%m-%d')
                                            except:
                                                new_date = "Recalculation needed"
                                        else:
                                            new_date = "Recalculation needed"
                                            
                                        affected_tasks_details += f"| {task_id} | {summary} | {current_date} | {new_date} |\n"
                                    
                                    # Create critical path tasks section
                                    critical_tasks = [t for t in task_details if t["priority"] in ["Highest", "High"]]
                                    if critical_tasks:
                                        critical_tasks_details = "\n### Critical Path Tasks Requiring Attention\n\n"
                                        critical_tasks_details += "| Task ID | Summary | Priority | Status |\n| --- | --- | --- | --- |\n"
                                        
                                        for task in critical_tasks:
                                            critical_tasks_details += f"| {task['key']} | {task['summary']} | {task['priority']} | {task['status']} |\n"
                                    
                                    # Create dependency visualization
                                    dependency_tasks = [t for t in task_details if t.get("dependencies") and str(t.get("dependencies")) != "nan" and str(t.get("dependencies")) != "None"]
                                    if dependency_tasks:
                                        rescheduled_tasks_details = "\n### Task Dependencies Requiring Rescheduling\n\n"
                                        rescheduled_tasks_details += "| Task ID | Summary | Dependencies |\n| --- | --- | --- |\n"
                                        
                                        for task in dependency_tasks:
                                            rescheduled_tasks_details += f"| {task['key']} | {task['summary']} | {task['dependencies']} |\n"
                                
                                simulation_result = f"""## Deadline Change Impact Assessment
                                
                                **Timeline Impact:**
                                - Project deadline {shift_type if 'shift_type' in locals() else 'change'}: {shift_description}
                                - {len(project_tasks) if 'project_tasks' in locals() else 'Several'} tasks affected
                                
                                ### Affected Tasks and Rescheduling
                                {affected_tasks_details}
                                
                                {critical_tasks_details}
                                
                                {rescheduled_tasks_details}
                                
                                **Resource Implications:**
                                - {'Resource adjustments needed' if adjust_resources else 'No resource adjustments requested'}
                                - {'Critical path tasks prioritized' if prioritize_critical else 'Standard task prioritization maintained'}
                                - {'Dependent tasks will be rescheduled' if reschedule_tasks else 'Manual rescheduling recommended'}
                                
                                **Risk Assessment:**
                                - {'HIGH RISK: Timeline compression increases project risk significantly' if 'deadline_shift' in locals() and deadline_shift < -14 else ('MEDIUM RISK: Timeline adjustments require careful planning' if 'deadline_shift' in locals() and deadline_shift < 0 else 'LOW RISK: Timeline extension provides additional buffer')}
                                - {'Risk of resource overallocation' if adjust_resources and 'deadline_shift' in locals() and deadline_shift < 0 else 'Normal resource allocation risk'}
                                - {'Risk of dependency conflicts' if reschedule_tasks and 'dependency_tasks' in locals() and len(dependency_tasks) > 0 else 'Minimal dependency risk'}
                                
                                **Actionable Recommendations:**
                                1. {'Review and reprioritize all ' + str(len(project_tasks) if 'project_tasks' in locals() else 'project') + ' tasks based on new deadline'}
                                2. {'Adjust team resources to focus on critical path tasks' if adjust_resources else 'Maintain current resource allocations'}
                                3. {'Reorganize sprint plan to accommodate the ' + shift_description if 'shift_description' in locals() else 'Update sprint planning'}
                                4. {'Identify tasks that can be descoped or simplified' if 'deadline_shift' in locals() and deadline_shift < 0 else 'Review task estimations for accuracy'}
                                5. {'Schedule dedicated time for dependency resolution' if reschedule_tasks else 'Monitor task dependencies closely'}
                                
                                **Timeline Visualization: Gantt Chart Simulation**

                                | Task Key | Original Dates | New Dates |
                                |-----------|-------------------|-------------------|
                                | Task-1 | Original Date | New Date |
                                | Task-2 | Original Date | New Date |
                                | Task-3 | Original Date | New Date |

                                **Summary:** {latest_due_date.strftime('%m/%d/%Y') if isinstance(latest_due_date, datetime) else 'Original'} → {new_deadline.strftime('%m/%d/%Y')}
                                {'Timeline compressed by ' + str(abs(deadline_shift)) + ' days' if 'deadline_shift' in locals() and deadline_shift < 0 else ('Timeline extended by ' + str(deadline_shift) + ' days' if 'deadline_shift' in locals() else 'Timeline adjusted')}
                                
                                *Note: To get AI-powered simulations with deeper insights, please configure your OpenAI API key in the Streamlit secrets.*
                                """
                            else:
                                # Prepare data for simulation
                                if 'project_tasks' in locals() and not project_tasks.empty:
                                    task_count = len(project_tasks)
                                    critical_tasks = project_tasks[project_tasks['Priority'].isin(['Highest', 'High'])].shape[0] if 'Priority' in project_tasks.columns else 0
                                    has_dependencies = 'Dependencies' in project_tasks.columns and project_tasks['Dependencies'].notna().any()
                                    workload_info = f"Current team capacity: {len(available_resources) if 'available_resources' in locals() else 'Unknown'} resources" 
                                else:
                                    task_count = 0
                                    critical_tasks = 0
                                    has_dependencies = False
                                    workload_info = "No workload data available"
                                
                                # Extract specific task information for the simulation
                                task_details = []
                                critical_path_tasks = []
                                dependency_map = {}
                                task_effort_map = {}
                                task_due_dates = {}
                                
                                if 'project_tasks' in locals() and not project_tasks.empty:
                                    for _, task in project_tasks.iterrows():
                                        issue_key = task['Issue Key'] if 'Issue Key' in task else f"Task-{_}"
                                        summary = task['Summary'] if 'Summary' in task else "Unnamed Task"
                                        status = task['Status'] if 'Status' in task else "Unknown"
                                        priority = task['Priority'] if 'Priority' in task else "Medium"
                                        due_date = task['Due Date'] if 'Due Date' in task else None
                                        effort = task['Original Estimate (days)'] if 'Original Estimate (days)' in task else None
                                        dependencies = task['Dependencies'] if 'Dependencies' in task and pd.notna(task['Dependencies']) else None
                                        
                                        # Add to maps
                                        if due_date is not None:
                                            task_due_dates[issue_key] = due_date
                                        if effort is not None and not pd.isna(effort):
                                            task_effort_map[issue_key] = effort
                                        if dependencies is not None:
                                            dependency_list = str(dependencies).split(',')
                                            dependency_map[issue_key] = [d.strip() for d in dependency_list]
                                        if priority in ['Highest', 'High']:
                                            critical_path_tasks.append(issue_key)
                                        
                                        # Create task detail record
                                        task_details.append({
                                            "key": issue_key,
                                            "summary": summary,
                                            "status": status,
                                            "priority": priority,
                                            "due_date": due_date.strftime('%Y-%m-%d') if isinstance(due_date, (datetime, pd.Timestamp)) else str(due_date),
                                            "effort": str(effort) if effort is not None and not pd.isna(effort) else "Unknown",
                                            "dependencies": dependencies
                                        })
                                
                                # Prepare task details as JSON string for the prompt
                                import json
                                task_details_json = json.dumps(task_details[:10]) # Limit to 10 tasks to avoid token limit
                                
                                # Create prompt for simulation
                                system_prompt = f"""You are an AI assistant that simulates the impact of project deadline changes.
                                
                                SIMULATION PARAMETERS:
                                - Project: {selected_project}
                                - Current Deadline: {latest_due_date.strftime('%Y-%m-%d') if isinstance(latest_due_date, datetime) else 'Unknown'}
                                - New Deadline: {new_deadline.strftime('%Y-%m-%d')}
                                - Deadline Shift: {deadline_shift if 'deadline_shift' in locals() else 'Unknown'} days
                                - Task Count: {task_count}
                                - Critical Tasks: {critical_tasks}
                                - Has Dependencies: {has_dependencies}
                                
                                SIMULATION SETTINGS:
                                - Adjust Resources: {adjust_resources}
                                - Prioritize Critical Path: {prioritize_critical}
                                - Reschedule Dependencies: {reschedule_tasks}
                                
                                TEAM WORKLOAD:
                                {workload_info}
                                
                                TASK DETAILS:
                                {task_details_json}
                                
                                Generate a detailed impact assessment that includes:
                                1. Timeline Impact - Specifically explain how this deadline change affects project milestones and individual tasks.
                                2. Task Rescheduling - Provide a concrete list of tasks that need to be rescheduled with original and new dates. 
                                3. Critical Path Analysis - Identify which tasks become critical under the new deadline and need special attention.
                                4. Resource Implications - Specify which team members will need different allocations and by how much.
                                5. Risk Assessment - List specific risks introduced by this change and their mitigation strategies.
                                6. Actionable Recommendations - Provide a numbered list of specific actions with estimated effort required.
                                
                                Format your response with markdown for better readability. Use tables where appropriate to show task rescheduling.
                                Being specific is essential - use task numbers, concrete dates, and quantifiable impacts rather than general statements.
                                
                                Include a visualization section showing how tasks shift on a timeline (use plain text/markdown to simulate a Gantt chart).
                                """
                                
                                # Optimize prompts if needed
                                optimized_system_prompt = optimize_prompt(system_prompt)
                                
                                # OpenAI API call
                                api_response = client.chat.completions.create(
                                    model="gpt-4o", # the newest OpenAI model (released May 13, 2024)
                                    messages=[
                                        {"role": "system", "content": optimized_system_prompt},
                                        {"role": "user", "content": f"Simulate the impact of changing the project deadline from {latest_due_date.strftime('%Y-%m-%d') if isinstance(latest_due_date, datetime) else 'current date'} to {new_deadline.strftime('%Y-%m-%d')}. {' Recommend resource adjustments.' if adjust_resources else ''} {' Prioritize critical path tasks.' if prioritize_critical else ''} {' Handle dependent task rescheduling.' if reschedule_tasks else ''}"}
                                    ],
                                    temperature=0.7
                                )
                                
                                # Extract the content from the response
                                simulation_result = api_response.choices[0].message.content
                                
                                # Track token usage
                                token_manager.track_usage(api_response)
                            
                            st.markdown(simulation_result)
                            
                            # Save to simulation history
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            if 'simulation_history' not in st.session_state:
                                st.session_state['simulation_history'] = []
                            
                            st.session_state['simulation_history'].append({
                                "type": simulation_type,
                                "project": selected_project,
                                "deadline_change": f"Changed from {latest_due_date.strftime('%Y-%m-%d') if isinstance(latest_due_date, datetime) else 'unknown'} to {new_deadline.strftime('%Y-%m-%d')}",
                                "result": simulation_result,
                                "timestamp": timestamp
                            })
                            
                            # Add download button for the simulation results
                            st.download_button(
                                label="📥 Download Deadline Simulation Report",
                                data=f"# {simulation_type} Simulation\nProject: {selected_project}\nDeadline Change: {latest_due_date.strftime('%Y-%m-%d') if isinstance(latest_due_date, datetime) else 'Unknown'} to {new_deadline.strftime('%Y-%m-%d')}\n\n{simulation_result}",
                                file_name=f"Deadline_Simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                            
                        except Exception as e:
                            st.error(f"Error running simulation: {e}")
                            st.info("To use the simulation feature, ensure you have valid data and OpenAI API key configured.")
            
            elif simulation_type == "Team Composition Change":
                st.markdown("### Team Composition Change Simulation")
                st.markdown("Analyze the impact of adding or removing team members.")
                
                # Get current team members
                if skills_df is not None and 'Resource' in skills_df.columns:
                    team_members = sorted(skills_df['Resource'].unique().tolist())
                    team_roles = {}
                    if 'Skillset' in skills_df.columns:
                        for _, row in skills_df.iterrows():
                            team_roles[row['Resource']] = row['Skillset']
                else:
                    team_members = ["No team members found"]
                    team_roles = {}
                
                # Show current team composition
                st.markdown("#### Current Team Composition")
                team_data = []
                for member in team_members:
                    role = team_roles.get(member, "Unknown")
                    experience = "Senior" if "Proficiency" in skills_df.columns and skills_df[skills_df["Resource"] == member]["Proficiency"].iloc[0] >= 4 else "Junior"
                    team_data.append({"Resource": member, "Role": role, "Experience": experience})
                
                current_team_df = pd.DataFrame(team_data)
                st.dataframe(current_team_df, use_container_width=True)
                
                # Simulation options
                st.markdown("#### Simulation Options")
                change_type = st.radio("Change Type", ["Add team members", "Remove team members", "Replace team members"])
                
                if change_type == "Add team members":
                    num_members = st.slider("Number of new members", 1, 5, 1)
                    new_roles = []
                    
                    st.markdown("#### New Team Members")
                    for i in range(num_members):
                        col1, col2 = st.columns(2)
                        with col1:
                            role = st.selectbox(f"Role {i+1}", ["Developer", "QA", "Business Analyst", "UX Designer", "DevOps", "Project Manager"], key=f"role_{i}")
                        with col2:
                            experience = st.selectbox(f"Experience {i+1}", ["Junior", "Mid-level", "Senior"], key=f"exp_{i}")
                        new_roles.append({"Role": role, "Experience": experience})
                    
                elif change_type == "Remove team members":
                    members_to_remove = st.multiselect("Select members to remove", team_members)
                    new_roles = None
                    
                else:  # Replace team members
                    members_to_replace = st.multiselect("Select members to replace", team_members)
                    new_roles = []
                    
                    if members_to_replace:
                        st.markdown("#### Replacement Team Members")
                        for i, member in enumerate(members_to_replace):
                            st.markdown(f"Replacing: **{member}** ({team_roles.get(member, 'Unknown')})")
                            col1, col2 = st.columns(2)
                            with col1:
                                role = st.selectbox(f"New Role {i+1}", ["Developer", "QA", "Business Analyst", "UX Designer", "DevOps", "Project Manager"], key=f"replace_role_{i}")
                            with col2:
                                experience = st.selectbox(f"Experience {i+1}", ["Junior", "Mid-level", "Senior"], key=f"replace_exp_{i}")
                            new_roles.append({"Role": role, "Experience": experience, "Replacing": member})
                
                adjust_timeline = st.checkbox("Adjust project timeline based on team changes", value=True)
                
                if st.button("Run Team Composition Simulation"):
                    with st.spinner("Running simulation..."):
                        try:
                            st.markdown("### Simulation Results")
                            
                            if client is None:
                                # Generate placeholder simulation results if OpenAI client is not available
                                if change_type == "Add team members":
                                    change_description = f"Adding {num_members} new team members"
                                    impact_description = "Potential acceleration of project timeline"
                                elif change_type == "Remove team members":
                                    change_description = f"Removing {len(members_to_remove)} team members" if 'members_to_remove' in locals() and members_to_remove else "No team members selected for removal"
                                    impact_description = "Potential delays in project timeline"
                                else:  # Replace team members
                                    change_description = f"Replacing {len(members_to_replace)} team members" if 'members_to_replace' in locals() and members_to_replace else "No team members selected for replacement"
                                    impact_description = "Temporary disruption followed by potential stabilization"
                                
                                simulation_result = f"""## Team Composition Change Impact Assessment
                                
                                **Change Summary:**
                                - {change_description}
                                - Timeline adjustments: {'Enabled' if adjust_timeline else 'Disabled'}
                                
                                **Project Impact:**
                                - {impact_description}
                                - Onboarding period required for new team members
                                - Knowledge transfer sessions recommended
                                
                                **Resource Allocation Impact:**
                                - Workload redistribution needed across remaining team members
                                - Skill coverage analysis recommended
                                
                                **Recommendations:**
                                - Review task assignments and priorities
                                - Establish knowledge transfer plan
                                - Update capacity planning for upcoming sprints
                                
                                *Please note: To get actual AI-generated simulations, please configure your OpenAI API key in the Streamlit secrets.*
                                """
                            else:
                                # Prepare data for simulation
                                if change_type == "Add team members":
                                    change_details = f"Adding {num_members} new team members with roles: " + ", ".join([f"{r['Role']} ({r['Experience']})" for r in new_roles])
                                elif change_type == "Remove team members":
                                    if 'members_to_remove' in locals() and members_to_remove:
                                        change_details = f"Removing team members: " + ", ".join([f"{m} ({team_roles.get(m, 'Unknown')})" for m in members_to_remove])
                                    else:
                                        change_details = "No team members selected for removal"
                                else:  # Replace team members
                                    if 'members_to_replace' in locals() and members_to_replace and new_roles:
                                        changes = []
                                        for i, member in enumerate(members_to_replace):
                                            changes.append(f"{member} ({team_roles.get(member, 'Unknown')}) → {new_roles[i]['Role']} ({new_roles[i]['Experience']})")
                                        change_details = f"Replacing team members: " + ", ".join(changes)
                                    else:
                                        change_details = "No team members selected for replacement"
                                
                                # Create prompt for simulation
                                system_prompt = f"""You are an AI assistant that simulates the impact of team composition changes on project delivery.
                                
                                SIMULATION PARAMETERS:
                                - Change Type: {change_type}
                                - Current Team: {', '.join([f"{m} ({team_roles.get(m, 'Unknown')})" for m in team_members])}
                                - Change Details: {change_details}
                                - Adjust Timeline: {adjust_timeline}
                                
                                PROJECT CONTEXT:
                                - Total Tasks: {len(issues_df) if issues_df is not None else 'Unknown'}
                                - Open Tasks: {len(issues_df[issues_df['Status'] != 'Done']) if issues_df is not None and 'Status' in issues_df.columns else 'Unknown'}
                                - Project Timeline: Based on current due dates
                                
                                Generate a detailed impact assessment that includes:
                                1. Project Timeline Impact - How will these team changes affect delivery dates?
                                2. Skill Coverage Analysis - What skills will be gained or lost?
                                3. Knowledge Transfer Requirements - What knowledge transfer is needed?
                                4. Specific Recommendations - What actions should be taken to optimize team performance after these changes?
                                
                                Format your response with markdown for better readability. Be specific and practical in your recommendations.
                                """
                                
                                # Optimize prompts if needed
                                optimized_system_prompt = optimize_prompt(system_prompt)
                                
                                # OpenAI API call
                                api_response = client.chat.completions.create(
                                    model="gpt-4o", # the newest OpenAI model (released May 13, 2024)
                                    messages=[
                                        {"role": "system", "content": optimized_system_prompt},
                                        {"role": "user", "content": f"Simulate the impact of {change_details} on the project team and delivery timeline. {' Consider project timeline adjustments.' if adjust_timeline else ''}"}
                                    ],
                                    temperature=0.7
                                )
                                
                                # Extract the content from the response
                                simulation_result = api_response.choices[0].message.content
                                
                                # Track token usage
                                token_manager.track_usage(api_response)
                            
                            st.markdown(simulation_result)
                            
                            # Save to simulation history
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            if 'simulation_history' not in st.session_state:
                                st.session_state['simulation_history'] = []
                            
                            st.session_state['simulation_history'].append({
                                "type": simulation_type,
                                "change": change_type,
                                "details": change_details if 'change_details' in locals() else change_type,
                                "result": simulation_result,
                                "timestamp": timestamp
                            })
                            
                            # Add download button for the simulation results
                            st.download_button(
                                label="📥 Download Team Composition Simulation Report",
                                data=f"# {simulation_type} Simulation\nChange Type: {change_type}\nDetails: {change_details if 'change_details' in locals() else ''}\n\n{simulation_result}",
                                file_name=f"Team_Composition_Simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                            
                        except Exception as e:
                            st.error(f"Error running simulation: {e}")
                            st.info("To use the simulation feature, ensure you have valid data and OpenAI API key configured.")
            
            elif simulation_type == "Priority Shift":
                st.markdown("### Priority Shift Simulation")
                st.markdown("Analyze the impact of shifting priorities between different user stories and tasks.")
                
                # Initialize priority_changes variable at the start of this section
                priority_changes = []
                
                # Get available projects and tasks for prioritization
                if issues_df is not None:
                    if 'Project' in issues_df.columns:
                        project_keys = sorted(issues_df['Project'].unique().tolist())
                    else:
                        project_keys = ["No projects found"]
                        
                    if 'Priority' in issues_df.columns:
                        priorities = sorted(issues_df['Priority'].dropna().unique().tolist())
                    else:
                        priorities = ["Highest", "High", "Medium", "Low", "Lowest"]
                else:
                    project_keys = ["No projects found"]
                    priorities = ["Highest", "High", "Medium", "Low", "Lowest"]
                
                # First select a project to see its user stories
                st.markdown("#### Project Selection")
                selected_project = st.selectbox("Select Project", project_keys)
                
                if selected_project and selected_project != "No projects found":
                    # Get user stories for the selected project
                    if issues_df is not None and 'Project' in issues_df.columns and 'Issue Key' in issues_df.columns and 'Summary' in issues_df.columns:
                        project_stories = issues_df[issues_df['Project'] == selected_project]
                        
                        if not project_stories.empty:
                            st.markdown("#### User Story Prioritization")
                            st.markdown("Select user stories to change their priorities:")
                            
                            # Display current stories with their priorities
                            story_cols = ['Issue Key', 'Summary', 'Priority', 'Status']
                            available_cols = [col for col in story_cols if col in project_stories.columns]
                            
                            if available_cols:
                                stories_df = project_stories[available_cols].copy()
                                st.dataframe(stories_df, use_container_width=True)
                                
                                # Allow users to select multiple stories to reprioritize
                                if 'Issue Key' in project_stories.columns and 'Summary' in project_stories.columns:
                                    # Create selection options with both key and summary
                                    story_options = [f"{row['Issue Key']} - {row['Summary']}" for _, row in project_stories.iterrows()]
                                    selected_stories = st.multiselect("Select User Stories to Reprioritize", story_options)
                                    
                                    if selected_stories:
                                        st.markdown("### Priority Shifts")
                                        
                                        priority_changes = []
                                        for story in selected_stories:
                                            # Extract the issue key from the selection
                                            issue_key = story.split(' - ')[0]
                                            
                                            # Get current priority if available
                                            current_priority = "Medium"
                                            if 'Priority' in project_stories.columns:
                                                story_data = project_stories[project_stories['Issue Key'] == issue_key]
                                                if not story_data.empty and pd.notna(story_data['Priority'].iloc[0]):
                                                    current_priority = story_data['Priority'].iloc[0]
                                            
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.markdown(f"**{issue_key}:** {story.split(' - ', 1)[1] if ' - ' in story else ''}")
                                                st.text(f"Current Priority: {current_priority}")
                                            with col2:
                                                new_priority = st.selectbox(f"New Priority", priorities, key=f"new_{issue_key}")
                                            
                                            priority_changes.append({"Issue Key": issue_key, "Story": story.split(' - ', 1)[1] if ' - ' in story else '', "Current": current_priority, "New": new_priority})
                                        
                                        adjust_resources = st.checkbox("Adjust resource allocations based on new priorities", value=True)
                                        recalculate_timeline = st.checkbox("Recalculate delivery timeline based on new priorities", value=True)
                                    else:
                                        st.info("Please select at least one user story to continue.")
                                else:
                                    st.error("Required columns 'Issue Key' or 'Summary' not found in the data.")
                            else:
                                st.error("No story data columns available.")
                        else:
                            st.info(f"No user stories found for project '{selected_project}'.")
                    else:
                        st.error("Required columns not found in the data.")
                else:
                    st.info("Please select a valid project to continue.")
                    priority_changes = []
                
                if priority_changes and st.button("Run Priority Shift Simulation"):
                    with st.spinner("Running simulation..."):
                        try:
                            st.markdown("### Simulation Results")
                            
                            if client is None:
                                # Generate enhanced placeholder simulation results if OpenAI client is not available
                                priority_changes_summary = "\n".join([f"- {p['Project']}: {p['Current']} → {p['New']}" for p in priority_changes])
                                
                                # Extract specific task information for each affected project
                                affected_projects_tasks = ""
                                resource_impact = ""
                                critical_tasks = ""
                                
                                # Define shift_all_tasks variable with default value
                                shift_all_tasks = True
                                
                                # Get list of projects from the priority changes
                                selected_projects = []
                                for change in priority_changes:
                                    if 'Issue Key' in change:
                                        issue_key = change['Issue Key']
                                        # Find project for this issue key
                                        if issues_df is not None and 'Issue Key' in issues_df.columns and 'Project' in issues_df.columns:
                                            project_row = issues_df[issues_df['Issue Key'] == issue_key]
                                            if not project_row.empty and 'Project' in project_row.columns:
                                                project = project_row['Project'].iloc[0]
                                                if project not in selected_projects:
                                                    selected_projects.append(project)
                                
                                if issues_df is not None and 'Project' in issues_df.columns and selected_projects:
                                    for project in selected_projects:
                                        project_tasks = issues_df[issues_df['Project'] == project]
                                        task_count = len(project_tasks)
                                        
                                        if task_count > 0:
                                            # Generate table of affected tasks
                                            affected_projects_tasks += f"\n### {project} - {task_count} Tasks Affected\n\n"
                                            affected_projects_tasks += "| Task ID | Summary | Current Priority | New Priority | Status |\n| --- | --- | --- | --- | --- |\n"
                                            
                                            # Get current project priority change for this project
                                            issue_keys_in_project = project_tasks['Issue Key'].tolist()
                                            project_changes = [p for p in priority_changes if p.get('Issue Key') in issue_keys_in_project]
                                            
                                            # Use the first change as a representative for the project
                                            if project_changes:
                                                current_priority = project_changes[0].get('Current', 'Unknown')
                                                new_priority = project_changes[0].get('New', 'Unknown')
                                            else:
                                                current_priority = "Unknown"
                                                new_priority = "Unknown"
                                            
                                            # Add up to 5 tasks from this project
                                            for _, task in project_tasks.head(5).iterrows():
                                                task_id = task['Issue Key'] if 'Issue Key' in task else "Unknown"
                                                summary = task['Summary'] if 'Summary' in task else "Unknown"
                                                task_priority = task['Priority'] if 'Priority' in task else "Unknown"
                                                status = task['Status'] if 'Status' in task else "Unknown"
                                                
                                                # Only change individual task priorities if shift_all_tasks is True
                                                task_new_priority = new_priority if shift_all_tasks else task_priority
                                                
                                                affected_projects_tasks += f"| {task_id} | {summary} | {task_priority} | {task_new_priority} | {status} |\n"
                                            
                                            # Add resource impact for this project
                                            if adjust_resources:
                                                resource_impact += f"\n### Resource Impact for {project}\n"
                                                
                                                # Calculate simplified resource adjustment
                                                priority_increase = ['Highest', 'High', 'Medium', 'Low', 'Lowest'].index(current_priority) > ['Highest', 'High', 'Medium', 'Low', 'Lowest'].index(new_priority)
                                                
                                                if priority_increase:
                                                    resource_impact += f"- **Increased Resource Need**: {project} priority increased from {current_priority} to {new_priority}\n"
                                                    resource_impact += f"- Recommended additional allocation: +20% for the {task_count} affected tasks\n"
                                                else:
                                                    resource_impact += f"- **Decreased Resource Need**: {project} priority decreased from {current_priority} to {new_priority}\n"
                                                    resource_impact += f"- Resources can be reallocated to higher priority projects\n"
                                            
                                            # Add critical tasks for high priority projects
                                            if new_priority in ['Highest', 'High']:
                                                high_pri_tasks = project_tasks[(project_tasks['Status'] != 'Done') & (project_tasks['Priority'].isin(['Highest', 'High']))].head(3)
                                                
                                                if not high_pri_tasks.empty:
                                                    critical_tasks += f"\n### Critical Tasks for {project}\n\n"
                                                    critical_tasks += "| Task ID | Summary | Due Date | Assignee |\n| --- | --- | --- | --- |\n"
                                                    
                                                    for _, task in high_pri_tasks.iterrows():
                                                        task_id = task['Issue Key'] if 'Issue Key' in task else "Unknown"
                                                        summary = task['Summary'] if 'Summary' in task else "Unknown"
                                                        due_date = task['Due Date'].strftime('%Y-%m-%d') if 'Due Date' in task and pd.notna(task['Due Date']) else "Not set"
                                                        assignee = task['Assignee'] if 'Assignee' in task and pd.notna(task['Assignee']) else "Unassigned"
                                                        
                                                        critical_tasks += f"| {task_id} | {summary} | {due_date} | {assignee} |\n"
                                
                                # Timeline visualization
                                timeline_viz = ""
                                if recalculate_timeline:
                                    timeline_viz = "\n### Timeline Visualization\n\n```\n"
                                    
                                    for project in selected_projects:
                                        # Look for any issues in this project with priority changes
                                        project_tasks = issues_df[issues_df['Project'] == project] if issues_df is not None and 'Project' in issues_df.columns else pd.DataFrame()
                                        if not project_tasks.empty:
                                            issue_keys_in_project = project_tasks['Issue Key'].tolist() if 'Issue Key' in project_tasks.columns else []
                                            project_changes = [p for p in priority_changes if p.get('Issue Key') in issue_keys_in_project]
                                            
                                            if project_changes:
                                                # Use the first change as representative
                                                current_priority = project_changes[0].get('Current', 'Medium')
                                                new_priority = project_changes[0].get('New', 'Medium')
                                                
                                                # Simple visual representation of timeline impact
                                                try:
                                                    priority_increase = ['Highest', 'High', 'Medium', 'Low', 'Lowest'].index(current_priority) > ['Highest', 'High', 'Medium', 'Low', 'Lowest'].index(new_priority)
                                                    
                                                    if priority_increase:
                                                        timeline_viz += f"{project} ({current_priority} → {new_priority}): |-------(Current)-------|\n"
                                                        timeline_viz += f"                          |-----(New)-----|  Timeline reduced by ~20%\n\n"
                                                    else:
                                                        timeline_viz += f"{project} ({current_priority} → {new_priority}): |-------(Current)-------|\n"
                                                        timeline_viz += f"                          |----------(New)----------| Timeline extended by ~15%\n\n"
                                                except ValueError:
                                                    # Handle any unexpected priority values
                                                    timeline_viz += f"{project} ({current_priority} → {new_priority}): |-------(Current)-------|\n"
                                                    timeline_viz += f"                          |------(New)------| Timeline impact unknown\n\n"
                                    
                                    timeline_viz += "```\n"
                                
                                simulation_result = f"""## Priority Shift Impact Assessment
                                
                                **Priority Changes:**
                                {priority_changes_summary}
                                
                                **Affected Tasks:**
                                {affected_projects_tasks}
                                
                                **Impact on Resource Allocation:**
                                - {'Resource reallocation needed based on new priorities' if adjust_resources else 'No resource reallocation requested'}
                                - {'All tasks within projects will be reprioritized' if shift_all_tasks else 'Only project-level priorities will change'}
                                {resource_impact}
                                
                                **Timeline Impact:**
                                - {'Timeline recalculation based on new priorities' if recalculate_timeline else 'Timeline remains unchanged'}
                                - Higher priority projects will be delivered faster, potentially delaying lower priority projects
                                {timeline_viz}
                                
                                {critical_tasks}
                                
                                **Recommendations:**
                                1. {'Update all task priorities in the affected projects to match new project priorities' if shift_all_tasks else 'Maintain individual task priorities while updating project-level priority'}
                                2. {'Reassign resources to focus on newly prioritized work' if adjust_resources else 'Review resource allocations to ensure alignment with priorities'}
                                3. {'Revise sprint planning based on new priorities' if recalculate_timeline else 'Monitor delivery timeline impacts'}
                                4. Communicate priority changes to all stakeholders and teams
                                5. Review dependencies between tasks to ensure critical path is still valid
                                
                                *Note: To get AI-powered simulations with deeper insights, please configure your OpenAI API key in the Streamlit secrets.*
                                """
                            else:
                                # Prepare data for simulation
                                priority_changes_text = "\n".join([f"- {p['Project']}: {p['Current']} → {p['New']}" for p in priority_changes])
                                
                                # Count affected tasks
                                affected_tasks_count = 0
                                if issues_df is not None and 'Project' in issues_df.columns:
                                    affected_tasks_count = issues_df[issues_df['Project'].isin(selected_projects)].shape[0]
                                
                                # Create prompt for simulation
                                system_prompt = f"""You are an AI assistant that simulates the impact of priority shifts on project delivery.
                                
                                SIMULATION PARAMETERS:
                                - Priority Changes:
                                {priority_changes_text}
                                - Shift All Tasks: {shift_all_tasks}
                                - Adjust Resources: {adjust_resources}
                                - Recalculate Timeline: {recalculate_timeline}
                                
                                PROJECT CONTEXT:
                                - Affected Projects: {', '.join(selected_projects)}
                                - Affected Tasks: {affected_tasks_count}
                                
                                Generate a detailed impact assessment that includes:
                                1. Resource Impact - How will these priority changes affect resource allocations?
                                2. Timeline Impact - How will delivery dates be affected?
                                3. Cross-Project Effects - What impacts might occur on other projects?
                                4. Specific Recommendations - What actions should be taken to implement these priority changes effectively?
                                
                                Format your response with markdown for better readability. Be specific and practical in your recommendations.
                                """
                                    
                                # Optimize prompts if needed
                                optimized_system_prompt = optimize_prompt(system_prompt)
                                
                                # OpenAI API call
                                api_response = client.chat.completions.create(
                                    model="gpt-4o", # the newest OpenAI model (released May 13, 2024)
                                    messages=[
                                        {"role": "system", "content": optimized_system_prompt},
                                        {"role": "user", "content": f"Simulate the impact of the following priority changes: {priority_changes_text}. {' Apply changes to all tasks within these projects.' if shift_all_tasks else ''} {' Adjust resource allocations accordingly.' if adjust_resources else ''} {' Recalculate project timelines.' if recalculate_timeline else ''}"}
                                    ],
                                    temperature=0.7
                                )
                                
                                # Extract the content from the response
                                simulation_result = api_response.choices[0].message.content
                                
                                # Track token usage
                                token_manager.track_usage(api_response)
                                
                                st.markdown(simulation_result)
                                
                                # Save to simulation history
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                if 'simulation_history' not in st.session_state:
                                    st.session_state['simulation_history'] = []
                                
                                st.session_state['simulation_history'].append({
                                    "type": simulation_type,
                                    "projects": ", ".join(selected_projects),
                                    "changes": priority_changes_text,
                                    "result": simulation_result,
                                    "timestamp": timestamp
                                })
                                
                                # Add download button for the simulation results
                                st.download_button(
                                    label="📥 Download Priority Shift Simulation Report",
                                    data=f"# {simulation_type} Simulation\nProjects: {', '.join(selected_projects)}\nPriority Changes:\n{priority_changes_text}\n\n{simulation_result}",
                                    file_name=f"Priority_Shift_Simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    mime="text/plain"
                                )
                        except Exception as e:
                            st.error(f"Error running simulation: {e}")
                            st.info("To use the simulation feature, ensure you have valid data and OpenAI API key configured.")
            
            elif simulation_type == "Budget Change":
                st.markdown("### Budget Change Simulation")
                st.markdown("Analyze the impact of budget changes on project staffing and timelines.")
                
                # Project selection for budget simulation
                if issues_df is not None and 'Project' in issues_df.columns:
                    project_keys = sorted(issues_df['Project'].unique().tolist())
                else:
                    project_keys = ["No projects found"]
                
                selected_project = st.selectbox("Select Project for Budget Simulation", project_keys)
                
                # Budget information
                st.markdown("#### Budget Information")
                col1, col2 = st.columns(2)
                with col1:
                    current_budget = st.number_input("Current Budget ($)", min_value=10000, value=100000, step=10000)
                with col2:
                    new_budget = st.number_input("New Budget ($)", min_value=10000, value=80000, step=10000)
                
                budget_change = new_budget - current_budget
                budget_change_percent = (budget_change / current_budget) * 100 if current_budget > 0 else 0
                
                if budget_change < 0:
                    st.warning(f"Budget reduction of ${abs(budget_change):,} ({abs(budget_change_percent):.1f}%)")
                else:
                    st.success(f"Budget increase of ${budget_change:,} ({budget_change_percent:.1f}%)")
                
                # Simulation options
                st.markdown("#### Simulation Options")
                adjust_resources = st.checkbox("Adjust team size based on budget", value=True)
                adjust_scope = st.checkbox("Adjust project scope based on budget", value=True)
                adjust_timeline = st.checkbox("Adjust timeline based on budget", value=True)
                
                if st.button("Run Budget Change Simulation"):
                    with st.spinner("Running simulation..."):
                        try:
                            st.markdown("### Simulation Results")
                            
                            if client is None:
                                # Generate enhanced placeholder simulation results if OpenAI client is not available
                                if budget_change < 0:
                                    resource_impact = "Reduction in team size may be necessary"
                                    scope_impact = "Some features may need to be descoped"
                                    timeline_impact = "Timeline extension may be required"
                                else:
                                    resource_impact = "Additional resources can be allocated"
                                    scope_impact = "Scope can be expanded to include more features"
                                    timeline_impact = "Timeline could potentially be accelerated"
                                
                                # Extract specific task information for project-specific recommendations
                                affected_tasks_details = ""
                                critical_tasks_list = ""
                                reschedule_recommendations = ""
                                resource_allocation_details = ""
                                
                                # Estimated resource change based on budget
                                team_size_change = round(budget_change_percent/100 * 5)
                                timeline_adjustment_weeks = abs(round(budget_change_percent/100 * 4))
                                
                                if issues_df is not None and 'Project' in issues_df.columns:
                                    project_tasks = issues_df[issues_df['Project'] == selected_project]
                                    task_count = len(project_tasks)
                                    open_tasks = len(project_tasks[project_tasks['Status'] != 'Done']) if 'Status' in project_tasks.columns else 0
                                    
                                    if not project_tasks.empty:
                                        # Get highest priority tasks for critical tasks list
                                        if 'Priority' in project_tasks.columns:
                                            high_priority = project_tasks[project_tasks['Priority'].isin(['Highest', 'High'])]
                                            if not high_priority.empty:
                                                critical_tasks_list = "\n### Critical Tasks to Protect\n\n"
                                                critical_tasks_list += "| Task ID | Summary | Status | Priority | Due Date |\n| --- | --- | --- | --- | --- |\n"
                                                
                                                for _, task in high_priority.head(5).iterrows():
                                                    task_id = task['Issue Key'] if 'Issue Key' in task else "Unknown"
                                                    summary = task['Summary'] if 'Summary' in task else "Unknown"
                                                    status = task['Status'] if 'Status' in task else "Unknown"
                                                    priority = task['Priority'] if 'Priority' in task else "Unknown"
                                                    due_date = task['Due Date'].strftime('%Y-%m-%d') if 'Due Date' in task and pd.notna(task['Due Date']) else "Not set"
                                                    
                                                    critical_tasks_list += f"| {task_id} | {summary} | {status} | {priority} | {due_date} |\n"
                                        
                                        # Generate table of tasks affected by budget change
                                        if budget_change < 0 and adjust_scope:
                                            # For budget reduction, identify lowest priority tasks that might be descoped
                                            if 'Priority' in project_tasks.columns:
                                                low_priority = project_tasks[project_tasks['Priority'].isin(['Low', 'Lowest']) & (project_tasks['Status'] != 'Done')]
                                                if not low_priority.empty:
                                                    affected_tasks_details = "\n### Tasks to Consider for Descoping\n\n"
                                                    affected_tasks_details += "| Task ID | Summary | Priority | Status | Effort |\n| --- | --- | --- | --- | --- |\n"
                                                    
                                                    for _, task in low_priority.head(5).iterrows():
                                                        task_id = task['Issue Key'] if 'Issue Key' in task else "Unknown"
                                                        summary = task['Summary'] if 'Summary' in task else "Unknown"
                                                        priority = task['Priority'] if 'Priority' in task else "Unknown"
                                                        status = task['Status'] if 'Status' in task else "Unknown"
                                                        effort = task['Original Estimate (days)'] if 'Original Estimate (days)' in task and pd.notna(task['Original Estimate (days)']) else "Unknown"
                                                        
                                                        affected_tasks_details += f"| {task_id} | {summary} | {priority} | {status} | {effort} |\n"
                                        
                                        # Create rescheduling recommendations based on budget impact
                                        if adjust_timeline and open_tasks > 0:
                                            # For budget reduction, create timeline extension visualization
                                            if budget_change < 0:
                                                reschedule_recommendations = "\n### Task Rescheduling Recommendations\n\n"
                                                
                                                # Find tasks that need to be rescheduled
                                                if 'Due Date' in project_tasks.columns:
                                                    upcoming_tasks = project_tasks[(project_tasks['Status'] != 'Done') & pd.notna(project_tasks['Due Date'])].sort_values('Due Date')
                                                    
                                                    if not upcoming_tasks.empty:
                                                        reschedule_recommendations += "| Task ID | Summary | Current Due Date | Recommended New Due Date |\n| --- | --- | --- | --- |\n"
                                                        
                                                        # Add extension factor based on budget reduction percentage
                                                        extension_days = round(abs(budget_change_percent) / 5)
                                                        
                                                        for _, task in upcoming_tasks.head(5).iterrows():
                                                            task_id = task['Issue Key'] if 'Issue Key' in task else "Unknown"
                                                            summary = task['Summary'] if 'Summary' in task else "Unknown"
                                                            due_date = task['Due Date']
                                                            priority = task['Priority'] if 'Priority' in task else "Medium"
                                                            
                                                            # High priority tasks get less extension
                                                            if priority in ['Highest', 'High']:
                                                                task_extension = max(1, round(extension_days / 2))
                                                            else:
                                                                task_extension = extension_days
                                                                
                                                            new_date = (due_date + pd.Timedelta(days=task_extension)).strftime('%Y-%m-%d') if isinstance(due_date, pd.Timestamp) else "Recalculation needed"
                                                            reschedule_recommendations += f"| {task_id} | {summary} | {due_date.strftime('%Y-%m-%d') if isinstance(due_date, pd.Timestamp) else due_date} | {new_date} |\n"
                                                
                                                # Add timeline visualization
                                                reschedule_recommendations += "\n### Timeline Visualization\n\n```\n"
                                                reschedule_recommendations += f"Current Timeline: |-----(Tasks)-----| End\n"
                                                reschedule_recommendations += f"New Timeline:     |--------(Tasks with {timeline_adjustment_weeks} week extension)--------| End\n"
                                                reschedule_recommendations += "```\n"
                                                
                                        # Create resource allocation details
                                        if adjust_resources and 'Assignee' in project_tasks.columns:
                                            assignee_counts = project_tasks['Assignee'].value_counts()
                                            if not assignee_counts.empty:
                                                resource_allocation_details = "\n### Resource Allocation Impact\n\n"
                                                resource_allocation_details += "| Team Member | Current Task Count | Recommended Adjustment |\n| --- | --- | --- |\n"
                                                
                                                for assignee, count in assignee_counts.head(5).items():
                                                    if pd.notna(assignee):
                                                        # Calculate adjustment based on budget change
                                                        if budget_change < 0:
                                                            adjustment = f"Reduce by {max(1, round(abs(budget_change_percent)/20))} tasks"
                                                        else:
                                                            adjustment = f"Can take on {max(1, round(budget_change_percent/20))} more tasks"
                                                            
                                                        resource_allocation_details += f"| {assignee} | {count} | {adjustment} |\n"
                                
                                # Combine all project-specific details
                                simulation_result = f"""## Budget Change Impact Assessment
                                
                                **Budget Change:**
                                - Current Budget: ${current_budget:,}
                                - New Budget: ${new_budget:,}
                                - Change: ${budget_change:,} ({budget_change_percent:.1f}%)
                                
                                **Resource Impact:**
                                - {resource_impact if adjust_resources else 'No resource adjustments requested'}
                                - {'Estimated team size adjustment: ' + str(team_size_change) + ' team members' if adjust_resources else ''}
                                {resource_allocation_details}
                                
                                **Scope Impact:**
                                - {scope_impact if adjust_scope else 'No scope adjustments requested'}
                                {affected_tasks_details}
                                {critical_tasks_list}
                                
                                **Timeline Impact:**
                                - {timeline_impact if adjust_timeline else 'No timeline adjustments requested'}
                                - {'Estimated timeline adjustment: ' + str(timeline_adjustment_weeks) + ' weeks' if adjust_timeline else ''}
                                {reschedule_recommendations}
                                
                                **Actionable Recommendations:**
                                1. {'Review resource allocation across ' + str(open_tasks) + ' open tasks' if 'open_tasks' in locals() and open_tasks != 'Unknown' else 'Review resource allocation and team composition'}
                                2. {'Re-prioritize tasks based on budget constraints' if budget_change < 0 else 'Evaluate potential scope additions with increased budget'}
                                3. {'Adjust timeline expectations with stakeholders' if adjust_timeline else 'Maintain current timeline commitments'}
                                4. {'Consider reducing team size by ' + str(abs(team_size_change)) + ' members' if budget_change < 0 and adjust_resources else ('Consider adding ' + str(team_size_change) + ' team members' if budget_change > 0 and adjust_resources else 'Review team composition')}
                                5. {'Identify lowest priority features for potential descoping' if budget_change < 0 and adjust_scope else ('Identify high-value features to add to scope' if budget_change > 0 and adjust_scope else 'Review project scope and priorities')}
                                
                                *Note: To get AI-powered simulations with deeper insights, please configure your OpenAI API key in the Streamlit secrets.*
                                """
                            else:
                                # Prepare data for simulation
                                if 'team_members' in locals() and team_members:
                                    team_size = len(team_members)
                                else:
                                    team_size = "Unknown"
                                    
                                if issues_df is not None and 'Project' in issues_df.columns:
                                    project_tasks = issues_df[issues_df['Project'] == selected_project]
                                    task_count = len(project_tasks)
                                    open_tasks = len(project_tasks[project_tasks['Status'] != 'Done']) if 'Status' in project_tasks.columns else "Unknown"
                                else:
                                    task_count = "Unknown"
                                    open_tasks = "Unknown"
                                
                                # Create prompt for simulation
                                system_prompt = f"""You are an AI assistant that simulates the impact of budget changes on project delivery.
                                
                                SIMULATION PARAMETERS:
                                - Project: {selected_project}
                                - Current Budget: ${current_budget:,}
                                - New Budget: ${new_budget:,}
                                - Budget Change: ${budget_change:,} ({budget_change_percent:.1f}%)
                                - Adjust Resources: {adjust_resources}
                                - Adjust Scope: {adjust_scope}
                                - Adjust Timeline: {adjust_timeline}
                                
                                PROJECT CONTEXT:
                                - Team Size: {team_size}
                                - Total Tasks: {task_count}
                                - Open Tasks: {open_tasks}
                                
                                Generate a detailed impact assessment that includes:
                                1. Resource Impact - How will this budget change affect staffing levels?
                                2. Scope Impact - What features or deliverables might need adjustment?
                                3. Timeline Impact - How will delivery dates be affected?
                                4. Specific Recommendations - What actions should be taken to adapt to this budget change?
                                
                                Format your response with markdown for better readability. Be specific and practical in your recommendations.
                                """
                                
                                # Optimize prompts if needed
                                optimized_system_prompt = optimize_prompt(system_prompt)
                                
                                # OpenAI API call
                                api_response = client.chat.completions.create(
                                    model="gpt-4o", # the newest OpenAI model (released May 13, 2024)
                                    messages=[
                                        {"role": "system", "content": optimized_system_prompt},
                                        {"role": "user", "content": f"Simulate the impact of changing the project budget from ${current_budget:,} to ${new_budget:,}. {' Consider resource adjustments.' if adjust_resources else ''} {' Consider scope adjustments.' if adjust_scope else ''} {' Consider timeline adjustments.' if adjust_timeline else ''}"}
                                    ],
                                    temperature=0.7
                                )
                                
                                # Extract the content from the response
                                simulation_result = api_response.choices[0].message.content
                                
                                # Track token usage
                                token_manager.track_usage(api_response)
                            
                            st.markdown(simulation_result)
                            
                            # Save to simulation history
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            if 'simulation_history' not in st.session_state:
                                st.session_state['simulation_history'] = []
                            
                            st.session_state['simulation_history'].append({
                                "type": simulation_type,
                                "project": selected_project,
                                "budget_change": f"${current_budget:,} to ${new_budget:,} ({budget_change_percent:.1f}%)",
                                "result": simulation_result,
                                "timestamp": timestamp
                            })
                            
                            # Add download button for the simulation results
                            st.download_button(
                                label="📥 Download Budget Simulation Report",
                                data=f"# {simulation_type} Simulation\nProject: {selected_project}\nBudget Change: ${current_budget:,} to ${new_budget:,} ({budget_change_percent:.1f}%)\n\n{simulation_result}",
                                file_name=f"Budget_Simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                            
                        except Exception as e:
                            st.error(f"Error running simulation: {e}")
                            st.info("To use the simulation feature, ensure you have valid data and OpenAI API key configured.")

        # ---------- Buddy Brain Tab ----------
        with ai_tabs[3]:
            st.subheader("🧠 Buddy Brain")
            st.markdown("""
            Advanced AI features for project management including project health summary, task prioritization, and effort estimation.            
            """)
            
            # Create subtabs for different AI features
            ai_features_tabs = st.tabs(["Project Health Summary", "Task Prioritization", "Effort Estimation"])
            
            with ai_features_tabs[0]:
                # Project Health Summary
                try:
                    if issues_df is not None and worklogs_df is not None and skills_df is not None and leaves_df is not None:
                        generate_project_health_summary(issues_df, worklogs_df, skills_df, leaves_df)
                    else:
                        st.warning("Please load data first to generate project health summary.")
                except Exception as e:
                    st.error(f"Error generating project health summary: {e}")
                    st.info("To use the Project Health Summary feature, ensure you have valid data and OpenAI API key configured.")
            
            with ai_features_tabs[1]:
                # Task Prioritization
                try:
                    if issues_df is not None and worklogs_df is not None and skills_df is not None and leaves_df is not None:
                        ai_driven_task_prioritization(issues_df, worklogs_df, skills_df, leaves_df)
                    else:
                        st.warning("Please load data first to use task prioritization features.")
                except Exception as e:
                    st.error(f"Error with task prioritization: {e}")
                    st.info("To use the Task Prioritization feature, ensure you have valid data and OpenAI API key configured.")
            
            with ai_features_tabs[2]:
                # Effort Estimation
                try:
                    if issues_df is not None and worklogs_df is not None and skills_df is not None and leaves_df is not None:
                        effort_estimation_refinement(issues_df, worklogs_df, skills_df, leaves_df)
                    else:
                        st.warning("Please load data first to use effort estimation features.")
                except Exception as e:
                    st.error(f"Error with effort estimation: {e}")
                    st.info("To use the Effort Estimation feature, ensure you have valid data and OpenAI API key configured.")

        # ---------- Doc Dock Tab ----------
        with ai_tabs[4]:
            doc_dock_ui(client=client, token_manager=token_manager)

    except Exception as e:
        st.error(f"Error in AI PM Buddy: {e}")
        print(f"Detailed error in AI PM Buddy: {str(e)}")
        st.info("Please check the console for detailed error information.")

# ---------- MAIN NAVIGATION LOGIC ----------

# NOTE: Navigation starts here for all non-dashboard views
# The Dashboard section is already implemented earlier (line ~260)

# ---------- 2. RESOURCE MANAGEMENT SECTION ----------
if nav_selection == "🎯 Resource Management":
    # Header section
    st.title("🎯 Resource Management")
    
    if resource_view == "Team Workload":
        # Team Workload
        st.markdown("## Team Workload Analysis")
        st.markdown("Analyze and optimize resource allocation across the team.")
        
        # Standard filter section
        filters = standard_filter_section(section_id="resource_management")
        
        # Apply filters
        filtered_issues_df, filtered_worklogs_df, filtered_skills_df, filtered_leaves_df = apply_filters(filters)
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate overall team metrics
        if filtered_issues_df is not None and filtered_worklogs_df is not None:
            with col1:
                active_tasks = len(filtered_issues_df[filtered_issues_df['Status'] != 'Done'])
                st.metric("Active Tasks", active_tasks)
            
            with col2:
                if 'Time Spent (hrs)' in filtered_worklogs_df.columns:
                    total_hours = filtered_worklogs_df['Time Spent (hrs)'].sum()
                    st.metric("Total Hours Logged", f"{total_hours:.1f}")
            
            with col3:
                if 'Assignee' in filtered_issues_df.columns:
                    assignees = filtered_issues_df['Assignee'].nunique()
                    st.metric("Team Members", assignees)
            
            with col4:
                if 'Time Spent (hrs)' in filtered_worklogs_df.columns and 'Resource' in filtered_worklogs_df.columns:
                    avg_hours = filtered_worklogs_df.groupby('Resource')['Time Spent (hrs)'].sum().mean()
                    st.metric("Avg Hours per Resource", f"{avg_hours:.1f}")
        
        # Team workload analysis
        st.subheader("Current Team Workload")
        if filtered_issues_df is not None and 'Assignee' in filtered_issues_df.columns:
            # Get workload per assignee
            assignee_counts = filtered_issues_df['Assignee'].value_counts().reset_index()
            assignee_counts.columns = ['Assignee', 'Task Count']
            
            # Get active task counts
            active_counts = filtered_issues_df[filtered_issues_df['Status'] != 'Done']['Assignee'].value_counts().reset_index()
            active_counts.columns = ['Assignee', 'Active Tasks']
            
            # Merge with worklog data if available
            if filtered_worklogs_df is not None and 'Resource' in filtered_worklogs_df.columns and 'Time Spent (hrs)' in filtered_worklogs_df.columns:
                workloads = filtered_worklogs_df.groupby('Resource')['Time Spent (hrs)'].sum().reset_index()
                workloads.columns = ['Assignee', 'Hours Logged']
                
                # Merge datasets
                workload_analysis = pd.merge(assignee_counts, active_counts, on='Assignee', how='left')
                workload_analysis = pd.merge(workload_analysis, workloads, on='Assignee', how='left')
                workload_analysis = workload_analysis.fillna(0)
                
                # Calculate workload metrics
                team_avg_tasks = workload_analysis['Task Count'].mean()
                workload_analysis['Overloaded'] = workload_analysis['Active Tasks'] > team_avg_tasks * 1.2
                
                # Identify overloaded resources
                overloaded = workload_analysis[workload_analysis['Overloaded']]
                if not overloaded.empty:
                    st.warning(f"⚠️ {len(overloaded)} team members are overloaded compared to team average.")
                
                # Display workload table with visual indicators
                def highlight_overload(val):
                    if val == True:
                        return 'background-color: #ffcccc'
                    return ''
                
                st.dataframe(
                    workload_analysis.style.applymap(highlight_overload, subset=['Overloaded']),
                    use_container_width=True
                )
                
                # Workload distribution chart
                st.subheader("Workload Distribution")
                fig = px.bar(
                    workload_analysis, 
                    x='Assignee', 
                    y=['Active Tasks', 'Task Count'],
                    barmode='group',
                    color_discrete_sequence=['#FF9999', '#99CCFF'],
                    labels={'value': 'Tasks', 'variable': 'Type'},
                    title="Team Workload Distribution"
                )
                st.plotly_chart(fig, use_container_width=True, key="plotly_13508bf08369")
                
                # Add hours distribution chart
                st.subheader("Hours Logged Distribution")
                fig = px.bar(
                    workload_analysis.sort_values('Hours Logged', ascending=False),
                    x='Assignee',
                    y='Hours Logged',
                    color='Hours Logged',
                    color_continuous_scale='RdYlGn_r',  # Reversed to show high hours in red
                    title="Hours Logged by Team Member"
                )
                st.plotly_chart(fig, use_container_width=True, key="plotly_dafc2b59adaf")
                
                # Add information section with recommendations
                st.subheader("Workload Optimization Recommendations")
                if not overloaded.empty:
                    # Find underloaded resources
                    underloaded = workload_analysis[workload_analysis['Active Tasks'] < team_avg_tasks * 0.8]
                    
                    if not underloaded.empty:
                        st.info(f"💡 Consider redistributing tasks from overloaded resources ({', '.join(overloaded['Assignee'])}) to resources with capacity ({', '.join(underloaded['Assignee'])}).")
                        
                        # Add a button to view detailed redistribution recommendations
                        if st.button("View AI Task Redistribution Recommendations"):
                            st.session_state['view_redistribution'] = True
                    else:
                        st.info("💡 The team is unevenly loaded, but there are no significantly underloaded resources. Consider adjusting sprint commitments or bringing in additional resources.")
                else:
                    st.success("✅ The team has a balanced workload distribution.")
            else:
                st.warning("Worklog data not available or missing required columns.")
        else:
            st.warning("Issue data not available or missing required columns.")
    
    elif resource_view == "Skill Distribution":
        # Skill Distribution
        st.markdown("## Skill Distribution Analysis")
        st.markdown("Analyze team skills coverage and identify skill gaps.")
        
        # Standard filter section
        filters = standard_filter_section(section_id="planning")
        
        # Apply filters
        filtered_issues_df, filtered_worklogs_df, filtered_skills_df, filtered_leaves_df = apply_filters(filters)
        
        if filtered_skills_df is not None and not filtered_skills_df.empty:
            # Standardize columns
            if 'Resource' in filtered_skills_df.columns and 'Name' not in filtered_skills_df.columns:
                filtered_skills_df = filtered_skills_df.rename(columns={'Resource': 'Name'})
            
            if 'Name' in filtered_skills_df.columns and 'Skillset' in filtered_skills_df.columns:
                # Create tabs for different skill views
                tab1, tab2, tab3 = st.tabs(["Skill Overview", "Team Coverage", "Skill Gap Analysis"])
                
                with tab1:
                    # Skill overview
                    st.subheader("Team Skill Inventory")
                    
                    # Overall skill distribution
                    skill_counts = filtered_skills_df['Skillset'].value_counts().reset_index()
                    skill_counts.columns = ['Skill', 'Count']
                    
                    fig = px.pie(
                        skill_counts,
                        names='Skill',
                        values='Count',
                        title="Team Skill Distribution",
                        hole=0.4
                    )
                    chart_id = f"skill_pie_{st.session_state.get('chart_counter', 0)}"
                    st.session_state['chart_counter'] = st.session_state.get('chart_counter', 0) + 1
                    st.plotly_chart(fig, use_container_width=True, key=chart_id)
                    
                    # Resources per skill
                    st.subheader("Resources per Skill")
                    resource_skills = filtered_skills_df.groupby('Skillset')['Name'].nunique().reset_index()
                    resource_skills.columns = ['Skill', 'Resource Count']
                    
                    fig = px.bar(
                        resource_skills.sort_values('Resource Count'),
                        x='Skill',
                        y='Resource Count',
                        color='Skill',
                        title="Number of Resources per Skill"
                    )
                    chart_id = f"resource_skill_{st.session_state.get('chart_counter', 0)}"
                    st.session_state['chart_counter'] = st.session_state.get('chart_counter', 0) + 1
                    st.plotly_chart(fig, use_container_width=True, key=chart_id)
                    
                    # Skills inventory table
                    st.subheader("Skills Inventory")
                    skills_table = pd.crosstab(filtered_skills_df['Name'], filtered_skills_df['Skillset'])
                    
                    # Fill NAs with 0 and replace 1s with checkmarks for better visuals
                    skills_table = skills_table.fillna(0)
                    
                    # Display the table
                    st.dataframe(skills_table, use_container_width=True)
                
                with tab2:
                    # Team coverage by skill
                    st.subheader("Team Coverage Analysis")
                    
                    # Get unique skillsets
                    skillsets = filtered_skills_df['Skillset'].unique()
                    
                    for skill in skillsets:
                        st.subheader(f"{skill} Team Coverage")
                        skill_resources = filtered_skills_df[filtered_skills_df['Skillset'] == skill]['Name'].unique()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Resources with {skill} skills:** {len(skill_resources)}")
                            for resource in skill_resources:
                                st.markdown(f"- {resource}")
                        
                        with col2:
                            # If proficiency data is available, show it
                            if 'Proficiency' in filtered_skills_df.columns:
                                prof_data = filtered_skills_df[filtered_skills_df['Skillset'] == skill][['Name', 'Proficiency']]
                                fig = px.bar(
                                    prof_data,
                                    x='Name',
                                    y='Proficiency',
                                    title=f"{skill} Proficiency Levels",
                                    color='Proficiency',
                                    color_continuous_scale='Viridis'
                                )
                                chart_id = f"skill_prof_{skill}_{st.session_state.get('chart_counter', 0)}"
                                st.session_state['chart_counter'] = st.session_state.get('chart_counter', 0) + 1
                                st.plotly_chart(fig, use_container_width=True, key=chart_id)
                            else:
                                st.info(f"No proficiency data available for {skill} skills.")
                
                with tab3:
                    # Skill gap analysis
                    st.subheader("Skill Gap Analysis")
                    
                    # Calculate total resources
                    total_resources = filtered_skills_df['Name'].nunique()
                    
                    # Calculate coverage percentages
                    skill_coverage = resource_skills.copy()
                    skill_coverage['Coverage %'] = (skill_coverage['Resource Count'] / total_resources * 100).round(1)
                    skill_coverage = skill_coverage.sort_values('Coverage %')
                    
                    # Identify low coverage skills (below 30%)
                    low_coverage = skill_coverage[skill_coverage['Coverage %'] < 30]
                    
                    if not low_coverage.empty:
                        st.warning(f"⚠️ The following skills have low team coverage (below 30%): {', '.join(low_coverage['Skill'])}")
                    
                    # Create coverage chart
                    fig = px.bar(
                        skill_coverage,
                        x='Skill',
                        y='Coverage %',
                        color='Coverage %',
                        color_continuous_scale='RdYlGn',  # Red for low coverage, green for high
                        title="Skill Coverage Percentage"
                    )
                    # Add a reference line at 30%
                    fig.add_shape(
                        type="line",
                        x0=-0.5,
                        x1=len(skill_coverage)-0.5,
                        y0=30,
                        y1=30,
                        line=dict(color="red", width=2, dash="dash"),
                    )
                    fig.add_annotation(
                        x=0,
                        y=30,
                        text="Critical Coverage Threshold (30%)",
                        showarrow=False,
                        yshift=10,
                        xshift=100
                    )
                    chart_id = f"skill_coverage_{st.session_state.get('chart_counter', 0)}"
                    st.session_state['chart_counter'] = st.session_state.get('chart_counter', 0) + 1
                    st.plotly_chart(fig, use_container_width=True, key=chart_id)
                    
                    # Recommendations section
                    st.subheader("Skill Gap Recommendations")
                    if not low_coverage.empty:
                        st.info("""
                        💡 **Recommendations to address skill gaps:**
                        - Consider upskilling team members in low-coverage skill areas
                        - Prioritize hiring for critical skill gaps
                        - Develop cross-training programs for skills with single-resource dependencies
                        - Consider strategic partnerships or contractors for specialized skills
                        """)
                        
                        # Add a button to generate detailed recommendations
                        if st.button("Generate Detailed Skill Gap Analysis"):
                            # This could be connected to an AI-powered analysis
                            st.success("Detailed skill gap analysis would be generated here using AI capabilities.")
                    else:
                        st.success("✅ The team has good coverage across all skill areas.")
            else:
                st.warning("Skills data missing required columns.")
        else:
            st.warning("No skills data available with current filters.")
    
    elif resource_view == "Task Redistribution (AI)":
        # Call the AI Task Redistribution module
        ai_based_task_redistribution(issues_df, skills_df, worklogs_df, leaves_df)


# ---------- 3. PLANNING & SCHEDULING SECTION ----------
elif nav_selection == "📆 Planning & Scheduling":
    # Header section
    st.title("📆 Planning & Scheduling")
    
    if planning_view == "Sprint Planning":
        # Use the sprint planning assistant module
        sprint_planning_assistant(issues_df, skills_df, worklogs_df, leaves_df)
    
    elif planning_view == "Leave Impact Analysis":
        # Use the leave conflict detection module
        detect_leave_conflicts(issues_df, skills_df, worklogs_df, leaves_df)
    
    elif planning_view == "Timeline Forecasting":
        st.markdown("## Timeline Forecasting")
        st.markdown("Forecast project timelines based on historical data and team capacity.")
        
        # Standard filter section
        filters = standard_filter_section(section_id="timeline_forecasting")
        
        # Apply filters
        filtered_issues_df, filtered_worklogs_df, filtered_skills_df, filtered_leaves_df = apply_filters(filters)
        
        # Create tabs for different forecasting views
        tab1, tab2 = st.tabs(["Project Timeline", "Velocity Forecasting"])
        
        with tab1:
            # Project timeline forecasting
            st.subheader("Project Timeline Forecast")
            
            if filtered_issues_df is not None and not filtered_issues_df.empty:
                # Prepare the data for visualization
                # Convert date columns to datetime
                filtered_issues_df['Start Date'] = pd.to_datetime(filtered_issues_df['Start Date'], errors='coerce')
                filtered_issues_df['Due Date'] = pd.to_datetime(filtered_issues_df['Due Date'], errors='coerce')
                
                # Get valid data
                timeline_data = filtered_issues_df.dropna(subset=['Start Date', 'Due Date'])
                
                if not timeline_data.empty:
                    # Group tasks by milestone or project
                    if 'Project' in timeline_data.columns:
                        # Calculate earliest start and latest due date for each project
                        project_timeline = timeline_data.groupby('Project').agg(
                            Earliest_Start=('Start Date', 'min'),
                            Latest_Due=('Due Date', 'max'),
                            Task_Count=('Issue Key', 'count')
                        ).reset_index()
                        
                        # Create Gantt chart for project timeline
                        fig = px.timeline(
                            project_timeline,
                            x_start="Earliest_Start",
                            x_end="Latest_Due",
                            y="Project",
                            color="Project",
                            hover_data=["Task_Count"],
                            title="Project Timeline Overview"
                        )
                        fig.update_yaxes(autorange="reversed")
                        st.plotly_chart(fig, use_container_width=True, key="plotly_bd7119b9ed22")
                        
                        # Add milestone markers
                        st.subheader("Critical Milestones")
                        
                        # Create a detailed Gantt chart with task dependencies if available
                        st.subheader("Detailed Timeline with Dependencies")
                        
                        # Check if we have dependency data
                        deps_file = "enriched_jira_data_with_simulated.xlsx"
                        task_dependencies_df = None
                        
                        try:
                            if os.path.exists(deps_file):
                                task_dependencies_df = pd.read_excel(deps_file, sheet_name="Task Dependencies")
                        except Exception:
                            st.info("No dependency data available for visualizing task dependencies.")
                        
                        if task_dependencies_df is not None and not task_dependencies_df.empty:
                            # Merge dependency data with issue data
                            dependency_vis = timeline_data[['Issue Key', 'Summary', 'Assignee', 'Start Date', 'Due Date', 'Status']].copy()
                            
                            # Create a visualization that shows dependencies
                            # For now, we'll just show the raw dependency data
                            st.dataframe(task_dependencies_df)
                            
                            st.info("A detailed dependency graph would be shown here in a production implementation.")
                        else:
                            # Just show the regular Gantt chart
                            gantt_data = timeline_data.copy()
                            # Add task duration calculation
                            gantt_data['Duration (days)'] = (gantt_data['Due Date'] - gantt_data['Start Date']).dt.days + 1
                            
                            fig = px.timeline(
                                gantt_data,
                                x_start="Start Date",
                                x_end="Due Date",
                                y="Assignee",
                                color="Project", 
                                hover_name="Summary",
                                hover_data=["Issue Key", "Status", "Duration (days)"],
                                title="Detailed Task Timeline"
                            )
                            fig.update_yaxes(autorange="reversed")
                            st.plotly_chart(fig, use_container_width=True, key="plotly_2d307f299d7e")
                    else:
                        st.warning("No Project field available for timeline grouping.")
                else:
                    st.warning("No tasks with valid start and due dates available with current filters.")
            else:
                st.warning("No issue data available with current filters.")
            
            # Add ability to adjust timeline based on capacity, etc.
            st.subheader("Timeline Adjustment Tools")
            st.info("Timeline adjustment capabilities would be implemented here in a production version.")
        
        with tab2:
            # Velocity forecasting
            st.subheader("Velocity Forecasting")
            
            if filtered_issues_df is not None and not filtered_issues_df.empty:
                # Check if we have historical velocity data
                velocity_history_df = None
                try:
                    velocity_file = "enriched_jira_data_with_simulated.xlsx"
                    if os.path.exists(velocity_file):
                        velocity_history_df = pd.read_excel(velocity_file, sheet_name="Velocity History")
                except Exception:
                    st.info("No historical velocity data available for forecasting.")
                
                if velocity_history_df is not None and not velocity_history_df.empty:
                    # Display historical velocity
                    st.subheader("Historical Velocity Trends")
                    
                    # Plot the velocity data
                    # First check if the required columns exist
                    if 'Sprint' in velocity_history_df.columns and 'Velocity' in velocity_history_df.columns:
                        # Check if Team column exists
                        if 'Team' in velocity_history_df.columns:
                            # Use our safe line chart method instead of px.line directly
                            fig = safe_line_chart(
                                velocity_history_df,
                                x="Sprint",
                                y="Velocity",
                                color="Team",
                                markers=True,
                                title="Historical Velocity by Team"
                            )
                        else:
                            # If no Team column, create a simpler chart without color grouping
                            fig = safe_line_chart(
                                velocity_history_df,
                                x="Sprint",
                                y="Velocity",
                                markers=True,
                                title="Historical Velocity Trend"
                            )
                    else:
                        # Create an empty figure with a message if data isn't in the expected format
                        fig = go.Figure()
                        fig.add_annotation(
                            text="Velocity data is missing required columns",
                            showarrow=False,
                            font=dict(size=14)
                        )
                    st.plotly_chart(fig, use_container_width=True, key="plotly_3d8acca1346e")
                    
                    # Team member efficiency
                    st.subheader("Team Member Efficiency")
                    if 'Team Member' in velocity_history_df.columns and 'Completion Rate' in velocity_history_df.columns:
                        efficiency_data = velocity_history_df.groupby('Team Member')['Completion Rate'].mean().reset_index()
                        efficiency_data = efficiency_data.sort_values('Completion Rate', ascending=False)
                        
                        fig = px.bar(
                            efficiency_data,
                            x='Team Member',
                            y='Completion Rate',
                            color='Completion Rate',
                            title="Average Completion Rate by Team Member",
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig, use_container_width=True, key="plotly_3a3eaf18ba9c")
                    
                    # Velocity forecast
                    st.subheader("Velocity Forecast")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        forecast_sprints = st.slider("Number of Sprints to Forecast", 1, 6, 3)
                    
                    with col2:
                        confidence_level = st.selectbox("Confidence Level", ["Low (Optimistic)", "Medium (Realistic)", "High (Conservative)"])
                    
                    # Simple forecast based on historical data
                    if 'Velocity' in velocity_history_df.columns and 'Sprint' in velocity_history_df.columns:
                        # Calculate basic forecast
                        avg_velocity = velocity_history_df['Velocity'].mean()
                        std_dev = velocity_history_df['Velocity'].std()
                        
                        # Adjust based on confidence level
                        if confidence_level == "Low (Optimistic)":
                            forecast_velocity = avg_velocity + 0.5 * std_dev
                        elif confidence_level == "Medium (Realistic)":
                            forecast_velocity = avg_velocity
                        else:  # High (Conservative)
                            forecast_velocity = avg_velocity - 0.5 * std_dev
                        
                        # Create forecast data
                        last_sprint = velocity_history_df['Sprint'].max()
                        last_sprint_num = int(last_sprint.split()[-1]) if 'Sprint' in last_sprint else 0
                        
                        forecast_data = pd.DataFrame({
                            'Sprint': [f"Sprint {last_sprint_num + i + 1}" for i in range(forecast_sprints)],
                            'Forecasted Velocity': [round(forecast_velocity, 1)] * forecast_sprints
                        })
                        
                        # Display the forecast
                        st.subheader("Velocity Forecast for Upcoming Sprints")
                        st.dataframe(forecast_data, use_container_width=True)
                        
                        # Create a visualization with historical + forecasted data
                        # Prepare combined dataset
                        historical_data = velocity_history_df[['Sprint', 'Velocity']].rename(columns={'Velocity': 'Historical Velocity'})
                        historical_data['Type'] = 'Historical'
                        
                        forecast_visualization = forecast_data.rename(columns={'Forecasted Velocity': 'Velocity'})
                        forecast_visualization['Type'] = 'Forecasted'
                        
                        # Plot the combined data
                        fig = go.Figure()
                        
                        # Add historical line
                        fig.add_trace(go.Scatter(
                            x=historical_data['Sprint'],
                            y=historical_data['Historical Velocity'],
                            mode='lines+markers',
                            name='Historical',
                            line=dict(color='blue')
                        ))
                        
                        # Add forecast line with confidence band
                        fig.add_trace(go.Scatter(
                            x=forecast_visualization['Sprint'],
                            y=forecast_visualization['Velocity'],
                            mode='lines+markers',
                            name='Forecast',
                            line=dict(color='green', dash='dash')
                        ))
                        
                        # Add confidence band if not low confidence
                        if confidence_level != "Low (Optimistic)":
                            fig.add_trace(go.Scatter(
                                x=forecast_visualization['Sprint'],
                                y=forecast_visualization['Velocity'] + std_dev,
                                mode='lines',
                                name='Upper Bound',
                                line=dict(width=0),
                                showlegend=False
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=forecast_visualization['Sprint'],
                                y=forecast_visualization['Velocity'] - std_dev,
                                mode='lines',
                                name='Lower Bound',
                                line=dict(width=0),
                                fill='tonexty',
                                fillcolor='rgba(0, 176, 246, 0.2)',
                                showlegend=False
                            ))
                        
                        fig.update_layout(
                            title="Velocity Forecast with Historical Data",
                            xaxis_title="Sprint",
                            yaxis_title="Velocity (Story Points)",
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key="plotly_9404febdc7eb")
                        
                        # Add some insights
                        st.subheader("Velocity Insights")
                        st.markdown(f"**Average Historical Velocity:** {avg_velocity:.1f} story points per sprint")
                        st.markdown(f"**Velocity Variability:** {std_dev:.1f} story points (standard deviation)")
                        
                        # Capacity planning based on forecast
                        st.subheader("Capacity Planning")
                        st.markdown(f"Based on the forecasted velocity of **{forecast_velocity:.1f}** story points per sprint:")
                        st.markdown(f"- The team can complete approximately **{forecast_velocity * forecast_sprints:.1f}** story points in the next {forecast_sprints} sprints")
                        
                        # Calculate remaining work
                        if 'Story Points' in filtered_issues_df.columns:
                            remaining_points = filtered_issues_df[filtered_issues_df['Status'] != 'Done']['Story Points'].sum()
                            sprints_needed = remaining_points / forecast_velocity if forecast_velocity > 0 else 0
                            
                            st.markdown(f"- The remaining **{remaining_points:.1f}** story points would take approximately **{sprints_needed:.1f}** sprints to complete")
                    else:
                        st.warning("Velocity data is missing required columns.")
                else:
                    # Create a simple forecast based on current data
                    st.info("No historical velocity data available. Creating forecast based on current sprint data.")
                    
                    if 'Story Points' in filtered_issues_df.columns and 'Status' in filtered_issues_df.columns:
                        # Use current sprint data to estimate velocity
                        total_points = filtered_issues_df['Story Points'].sum()
                        completed_points = filtered_issues_df[filtered_issues_df['Status'] == 'Done']['Story Points'].sum()
                        completion_rate = completed_points / total_points if total_points > 0 else 0
                        
                        st.markdown(f"**Current Sprint Completion:** {completed_points:.1f} out of {total_points:.1f} story points ({completion_rate:.1%})")
                        
                        # Simple forecast
                        st.subheader("Simple Velocity Forecast")
                        st.markdown("Without historical data, we can only provide a simple forecast based on current sprint performance.")
                        
                        # Allow user to estimate velocity
                        estimated_velocity = st.slider("Estimated Velocity (Story Points per Sprint)", 5, 50, int(total_points))
                        
                        # Calculate forecast
                        remaining_points = filtered_issues_df[filtered_issues_df['Status'] != 'Done']['Story Points'].sum()
                        sprints_needed = remaining_points / estimated_velocity if estimated_velocity > 0 else 0
                        
                        st.markdown(f"Based on an estimated velocity of **{estimated_velocity}** story points per sprint:")
                        st.markdown(f"- The remaining **{remaining_points:.1f}** story points would take approximately **{sprints_needed:.1f}** sprints to complete")
                    else:
                        st.warning("Issue data is missing required columns for velocity calculations.")
            else:
                st.warning("No issue data available with current filters.")


# ---------- 4. RISK MANAGEMENT SECTION ----------
elif nav_selection == "🚨 Risk Management":
    # Header section
    st.title("🚨 Risk Management")
    
    if risk_view == "Daily Brief":
        # PM Daily Brief
        st.markdown("## Project Manager Daily Brief")
        st.markdown("Summary of action items, alerts, and recommendations for project management.")
        
        # Use the existing PM Daily Brief function
        pm_daily_brief()
    
    elif risk_view == "Technical Debt":
        # Technical Debt Management
        if tech_debt_df is not None:
            technical_debt_risk_management(issues_df, skills_df, worklogs_df, leaves_df, tech_debt_df)
        else:
            technical_debt_risk_management(issues_df, skills_df, worklogs_df, leaves_df)
    
    elif risk_view == "Risk Assessment":
        # Risk Assessment
        st.markdown("## Project Risk Assessment")
        st.markdown("Comprehensive risk analysis and mitigation strategies.")
        
        # Standard filter section
        filters = standard_filter_section(section_id="risk_assessment")
        
        # Apply filters
        filtered_issues_df, filtered_worklogs_df, filtered_skills_df, filtered_leaves_df = apply_filters(filters)
        
        # Create tabs for different risk views
        tab1, tab2, tab3 = st.tabs(["Risk Overview", "Detailed Analysis", "Mitigation Planning"])
        
        with tab1:
            # Risk Overview
            st.subheader("Project Risk Summary")
            
            if filtered_issues_df is not None and not filtered_issues_df.empty:
                # Check if we have risk data
                if 'Risk Level' in filtered_issues_df.columns or 'Priority' in filtered_issues_df.columns:
                    # Use Priority as a proxy for risk if Risk Level is not available
                    risk_column = 'Risk Level' if 'Risk Level' in filtered_issues_df.columns else 'Priority'
                    
                    # Create metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Calculate current date for overdue risks
                    current_date = pd.Timestamp.now().normalize()
                    filtered_issues_df['Due Date'] = pd.to_datetime(filtered_issues_df['Due Date'], errors='coerce')
                    
                    # Calculate metrics
                    total_issues = len(filtered_issues_df)
                    
                    # Count high risks from both regular Priority/Risk Level and user-defined risks
                    std_high_risks = len(filtered_issues_df[filtered_issues_df[risk_column].isin(['High', 'Highest'])])
                    
                    # Add user-defined high risks if available
                    user_high_risks = 0
                    if 'Risk Issue Key' in filtered_issues_df.columns and 'Risk Level' in filtered_issues_df.columns:
                        user_high_risks = len(filtered_issues_df[
                            (filtered_issues_df['Risk Issue Key'].notna()) & 
                            (filtered_issues_df['Risk Level'].isin(['High', 'Highest', 'Critical']))
                        ])
                        # Debug information
                        st.sidebar.markdown(f"Found {user_high_risks} user-defined high/critical risks")
                    
                    # Combine both standard and user-defined high risks
                    high_risks = std_high_risks + user_high_risks
                    
                    overdue_issues = len(filtered_issues_df[
                        (filtered_issues_df['Due Date'] < current_date) & 
                        (filtered_issues_df['Status'] != 'Done')
                    ])
                    open_issues = len(filtered_issues_df[filtered_issues_df['Status'] != 'Done'])
                    
                    # Display metrics
                    with col1:
                        st.metric("Total Issues", total_issues)
                    with col2:
                        st.metric("Open Issues", open_issues, f"{open_issues/total_issues:.1%}" if total_issues > 0 else "0%")
                    with col3:
                        st.metric("High Risk Items", high_risks, f"{high_risks/total_issues:.1%}" if total_issues > 0 else "0%")
                    with col4:
                        st.metric("Overdue Items", overdue_issues)
                    
                    # Risk distribution chart
                    st.subheader("Risk Distribution")
                    
                    if risk_column == 'Risk Level':
                        # Direct risk level data
                        risk_counts = filtered_issues_df['Risk Level'].value_counts().reset_index()
                        risk_counts.columns = ['Risk Level', 'Count']
                        
                        # Define risk order
                        risk_order = {"Critical": 1, "High": 2, "Medium": 3, "Low": 4, "Negligible": 5}
                        risk_counts['Order'] = risk_counts['Risk Level'].map(risk_order)
                        risk_counts = risk_counts.sort_values('Order')
                    else:
                        # Use priority as proxy for risk
                        risk_counts = filtered_issues_df['Priority'].value_counts().reset_index()
                        risk_counts.columns = ['Risk Level', 'Count']
                        
                        # Define priority order
                        risk_order = {"Highest": 1, "High": 2, "Medium": 3, "Low": 4, "Lowest": 5}
                        risk_counts['Order'] = risk_counts['Risk Level'].map(risk_order)
                        risk_counts = risk_counts.sort_values('Order')
                    
                    # Create visualization
                    fig = px.bar(
                        risk_counts,
                        x='Risk Level',
                        y='Count',
                        color='Risk Level',
                        color_discrete_map={'Critical': 'darkred', 'High': 'red', 'Medium': 'orange', 'Low': 'green', 'Negligible': 'lightgreen',
                                           'Highest': 'darkred', 'Lowest': 'lightgreen'},
                        title="Risk Level Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="plotly_34985f4a9cdd")
                    
                    # Risk status chart
                    st.subheader("Risk Status")
                    if 'Status' in filtered_issues_df.columns:
                        # Count issues by status and risk level
                        status_risk = pd.crosstab(filtered_issues_df['Status'], filtered_issues_df[risk_column])
                        
                        # Visualize
                        fig = px.bar(
                            status_risk,
                            title="Risk Status Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True, key="plotly_3985e0eafcae")
                    
                    # Risk over time
                    st.subheader("Risk Trend")
                    if 'Start Date' in filtered_issues_df.columns and 'Status' in filtered_issues_df.columns:
                        filtered_issues_df['Start Date'] = pd.to_datetime(filtered_issues_df['Start Date'], errors='coerce')
                        filtered_issues_df['Month'] = filtered_issues_df['Start Date'].dt.strftime('%Y-%m')
                        
                        # Group issues by month and risk level
                        if not filtered_issues_df['Month'].isna().all():
                            monthly_risks = pd.crosstab(filtered_issues_df['Month'], filtered_issues_df[risk_column])
                            monthly_risks = monthly_risks.sort_index()  # Sort by month
                            
                            # Visualize trend
                            # Use our safe line chart function instead of px.line
                            fig = safe_line_chart(
                                monthly_risks,
                                x=monthly_risks.index,
                                y=monthly_risks.columns.tolist(),
                                title="Risk Trend Over Time"
                            )
                            st.plotly_chart(fig, use_container_width=True, key="plotly_a19e9a73bd9e")
                        else:
                            st.info("No date data available for trend visualization.")
                else:
                    st.warning("No risk level or priority data available in the dataset.")
            else:
                st.warning("No issue data available with current filters.")
        
        with tab2:
            # Detailed Risk Analysis
            st.subheader("Detailed Risk Analysis")
            
            if filtered_issues_df is not None and not filtered_issues_df.empty:
                # Risk by category/project
                if 'Project' in filtered_issues_df.columns and 'Priority' in filtered_issues_df.columns:
                    # Create a heatmap of risks by project
                    project_risk = pd.crosstab(filtered_issues_df['Project'], filtered_issues_df['Priority'])
                    
                    # Calculate risk score per project
                    # Convert priority to numeric score: Highest=5, High=4, Medium=3, Low=2, Lowest=1
                    priority_score = {'Highest': 5, 'High': 4, 'Medium': 3, 'Low': 2, 'Lowest': 1}
                    
                    risk_scores = filtered_issues_df.copy()
                    risk_scores['Risk Score'] = risk_scores['Priority'].map(priority_score)
                    project_scores = risk_scores.groupby('Project')['Risk Score'].mean().reset_index()
                    project_scores = project_scores.sort_values('Risk Score', ascending=False)
                    
                    # Show project risk scores
                    st.subheader("Project Risk Scores")
                    fig = px.bar(
                        project_scores,
                        x='Project',
                        y='Risk Score',
                        color='Risk Score',
                        color_continuous_scale='RdYlGn_r',  # Reversed to show high risks in red
                        title="Average Risk Score by Project"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="plotly_9e53f825ebb9")
                    
                    # Show the heatmap
                    st.subheader("Risk Distribution by Project")
                    fig = px.imshow(
                        project_risk,
                        labels=dict(x="Priority", y="Project", color="Count"),
                        title="Risk Heatmap by Project"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="plotly_c90b0ee0dca1")
                
                # Risk by assignee
                if 'Assignee' in filtered_issues_df.columns and 'Priority' in filtered_issues_df.columns:
                    st.subheader("Risk by Assignee")
                    assignee_risks = filtered_issues_df[filtered_issues_df['Status'] != 'Done'].copy()
                    
                    if not assignee_risks.empty:
                        assignee_risks['Risk Score'] = assignee_risks['Priority'].map(priority_score)
                        assignee_scores = assignee_risks.groupby('Assignee').agg(
                            Avg_Risk=('Risk Score', 'mean'),
                            Open_Items=('Issue Key', 'count'),
                            High_Risk_Items=('Risk Score', lambda x: sum(x >= 4))  # Count items with score >= 4 (High or Highest)
                        ).reset_index()
                        
                        # Sort by risk score
                        assignee_scores = assignee_scores.sort_values('Avg_Risk', ascending=False)
                        
                        # Create a scatter plot
                        fig = px.scatter(
                            assignee_scores,
                            x='Open_Items',
                            y='Avg_Risk',
                            size='High_Risk_Items',
                            color='Avg_Risk',
                            hover_name='Assignee',
                            color_continuous_scale='RdYlGn_r',  # Reversed to show high risks in red
                            title="Risk Exposure by Assignee"
                        )
                        fig.update_layout(
                            xaxis_title="Number of Open Items",
                            yaxis_title="Average Risk Score",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True, key="plotly_541fb20ee691")
                        
                        # Show the table of assignee risks
                        st.dataframe(assignee_scores, use_container_width=True)
                    else:
                        st.info("No open issues with risk data.")
                
                # Risk detail table
                st.subheader("High Risk Items")
                if 'Priority' in filtered_issues_df.columns:
                    # Get standard high risk items
                    std_high_risk_items = filtered_issues_df[filtered_issues_df['Priority'].isin(['Highest', 'High']) & 
                                                       (filtered_issues_df['Status'] != 'Done')]
                    
                    # Get user-defined high risk items if available
                    user_high_risk_items = pd.DataFrame()
                    if 'Risk Issue Key' in filtered_issues_df.columns and 'Risk Level' in filtered_issues_df.columns:
                        user_high_risk_items = filtered_issues_df[
                            (filtered_issues_df['Risk Issue Key'].notna()) & 
                            (filtered_issues_df['Risk Level'].isin(['High', 'Highest', 'Critical'])) &
                            (filtered_issues_df['Status'] != 'Done')
                        ]
                        # Debug information about user-defined risks
                        if not user_high_risk_items.empty:
                            st.success(f"Found {len(user_high_risk_items)} user-defined high/critical risks")
                    
                    # Combine both sets of high risk items
                    all_high_risk_items = pd.concat([std_high_risk_items, user_high_risk_items]).drop_duplicates()
                    
                    if not all_high_risk_items.empty:
                        # Check if URisk ID column exists
                        display_columns = ['Issue Key', 'Summary', 'Assignee', 'Priority', 'Status', 'Due Date']
                        
                        # Add the URisk ID column if it exists
                        if 'Risk Issue Key' in all_high_risk_items.columns:
                            # Create a combined display dataframe with URisk ID column
                            display_df = all_high_risk_items.copy()
                            display_df['URisk ID'] = display_df['Risk Issue Key']
                            display_columns = ['Issue Key', 'URisk ID', 'Summary', 'Assignee', 'Priority', 'Status', 'Due Date'] 
                            st.dataframe(display_df[display_columns], use_container_width=True)
                        else:
                            st.dataframe(all_high_risk_items[display_columns], use_container_width=True)
                    else:
                        st.success("No high-risk items found with current filters.")
                else:
                    st.warning("Priority data not available.")
            else:
                st.warning("No issue data available with current filters.")
        
        with tab3:
            # Risk Mitigation Planning
            st.subheader("Risk Mitigation Planning")
            
            if filtered_issues_df is not None and not filtered_issues_df.empty:
                # Risk prioritization
                st.subheader("Risk Prioritization Matrix")
                
                # Check if we have the necessary columns
                required_columns = ['Priority', 'Status', 'Due Date']
                if all(col in filtered_issues_df.columns for col in required_columns):
                    # Prepare data for the matrix
                    risk_matrix_data = filtered_issues_df[filtered_issues_df['Status'] != 'Done'].copy()
                    
                    if not risk_matrix_data.empty:
                        # Calculate days to due date
                        current_date = pd.Timestamp.now().normalize()
                        risk_matrix_data['Days to Due'] = (risk_matrix_data['Due Date'] - current_date).dt.days
                        
                        # Convert priority to numeric
                        priority_map = {'Highest': 5, 'High': 4, 'Medium': 3, 'Low': 2, 'Lowest': 1}
                        risk_matrix_data['Priority Score'] = risk_matrix_data['Priority'].map(priority_map)
                        
                        # Create a custom column for urgency based on days to due date
                        def calculate_urgency(days):
                            if pd.isna(days):
                                return 3  # Medium urgency if no due date
                            elif days < 0:
                                return 5  # Highest urgency if overdue
                            elif days < 7:
                                return 4  # High urgency if due within a week
                            elif days < 14:
                                return 3  # Medium urgency if due within two weeks
                            elif days < 30:
                                return 2  # Low urgency if due within a month
                            else:
                                return 1  # Lowest urgency if due after a month
                        
                        risk_matrix_data['Urgency'] = risk_matrix_data['Days to Due'].apply(calculate_urgency)
                        
                        # Create the scatter plot for the matrix
                        fig = px.scatter(
                            risk_matrix_data,
                            x='Urgency',
                            y='Priority Score',
                            color='Priority',
                            size='Urgency',
                            hover_name='Summary',
                            hover_data=['Issue Key', 'Days to Due'],
                            title="Risk Prioritization Matrix",
                            labels={'Urgency': 'Urgency (Time Criticality)', 'Priority Score': 'Impact (Priority Level)'},
                            color_discrete_map={'Highest': 'darkred', 'High': 'red', 'Medium': 'orange', 'Low': 'green', 'Lowest': 'lightgreen'}
                        )
                        
                        # Customize the layout
                        fig.update_layout(
                            xaxis=dict(tickmode='array', tickvals=[1, 2, 3, 4, 5], 
                                       ticktext=['Very Low', 'Low', 'Medium', 'High', 'Very High']),
                            yaxis=dict(tickmode='array', tickvals=[1, 2, 3, 4, 5], 
                                       ticktext=['Very Low', 'Low', 'Medium', 'High', 'Very High']),
                            xaxis_title="Urgency (Time Criticality)",
                            yaxis_title="Impact (Priority Level)",
                            height=500
                        )
                        
                        # Add quadrant divisions
                        fig.add_shape(type="line", x0=3, y0=1, x1=3, y1=5, line=dict(color="gray", width=1, dash="dash"))
                        fig.add_shape(type="line", x0=1, y0=3, x1=5, y1=3, line=dict(color="gray", width=1, dash="dash"))
                        
                        # Add quadrant labels
                        fig.add_annotation(x=2, y=4, text="Major Risks", showarrow=False, font=dict(size=12))
                        fig.add_annotation(x=4, y=4, text="Critical Risks", showarrow=False, font=dict(size=12))
                        fig.add_annotation(x=2, y=2, text="Minor Risks", showarrow=False, font=dict(size=12))
                        fig.add_annotation(x=4, y=2, text="Moderate Risks", showarrow=False, font=dict(size=12))
                        
                        st.plotly_chart(fig, use_container_width=True, key="plotly_d16ef4de1851")
                        
                        # Risk mitigation recommendations
                        st.subheader("Risk Mitigation Recommendations")
                        
                        # Identify critical risks
                        critical_risks = risk_matrix_data[(risk_matrix_data['Priority Score'] >= 4) & (risk_matrix_data['Urgency'] >= 4)]
                        critical_count = len(critical_risks)
                        
                        # Identify major risks
                        major_risks = risk_matrix_data[(risk_matrix_data['Priority Score'] >= 4) & (risk_matrix_data['Urgency'] < 4)]
                        major_count = len(major_risks)
                        
                        # Identify moderate risks
                        moderate_risks = risk_matrix_data[(risk_matrix_data['Priority Score'] < 4) & (risk_matrix_data['Urgency'] >= 4)]
                        moderate_count = len(moderate_risks)
                        
                        # Risk mitigation recommendations based on counts
                        st.markdown(f"**Risk Distribution:**")
                        st.markdown(f"- Critical Risks: {critical_count}")
                        st.markdown(f"- Major Risks: {major_count}")
                        st.markdown(f"- Moderate Risks: {moderate_count}")
                        st.markdown(f"- Minor Risks: {len(risk_matrix_data) - critical_count - major_count - moderate_count}")
                        
                        # Show critical risks if any
                        if critical_count > 0:
                            st.markdown("### Critical Risks Requiring Immediate Attention")
                            st.dataframe(critical_risks[['Issue Key', 'Summary', 'Assignee', 'Priority', 'Days to Due']], use_container_width=True)
                            
                            # Generate mitigation recommendations
                            st.markdown("**Recommended Actions for Critical Risks:**")
                            st.markdown("1. Immediately escalate to project leadership")
                            st.markdown("2. Schedule emergency response meetings")
                            st.markdown("3. Allocate additional resources to critical items")
                            st.markdown("4. Implement daily progress tracking")
                        
                        # Show major risks if any
                        if major_count > 0:
                            st.markdown("### Major Risks Requiring Planning")
                            st.dataframe(major_risks[['Issue Key', 'Summary', 'Assignee', 'Priority', 'Days to Due']], use_container_width=True)
                            
                            # Generate mitigation recommendations
                            st.markdown("**Recommended Actions for Major Risks:**")
                            st.markdown("1. Develop detailed mitigation plans")
                            st.markdown("2. Assign ownership and accountability")
                            st.markdown("3. Schedule weekly review meetings")
                            st.markdown("4. Prepare contingency resources")
                    else:
                        st.success("No open issues with risk data.")
                else:
                    st.warning("Required data (Priority, Status, Due Date) not available for risk matrix.")
                
                # Risk response planning
                st.subheader("Risk Response Planning")
                st.markdown("Create proactive response plans for identified risks.")
                
                # Allow selection of a risk to plan for
                if 'Issue Key' in filtered_issues_df.columns and 'Summary' in filtered_issues_df.columns:
                    risk_options = filtered_issues_df[filtered_issues_df['Status'] != 'Done'].copy()
                    
                    if not risk_options.empty:
                        risk_options['Selection'] = risk_options['Issue Key'] + " - " + risk_options['Summary']
                        selected_risk = st.selectbox("Select a risk to create a response plan", risk_options['Selection'])
                        
                        # Create form for risk response
                        with st.form("risk_response_form"):
                            st.markdown("### Risk Response Plan")
                            
                            # Risk response strategy
                            response_strategy = st.radio(
                                "Response Strategy",
                                ["Avoid", "Mitigate", "Transfer", "Accept"],
                                horizontal=True
                            )
                            
                            # Response details
                            response_plan = st.text_area("Response Plan Details", 
                                                      "Describe the specific actions to be taken to address this risk...")
                            
                            # Owner and timeline
                            col1, col2 = st.columns(2)
                            with col1:
                                response_owner = st.text_input("Response Owner")
                            with col2:
                                response_timeline = st.date_input("Target Completion Date")
                            
                            # Contingency plan
                            contingency_plan = st.text_area("Contingency Plan", 
                                                          "Describe the backup plan if the primary response fails...")
                            
                            # Submit button
                            submit_button = st.form_submit_button("Save Response Plan")
                            
                            if submit_button:
                                st.success("Risk response plan saved! (Note: This is a mockup - no actual data is saved)")
                    else:
                        st.info("No open issues available for risk planning.")
                else:
                    st.warning("Issue data missing required columns.")
            else:
                st.warning("No issue data available with current filters.")


# ---------- 5. STRATEG-AIZ SECTION ----------
elif nav_selection == "🤖 Strateg-AIz":
    # Call the AI PM Buddy Assistant
    ai_pm_buddy_assistant()


# Execute Quick Actions if selected
if qa_brief:
    # Set session state and redirect to PM Daily Brief
    st.session_state['nav_selection'] = "🚨 Risk Management"
    st.session_state['risk_view'] = "Daily Brief"
    st.rerun()

elif qa_balance:
    # Set session state and redirect to Team Workload
    st.session_state['nav_selection'] = "🎯 Resource Management"
    st.session_state['resource_view'] = "Team Workload"
    st.rerun()

elif qa_optimize:
    # Set session state and redirect to Task Redistribution
    st.session_state['nav_selection'] = "🎯 Resource Management"
    st.session_state['resource_view'] = "Task Redistribution (AI)"
    st.rerun()

elif qa_plan:
    # Set session state and redirect to Sprint Planning
    st.session_state['nav_selection'] = "📆 Planning & Scheduling"
    st.session_state['planning_view'] = "Sprint Planning"
    st.rerun()
