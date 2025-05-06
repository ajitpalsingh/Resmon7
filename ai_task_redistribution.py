# AI Task Redistribution Module
# This module provides AI-powered functionality for optimizing task assignments

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import re
import json
import openai
import os
from openai import OpenAI
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    NETWORK_VISUALIZATION_AVAILABLE = True
except ImportError:
    NETWORK_VISUALIZATION_AVAILABLE = False
    # Define fallback function for dependency visualization if networkx is not available
    def visualize_dependencies(G, critical_path=None):
        """Fallback function when networkx is not available"""
        return "Dependency visualization requires networkx and matplotlib libraries."
import io
import base64
from PIL import Image

def ai_based_task_redistribution(issues_df, skills_df, worklogs_df, leaves_df, client=None):
    """Main function for AI-based task redistribution"""
    
    st.markdown("### 🤖 AI-Powered Task Redistribution")
    st.markdown("""
    This feature analyzes your team's current workload, skills, and task assignments 
    to suggest optimal task redistribution when:  
    - Team members are overloaded  
    - Skills aren't properly matched to tasks  
    - Sprint deadlines are at risk  
    - Resource conflicts are detected  
    """)
    
    # Create columns for analysis options and results
    redis_col1, redis_col2 = st.columns([1, 2])
    
    # Load the enriched data if available
    task_dependencies_df = None
    velocity_history_df = None
    
    try:
        # Try to load the dependency data from enriched Excel file
        deps_file = "enriched_jira_data_with_simulated.xlsx"
        if os.path.exists(deps_file):
            task_dependencies_df = pd.read_excel(deps_file, sheet_name="Task Dependencies")
            velocity_history_df = pd.read_excel(deps_file, sheet_name="Velocity History")
    except Exception as e:
        st.warning(f"Could not load enriched data: {str(e)}")
    
    with redis_col1:
        st.subheader("Analysis Options")
        analysis_type = st.radio(
            "Redistribution Analysis Type", 
            ["Workload Balancing", "Skill Optimization", "Critical Path Acceleration", "Leave Impact Mitigation"]
        )
        
        # Common parameters
        consider_skills = st.checkbox("Consider Skills", True)
        consider_velocity = st.checkbox("Consider Historical Velocity", True)
        consider_priority = st.checkbox("Consider Task Priority", True)
        consider_dependencies = st.checkbox("Consider Task Dependencies", True)
        
        if st.button("Generate Redistribution Plan", key="redis_plan"):
            with st.spinner("AI is analyzing team workload and generating optimal task distribution..."):
                # Get current date for comparison
                current_date = pd.Timestamp(datetime.now())
                
                # Check if we have the necessary data
                if issues_df is not None and not issues_df.empty and worklogs_df is not None and not worklogs_df.empty:
                    # Get active issues (not Done) and their assignees
                    if 'Status' in issues_df.columns and 'Assignee' in issues_df.columns:
                        active_issues = issues_df[issues_df['Status'] != 'Done']
                        
                        # Get workload per assignee from worklogs
                        if 'Resource' in worklogs_df.columns and 'Time Spent (hrs)' in worklogs_df.columns:
                            workloads = worklogs_df.groupby('Resource')['Time Spent (hrs)'].sum().reset_index()
                            workloads.columns = ['Assignee', 'Current Workload (hrs)']
                            
                            # Join with active issues to see who has most work
                            assignee_counts = active_issues['Assignee'].value_counts().reset_index()
                            assignee_counts.columns = ['Assignee', 'Task Count']
                            
                            # Make sure we have merged datasets to make inferences from
                            workload_analysis = pd.merge(assignee_counts, workloads, on='Assignee', how='outer').fillna(0)
                            
                            # Add a metric based on tasks per workload
                            workload_analysis['Tasks per Hour'] = workload_analysis['Task Count'] / workload_analysis['Current Workload (hrs)'].apply(lambda x: max(x, 1))
                            
                            # Sort by most overloaded people (highest task count, task density)
                            workload_analysis = workload_analysis.sort_values(by=['Task Count', 'Tasks per Hour'], ascending=False)
                            
                            # Get skill information if available
                            if skills_df is not None and not skills_df.empty and consider_skills:
                                # Two possible column names for resource
                                resource_col = 'Resource' if 'Resource' in skills_df.columns else 'Name'
                                if resource_col in skills_df.columns and 'Skillset' in skills_df.columns:
                                    # Create map of people to skills
                                    skill_map = skills_df.groupby(resource_col)['Skillset'].apply(list).to_dict()
                                    
                                    # Add proficiency information if available
                                    proficiency_map = {}
                                    if 'Proficiency' in skills_df.columns:
                                        for _, row in skills_df.iterrows():
                                            resource = row[resource_col]
                                            skill = row['Skillset']
                                            proficiency = row['Proficiency']
                                            if resource not in proficiency_map:
                                                proficiency_map[resource] = {}
                                            proficiency_map[resource][skill] = proficiency
                            
                            # Calculate who is overloaded based on a threshold (could be adjusted)
                            avg_tasks = workload_analysis['Task Count'].mean()
                            workload_analysis['Overloaded'] = workload_analysis['Task Count'] > avg_tasks * 1.2  # 20% over average
                            
                            # Display workload analysis
                            with redis_col2:
                                st.subheader("Current Team Workload")
                                st.dataframe(workload_analysis, use_container_width=True)
                                
                                # Visualize workload
                                fig = px.bar(
                                    workload_analysis, 
                                    x='Assignee', 
                                    y='Task Count',
                                    color='Overloaded',
                                    color_discrete_map={True: 'red', False: 'green'},
                                    title="Team Workload Distribution"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Analyze task dependencies if requested and data is available
                                if consider_dependencies and task_dependencies_df is not None and not task_dependencies_df.empty:
                                    st.subheader("Task Dependency Analysis")
                                    dependency_graph, critical_path, dependency_counts = analyze_task_dependencies(
                                        active_issues, task_dependencies_df
                                    )
                                    
                                    if critical_path:
                                        st.markdown(f"**Critical Path Tasks:** {', '.join(critical_path)}")
                                        
                                        # Create and display dependency visualization
                                        dependency_img = visualize_dependencies(dependency_graph, critical_path)
                                        if dependency_img:
                                            st.image(dependency_img, caption="Task Dependency Network Diagram", use_container_width=True)
                                
                                # Generate recommendations
                                st.subheader("AI-Generated Redistribution Recommendations")
                                
                                # Get overloaded and underloaded assignees
                                overloaded = workload_analysis[workload_analysis['Overloaded']]['Assignee'].tolist()
                                underloaded = workload_analysis[~workload_analysis['Overloaded']]['Assignee'].tolist()
                                
                                # Generate recommendations based on analysis type
                                recommendations = []
                                dependency_graph = None
                                critical_path = None
                                
                                if analysis_type == "Leave Impact Mitigation" and leaves_df is not None and not leaves_df.empty:
                                    recommendations = generate_leave_impact_recommendations(
                                        active_issues, leaves_df, underloaded, workload_analysis, 
                                        skill_map if 'skill_map' in locals() else None
                                    )
                                else:  # Default to workload balancing for other analysis types
                                    if len(overloaded) > 0 and len(underloaded) > 0:
                                        # Default to standard workload recommendations
                                        dependencies_to_use = task_dependencies_df if consider_dependencies else None
                                        velocity_to_use = velocity_history_df if consider_velocity else None
                                        
                                        recommendations, dependency_graph, critical_path = generate_workload_recommendations(
                                            active_issues, overloaded, underloaded, workload_analysis, 
                                            skill_map if 'skill_map' in locals() else None,
                                            consider_priority=consider_priority,
                                            dependencies_df=dependencies_to_use,
                                            velocity_df=velocity_to_use
                                        )
                                
                                # For critical path acceleration analysis, prioritize critical path tasks
                                if analysis_type == "Critical Path Acceleration" and critical_path:
                                    st.markdown("#### Critical Path Analysis")
                                    st.markdown("The following tasks are on the critical path and affect the overall project timeline:")
                                    
                                    # Filter active issues to get critical path tasks
                                    critical_tasks = active_issues[active_issues['Issue Key'].isin(critical_path)]
                                    if not critical_tasks.empty:
                                        st.dataframe(critical_tasks[['Issue Key', 'Summary', 'Assignee', 'Status', 'Due Date']], 
                                                     use_container_width=True)
                                
                                # Display recommendations
                                if recommendations:                                            
                                    recom_df = pd.DataFrame(recommendations)
                                    st.dataframe(recom_df, use_container_width=True)
                                    
                                    # Add an apply button
                                    if st.button("Apply Recommendations", key="apply_redis"):
                                        st.success("In a production environment, this would update task assignments in your JIRA instance")
                                        st.info("This is a UI demonstration - no actual changes are being made to the data")
                                        
                                        # Store redistribution plan in history
                                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        if 'redistribution_history' not in st.session_state:
                                            st.session_state['redistribution_history'] = []
                                        
                                        st.session_state['redistribution_history'].append({
                                            "timestamp": timestamp,
                                            "type": analysis_type,
                                            "recommendations": recommendations
                                        })
                                else:
                                    st.info("No specific task redistribution recommendations generated based on current data.")
                        else:
                            st.error("Worklog data missing required columns: 'Resource' or 'Time Spent (hrs)'")
                    else:
                        st.error("Issue data missing required columns: 'Status' or 'Assignee'")
                else:
                    st.error("Missing required data for task redistribution analysis. Please upload JIRA data.")

def analyze_task_dependencies(active_issues, dependencies_df=None):
    """Analyze task dependencies and identify critical path tasks"""
    # If we don't have dependencies data, return empty results
    if dependencies_df is None or dependencies_df.empty:
        return {}, [], {}
    
    # Create a directed graph for task dependencies
    G = nx.DiGraph()
    
    # Add all active issues as nodes
    for _, row in active_issues.iterrows():
        issue_key = row.get('Issue Key')
        if issue_key:
            G.add_node(issue_key, summary=row.get('Summary', ''), 
                      assignee=row.get('Assignee', ''), 
                      status=row.get('Status', ''))
    
    # Add dependency relationships as edges
    for _, row in dependencies_df.iterrows():
        source = row.get('Source Task')
        target = row.get('Dependent Task')
        dep_type = row.get('Dependency Type', 'Finish-to-Start')
        delay = row.get('Delay Impact (days)', 0)
        
        if source in G.nodes and target in G.nodes:
            G.add_edge(source, target, type=dep_type, delay=delay)
    
    # Calculate critical path (longest path through the network)
    critical_path = []
    critical_tasks = set()
    
    # Find the longest path in the graph
    if G.nodes:
        try:
            # Find all simple paths and identify the longest one
            all_paths = []
            for source in [n for n, d in G.in_degree() if d == 0]:  # Start nodes (no incoming edges)
                for target in [n for n, d in G.out_degree() if d == 0]:  # End nodes (no outgoing edges)
                    paths = list(nx.all_simple_paths(G, source, target))
                    if paths:
                        all_paths.extend(paths)
            
            if all_paths:
                # Calculate path lengths based on delay impact
                path_lengths = []
                for path in all_paths:
                    length = sum(G.edges[path[i], path[i+1]].get('delay', 0) 
                                for i in range(len(path)-1))
                    path_lengths.append((path, length))
                
                # Get the longest path
                critical_path = max(path_lengths, key=lambda x: x[1])[0] if path_lengths else []
                critical_tasks = set(critical_path)
        except nx.NetworkXNoPath:
            pass  # No path exists
    
    # Get dependency counts for each task
    dependency_counts = {}
    for node in G.nodes:
        dependency_counts[node] = {
            'predecessors': len(list(G.predecessors(node))),
            'successors': len(list(G.successors(node)))
        }
    
    return G, critical_path, dependency_counts

def visualize_dependencies(G, critical_path=None):
    """Create a visualization of task dependencies with critical path highlighted"""
    if not NETWORK_VISUALIZATION_AVAILABLE:
        return None
        
    if not G or len(G.nodes) == 0:
        return None
    
    try:
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)  # Consistent layout
        
        # Draw regular nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
        
        # Draw critical path nodes in red
        if critical_path:
            critical_nodes = set(critical_path)
            nx.draw_networkx_nodes(G, pos, nodelist=list(critical_nodes), 
                                node_size=500, node_color='red')
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.0, arrowsize=20)
        
        # Add edge labels (dependency types)
        edge_labels = {(u, v): d['type'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        # Add node labels
        node_labels = {}
        for node in G.nodes():
            assignee = G.nodes[node].get('assignee', '')
            summary_short = G.nodes[node].get('summary', '')[:15] + '...' \
                        if len(G.nodes[node].get('summary', '')) > 15 \
                        else G.nodes[node].get('summary', '')
            node_labels[node] = f"{node}\n{assignee}\n{summary_short}"
        
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, 
                            verticalalignment='center')
        
        plt.title("Task Dependency Network (Red = Critical Path)")
        plt.axis('off')
        
        # Save figure to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Convert to base64 for Streamlit
        img = Image.open(buf)
        return img
    except Exception as e:
        import streamlit as st
        st.warning(f"Could not create dependency visualization: {str(e)}")
        return None

def generate_workload_recommendations(active_issues, overloaded, underloaded, workload_analysis, 
                                     skill_map=None, consider_priority=True, dependencies_df=None, 
                                     velocity_df=None):
    """Generate workload balancing recommendations"""
    recommendations = []
    
    # Analyze dependencies if data is available
    dependency_graph, critical_path, dependency_counts = analyze_task_dependencies(active_issues, dependencies_df)
    
    # Get team velocity data if available
    team_velocity = {}
    if velocity_df is not None and not velocity_df.empty:
        # Get average completion rate per team member
        if 'Team Member' in velocity_df.columns and 'Completion Rate' in velocity_df.columns:
            for member, group in velocity_df.groupby('Team Member'):
                team_velocity[member] = group['Completion Rate'].mean()
    
    # Get tasks for overloaded people
    overloaded_tasks = []
    for person in overloaded:
        person_tasks = active_issues[active_issues['Assignee'] == person].reset_index()
        for _, row in person_tasks.iterrows():
            # Create task dictionary with relevant details
            task = {
                "Key": row.get('Issue Key', f"TASK-{_}"),
                "Summary": row.get('Summary', 'Unknown task'),
                "Assignee": person,
                "Priority": row.get('Priority', 'Medium'),
                "Status": row.get('Status', 'In Progress'),
                "Story Points": row.get('Story Points', 0),
                "Complexity": row.get('Complexity Score', 5)  # Default to medium complexity
            }
            
            # Add task dependency information if available
            if dependency_counts and task["Key"] in dependency_counts:
                task["Dependencies"] = dependency_counts[task["Key"]]
                task["Critical"] = task["Key"] in critical_path if critical_path else False
            else:
                task["Dependencies"] = {"predecessors": 0, "successors": 0}
                task["Critical"] = False
                
            overloaded_tasks.append(task)
    
    # Sort tasks by priority and dependency factors
    if overloaded_tasks:
        # Define priority order (High > Medium > Low)
        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        
        def task_score(task):
            # Base score from priority
            priority_score = priority_order.get(task["Priority"], 1) if consider_priority else 1
            
            # Dependency factor (tasks with more dependencies are more critical)
            dependency_score = task["Dependencies"]["successors"] * 2 + task["Dependencies"]["predecessors"]
            
            # Critical path factor (critical path tasks are highest priority)
            critical_factor = 0 if not task["Critical"] else 10
            
            # Complexity factor (higher complexity tasks may need more focused attention)
            complexity_factor = task.get("Complexity", 5) / 2
            
            # Lower score = higher priority for redistribution
            return -(critical_factor + dependency_score + complexity_factor) + priority_score
        
        # Sort tasks by the computed score
        overloaded_tasks.sort(key=task_score)
    
    # Find best assignee for each task based on multiple factors
    for task in overloaded_tasks[:5]:  # Limit to 5 recommendations
        if not underloaded:
            break
            
        # Score each potential assignee
        assignee_scores = []
        for candidate in underloaded:
            score = 0
            justifications = []
            
            # Workload factor
            workload = workload_analysis[workload_analysis['Assignee'] == candidate]['Task Count'].values[0] \
                      if candidate in workload_analysis['Assignee'].values else 0
            workload_factor = 10 - min(workload, 10)  # Lower workload = higher score (max 10)
            score += workload_factor
            justifications.append(f"has capacity ({workload} current tasks)")
            
            # Skill match factor
            if skill_map is not None:
                current_skills = skill_map.get(task['Assignee'], [])
                candidate_skills = skill_map.get(candidate, [])
                matching_skills = set(current_skills) & set(candidate_skills)
                skill_factor = len(matching_skills) * 3  # Each matching skill adds 3 points
                score += skill_factor
                if matching_skills:
                    justifications.append(f"has matching skills in {matching_skills}")
            
            # Velocity factor (higher completion rate = higher score)
            if team_velocity and candidate in team_velocity:
                velocity_factor = team_velocity[candidate] * 5  # Scale to 0-5 points
                score += velocity_factor
                justifications.append(f"has good velocity ({team_velocity[candidate]:.2f} completion rate)")
            
            # Store candidate score and justifications
            assignee_scores.append({
                "assignee": candidate,
                "score": score,
                "justifications": justifications
            })
        
        # Select the best candidate
        if assignee_scores:
            best_assignee = max(assignee_scores, key=lambda x: x["score"])
            
            # Create recommendation with detailed justification
            recommendation = {
                "Task": f"{task['Key']} - {task['Summary']}",
                "Current Assignee": task['Assignee'],
                "Recommended Assignee": best_assignee["assignee"],
                "Justification": f"{task['Assignee']} is overloaded and {best_assignee['assignee']} {', '.join(best_assignee['justifications'])}."  
            }
            
            # Add information about critical path if relevant
            if task["Critical"]:
                recommendation["Justification"] += " This task is on the critical path and affects project timeline."
                
            recommendations.append(recommendation)
            
            # Remove this assignee from candidates to avoid overloading them
            if len(underloaded) > 1:  # Keep at least one person in the list
                underloaded.remove(best_assignee["assignee"])
    
    return recommendations, dependency_graph, critical_path

def generate_leave_impact_recommendations(active_issues, leaves_df, available_resources, workload_analysis, skill_map=None):
    """Generate recommendations for tasks affected by upcoming leaves"""
    recommendations = []
    
    # Convert leave dates to datetime
    if 'Start Date' in leaves_df.columns and 'Resource' in leaves_df.columns:
        leaves_df['Start Date'] = pd.to_datetime(leaves_df['Start Date'], errors='coerce')
        current_date = pd.Timestamp.today()
        
        # Get upcoming leaves in the next 14 days
        upcoming_leaves = leaves_df[(leaves_df['Start Date'] > current_date) & 
                                   (leaves_df['Start Date'] <= current_date + pd.Timedelta(days=14))]
        
        # Find tasks assigned to people who will be on leave soon
        if not upcoming_leaves.empty:
            leave_resources = upcoming_leaves['Resource'].unique()
            
            for resource in leave_resources:
                resource_tasks = active_issues[active_issues['Assignee'] == resource].reset_index()
                leave_start = upcoming_leaves[upcoming_leaves['Resource'] == resource]['Start Date'].min()
                days_until_leave = (leave_start - current_date).days
                
                for _, row in resource_tasks.iterrows():
                    # Only recommend redistribution for tasks that won't be completed before the leave
                    task = {
                        "Key": row.get('Issue Key', f"TASK-{_}"),
                        "Summary": row.get('Summary', 'Unknown task'),
                        "Assignee": resource,
                        "Priority": row.get('Priority', 'Medium'),
                        "Status": row.get('Status', 'In Progress')
                    }
                    
                    # Find best assignee based on skills and workload
                    best_assignee = None
                    if skill_map is not None and resource in skill_map:
                        resource_skills = skill_map.get(resource, [])
                        
                        # Find candidates with matching skills
                        skill_candidates = []
                        for candidate in available_resources:
                            if candidate != resource and candidate in skill_map:
                                candidate_skills = skill_map.get(candidate, [])
                                matching_skills = set(resource_skills) & set(candidate_skills)
                                if matching_skills:
                                    skill_candidates.append({
                                        "assignee": candidate,
                                        "matching_skills": matching_skills,
                                        "workload": workload_analysis[workload_analysis['Assignee'] == candidate]['Task Count'].values[0] 
                                            if candidate in workload_analysis['Assignee'].values else 0
                                    })
                        
                        # Sort candidates by workload (ascending) and matching skills (descending)
                        if skill_candidates:
                            skill_candidates.sort(key=lambda x: (x["workload"], -len(x["matching_skills"])))
                            best_assignee = skill_candidates[0]["assignee"]
                            matching_skills = skill_candidates[0]["matching_skills"]
                    
                    # If no skill match, just pick the least loaded person
                    if best_assignee is None and available_resources:
                        available_workloads = workload_analysis[workload_analysis['Assignee'].isin(available_resources)]
                        if not available_workloads.empty:
                            best_assignee = available_workloads.sort_values('Task Count').iloc[0]['Assignee']
                    
                    # Create recommendation
                    if best_assignee is not None:
                        recommendation = {
                            "Task": f"{task['Key']} - {task['Summary']}",
                            "Current Assignee": resource,
                            "Recommended Assignee": best_assignee,
                            "Justification": f"{resource} will be on leave in {days_until_leave} days. "
                        }
                        
                        if 'matching_skills' in locals() and matching_skills:
                            recommendation["Justification"] += f"Both have matching skills in {matching_skills}."
                            
                        recommendations.append(recommendation)
    
    return recommendations

def detect_token_usage(response):
    """Track token usage from OpenAI API responses"""
    if hasattr(response, 'usage'):
        return {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    return None
