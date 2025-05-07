"""
Visualization enhancement functions for the AI PM Buddy app.
These functions improve chart appearance, fix label positioning, and optimize layout.
"""

import plotly.graph_objects as go
import plotly.express as px

def fix_axis_labels(fig, x_title=None, y_title=None, colors=None):
    """Fix axis labels positioning to prevent cropping"""
    updates = {}
    
    if x_title:
        updates['xaxis'] = {
            'title': x_title,
            'title_standoff': 30  # More space for title
        }
    
    if y_title:
        updates['yaxis'] = {
            'title': y_title,
            'title_standoff': 30  # More space for title
        }

    fig.update_layout(**updates)
    
    # Ensure label text is visible with proper color
    if colors and 'text' in colors:
        if x_title:
            fig.update_xaxes(title_font=dict(color=colors['text'], size=13))
        if y_title:
            fig.update_yaxes(title_font=dict(color=colors['text'], size=13))
    
    return fig

def optimize_chart_layout(fig, chart_type="default", colors=None):
    """Optimize chart layout based on chart type to prevent overlapping and improve spacing"""
    # Common optimizations for all chart types
    fig.update_layout(
        margin=dict(l=70, r=50, t=80, b=80),  # Increased margins
        hovermode="closest"
    )
    
    # Chart-specific optimizations
    if chart_type == "burnup":
        # Burnup charts need more space for date labels
        fig.update_layout(
            margin=dict(l=70, r=50, t=80, b=100),  # More bottom space for date labels
            xaxis=dict(
                tickangle=-45,  # Angled date labels
                tickformat='%b %d',  # Month-day format
                tickmode='auto',
                nticks=10
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            hovermode='x unified'  # Show all traces for a given x-value
        )
    
    elif chart_type == "gantt":
        # Gantt charts need special handling for date ranges
        fig.update_layout(
            margin=dict(l=100, r=50, t=80, b=80),  # More left space for resource names
            xaxis=dict(
                tickangle=-45,
                tickformat='%b %d',  # Month-day format
                type='date'
            ),
            yaxis=dict(
                autorange="reversed"  # Reverse y-axis so top items appear first
            )
        )
    
    elif chart_type == "bubble":
        # Bubble charts need more space for bubbles and legends
        fig.update_layout(
            margin=dict(l=70, r=50, t=80, b=120),  # More bottom space for legend
            legend=dict(
                orientation='h',
                yanchor='bottom', 
                y=-0.3,
                xanchor='center',
                x=0.5
            ),
            # Remove gridlines for cleaner look
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
    
    elif chart_type == "task_distribution":
        # Task distribution bar charts need room for labels
        fig.update_layout(
            margin=dict(l=70, r=50, t=80, b=100),
            xaxis=dict(
                tickangle=-45  # Angled text for category labels
            ),
            bargap=0.3,  # Spacing between bars
            bargroupgap=0.1  # Spacing between bar groups
        )
    
    return fig

def enhance_chart_for_dark_mode(fig, colors):
    """Enhance charts specifically for dark mode"""
    if colors['background'] != '#FFFFFF':  # Assuming light mode has white background
        # Improve contrast for dark mode
        fig.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font=dict(color=colors['text']),
            title_font=dict(color=colors['text']),
            legend=dict(
                font=dict(color=colors['text']),
                bgcolor=colors['background'],
                bordercolor=colors['grid']
            )
        )
        
        # Make grid lines more visible but subtle
        fig.update_xaxes(
            gridcolor=colors['grid'],
            linecolor=colors['grid'],
            zerolinecolor=colors['grid'],
            tickfont=dict(color=colors['text'])
        )
        
        fig.update_yaxes(
            gridcolor=colors['grid'],
            linecolor=colors['grid'],
            zerolinecolor=colors['grid'],
            tickfont=dict(color=colors['text'])
        )
        
        # Adjust marker borders for better visibility
        for trace in fig.data:
            if hasattr(trace, 'marker') and trace.marker:
                if not trace.marker.line:
                    trace.marker.line = dict(width=1, color=colors['background'])
    
    return fig

def improve_sprint_burnup_chart(fig, colors):
    """Specific improvements for sprint burnup charts"""
    # Check if it's a burnup chart by looking for specific trace names
    trace_names = [trace.name for trace in fig.data if hasattr(trace, 'name') and trace.name]
    burnup_terms = ['Completed', 'Ideal', 'Total']
    
    if any(term in ' '.join(trace_names) for term in burnup_terms):
        # Customize line appearance
        for i, trace in enumerate(fig.data):
            if hasattr(trace, 'name'):
                if 'Completed' in trace.name:
                    trace.line.width = 3  # Thicker line for completed
                elif 'Ideal' in trace.name:
                    trace.line.dash = 'dot'
                    trace.line.width = 2
                elif 'Total' in trace.name:
                    trace.line.dash = 'dash'
                    trace.line.width = 2
        
        # Improve date axis
        fig.update_xaxes(
            tickangle=-45,
            tickformat='%b %d',
            nticks=10,
            type='date'
        )
        
        # Improve y-axis
        fig.update_yaxes(
            zeroline=True,
            zerolinecolor=colors['grid'],
            zerolinewidth=1.5
        )
        
        # Better interactive features
        fig.update_layout(
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
    
    return fig

def improve_task_distribution_chart(fig, colors):
    """Specific improvements for task distribution bar charts"""
    # Check if it's a bar chart with task-related categories
    if fig.data and hasattr(fig.data[0], 'type') and fig.data[0].type == 'bar':
        # Add value labels above bars
        for trace in fig.data:
            trace.textposition = 'outside'
            trace.texttemplate = '%{y}'
            trace.marker.line.width = 1
            trace.marker.line.color = colors['background']
        
        # Improve bar appearance
        fig.update_layout(
            bargap=0.3,
            bargroupgap=0.1,
            uniformtext=dict(minsize=10, mode='hide')  # Ensure text is readable
        )
        
        # Better y-axis for counts
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=colors['grid'],
            zeroline=True,
            zerolinecolor=colors['grid'],
            zerolinewidth=1.5
        )
        
        # Clean x-axis
        fig.update_xaxes(
            showgrid=False,
            tickangle=-45 if len(fig.data[0].x) > 3 else 0  # Angle labels only if many categories
        )
    
    return fig

def improve_gantt_chart(fig, colors):
    """Specific improvements for gantt timeline charts"""
    # Check if it's a timeline/gantt chart
    if fig.data and hasattr(fig.data[0], 'mode') and fig.data[0].mode == 'lines':
        # This is likely a gantt chart using Plotly's timeline
        
        # Improve bar appearance
        for trace in fig.data:
            if hasattr(trace, 'marker'):
                trace.marker.line.width = 1
                trace.marker.line.color = colors['background']
        
        # Better date axis
        fig.update_xaxes(
            type='date',
            tickformat='%b %d',
            tickangle=-45,
            showgrid=True,
            gridcolor=colors['grid'],
            gridwidth=0.5
        )
        
        # Better y-axis for resources
        fig.update_yaxes(
            autorange="reversed",  # Top items first
            showgrid=False,  # No horizontal gridlines
            zeroline=False
        )
        
        # Better hovering
        fig.update_layout(
            hovermode='closest'
        )
    
    return fig

def improve_bubble_chart(fig, colors):
    """Specific improvements for bubble charts like workload vs velocity"""
    # Check if it's a scatter plot with size variation (bubble chart)
    if (fig.data and hasattr(fig.data[0], 'type') and fig.data[0].type == 'scatter' and 
        hasattr(fig.data[0], 'marker') and hasattr(fig.data[0].marker, 'size')):
        
        # Improve bubble appearance
        for trace in fig.data:
            # Add thin border to bubbles
            trace.marker.line.width = 1
            trace.marker.line.color = colors['background']
            # Add some transparency
            if not hasattr(trace, 'opacity') or trace.opacity is None:
                trace.opacity = 0.85
        
        # Better axis
        fig.update_xaxes(
            zeroline=True,
            zerolinecolor=colors['grid'],
            zerolinewidth=1,
            showgrid=True,
            gridcolor=colors['grid'],
            gridwidth=0.5
        )
        
        fig.update_yaxes(
            zeroline=True,
            zerolinecolor=colors['grid'],
            zerolinewidth=1,
            showgrid=True,
            gridcolor=colors['grid'],
            gridwidth=0.5
        )
        
        # Better legend position for bubble charts
        fig.update_layout(
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.3,
                xanchor='center',
                x=0.5
            ),
            hovermode='closest'
        )
    
    return fig