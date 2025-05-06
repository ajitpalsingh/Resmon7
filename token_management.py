# Token Management for GPT Interactions
# Tracks and manages OpenAI token usage to optimize costs and prevent quota limits

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import json

class TokenManager:
    def __init__(self):
        """Initialize token manager"""
        # Initialize token tracking in session state if not present
        if 'token_usage_history' not in st.session_state:
            st.session_state['token_usage_history'] = []
        
        # Set default token limits
        if 'token_limit_daily' not in st.session_state:
            st.session_state['token_limit_daily'] = 100000  # Default daily limit
            
        if 'token_limit_monthly' not in st.session_state:
            st.session_state['token_limit_monthly'] = 3000000  # Default monthly limit
        
        # Calculate token statistics
        self.calculate_statistics()
    
    def track_usage(self, response):
        """Track token usage from OpenAI API responses"""
        # Only track if the response has a usage attribute
        if hasattr(response, 'usage'):
            usage_data = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens,
                'model': response.model,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'feature': self._get_current_feature()
            }
            
            # Add to history
            st.session_state['token_usage_history'].append(usage_data)
            
            # Recalculate statistics
            self.calculate_statistics()
            
            return usage_data
        
        return None
    
    def _get_current_feature(self):
        """Determine which feature is currently using tokens"""
        # This is a placeholder - in a real implementation, this would 
        # use the call stack or context to determine which feature 
        # made the API call
        return "Unknown"
    
    def calculate_statistics(self):
        """Calculate token usage statistics"""
        history = st.session_state['token_usage_history']
        
        # Initialize statistics
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.daily_tokens = 0
        self.monthly_tokens = 0
        self.feature_usage = {}
        
        # Current date for comparison
        now = datetime.now()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Calculate statistics
        for entry in history:
            # Convert timestamp to datetime
            try:
                entry_time = datetime.strptime(entry['timestamp'], "%Y-%m-%d %H:%M:%S")
                
                # Update totals
                self.total_tokens += entry['total_tokens']
                self.total_prompt_tokens += entry['prompt_tokens']
                self.total_completion_tokens += entry['completion_tokens']
                
                # Check if entry is from today
                if entry_time >= today:
                    self.daily_tokens += entry['total_tokens']
                
                # Check if entry is from this month
                if entry_time >= month_start:
                    self.monthly_tokens += entry['total_tokens']
                
                # Track by feature
                feature = entry.get('feature', 'Unknown')
                if feature not in self.feature_usage:
                    self.feature_usage[feature] = 0
                self.feature_usage[feature] += entry['total_tokens']
                
            except Exception:
                pass
        
        # Calculate daily limit status
        self.daily_limit = st.session_state['token_limit_daily']
        self.daily_limit_pct = min(100, round(self.daily_tokens / self.daily_limit * 100, 1)) if self.daily_limit > 0 else 0
        
        # Calculate monthly limit status
        self.monthly_limit = st.session_state['token_limit_monthly']
        self.monthly_limit_pct = min(100, round(self.monthly_tokens / self.monthly_limit * 100, 1)) if self.monthly_limit > 0 else 0
        
        # Estimate cost (approximate based on current prices)
        # Updated rates as of May 2025
        self.estimated_cost = (self.total_prompt_tokens / 1000 * 0.0015) + (self.total_completion_tokens / 1000 * 0.002)
    
    def get_usage_warnings(self):
        """Get warnings about token usage limits"""
        warnings = []
        
        # Check daily limit
        if self.daily_limit_pct >= 90:
            warnings.append(f"⚠️ CRITICAL: Daily token usage at {self.daily_limit_pct}% of limit")
        elif self.daily_limit_pct >= 75:
            warnings.append(f"⚠️ WARNING: Daily token usage at {self.daily_limit_pct}% of limit")
        
        # Check monthly limit
        if self.monthly_limit_pct >= 90:
            warnings.append(f"⚠️ CRITICAL: Monthly token usage at {self.monthly_limit_pct}% of limit")
        elif self.monthly_limit_pct >= 75:
            warnings.append(f"⚠️ WARNING: Monthly token usage at {self.monthly_limit_pct}% of limit")
        
        return warnings
    
    def should_throttle(self):
        """Determine if token usage should be throttled"""
        return self.daily_limit_pct >= 95 or self.monthly_limit_pct >= 95
    
    def display_token_dashboard(self):
        """Display token usage dashboard"""
        st.markdown("### 💰 GPT Token Usage Dashboard")
        st.markdown("""
        This dashboard helps you monitor and manage your OpenAI API token usage to control costs 
        and prevent quota limits from affecting your project management activities.
        """)
        
        # Display warnings at the top if any
        warnings = self.get_usage_warnings()
        if warnings:
            for warning in warnings:
                st.warning(warning)
        
        # Daily and Monthly Usage overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Daily Token Usage", 
                f"{self.daily_tokens:,}/{self.daily_limit:,}", 
                f"{self.daily_limit_pct}%"
            )
            progress_color = 'normal' if self.daily_limit_pct < 75 else ('warning' if self.daily_limit_pct < 90 else 'error')
            st.progress(self.daily_limit_pct / 100, text=f"{self.daily_limit_pct}%")
            
        with col2:
            st.metric(
                "Monthly Token Usage", 
                f"{self.monthly_tokens:,}/{self.monthly_limit:,}", 
                f"{self.monthly_limit_pct}%"
            )
            progress_color = 'normal' if self.monthly_limit_pct < 75 else ('warning' if self.monthly_limit_pct < 90 else 'error')
            st.progress(self.monthly_limit_pct / 100, text=f"{self.monthly_limit_pct}%")
        
        # Usage by Feature
        st.subheader("Token Usage by Feature")
        if self.feature_usage:
            # Convert to DataFrame for visualization
            feature_df = pd.DataFrame({
                'Feature': list(self.feature_usage.keys()),
                'Tokens': list(self.feature_usage.values())
            })
            
            # Sort by usage
            feature_df = feature_df.sort_values('Tokens', ascending=False)
            
            # Create chart
            fig = px.bar(
                feature_df,
                x='Feature',
                y='Tokens',
                title="Token Usage by Feature",
                color='Tokens',
                color_continuous_scale='Blues'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No feature-specific usage data available yet.")
        
        # Usage over time
        st.subheader("Token Usage Over Time")
        if st.session_state['token_usage_history']:
            # Convert history to DataFrame
            history_df = pd.DataFrame(st.session_state['token_usage_history'])
            history_df['date'] = pd.to_datetime(history_df['timestamp'])
            
            # Group by day
            daily_usage = history_df.groupby(history_df['date'].dt.date)['total_tokens'].sum().reset_index()
            daily_usage.columns = ['Date', 'Tokens']
            
            # Create time series chart
            fig = px.line(
                daily_usage,
                x='Date',
                y='Tokens',
                title="Daily Token Usage",
                markers=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No historical usage data available yet.")
        
        # Cost estimate
        st.metric("Estimated API Cost", f"${self.estimated_cost:.2f}")
        
        # Settings for token limits
        with st.expander("Token Limit Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_daily_limit = st.number_input(
                    "Daily Token Limit", 
                    min_value=1000, 
                    max_value=1000000, 
                    value=st.session_state['token_limit_daily'],
                    step=1000
                )
            
            with col2:
                new_monthly_limit = st.number_input(
                    "Monthly Token Limit", 
                    min_value=10000, 
                    max_value=10000000, 
                    value=st.session_state['token_limit_monthly'],
                    step=10000
                )
            
            if st.button("Update Limits"):
                st.session_state['token_limit_daily'] = new_daily_limit
                st.session_state['token_limit_monthly'] = new_monthly_limit
                self.calculate_statistics()
                st.success("Token limits updated successfully!")
            
            # Option to clear history
            if st.button("Clear Usage History", type="secondary"):
                st.session_state['token_usage_history'] = []
                self.calculate_statistics()
                st.success("Usage history cleared successfully!")
                st.experimental_rerun()

def optimize_prompt(prompt, max_tokens=4000):
    """Optimize a prompt to reduce token usage"""
    # Simple optimization by truncating prompt if too long
    if len(prompt) > max_tokens * 4:  # rough estimate as 1 token ≈ 4 chars
        # Keep the first and last parts of the prompt
        prompt_start = prompt[:max_tokens * 2]  # Keep start (instructions, etc)
        prompt_end = prompt[-max_tokens * 2:]   # Keep end (recent context)
        return prompt_start + "\n[...content truncated to save tokens...]\n" + prompt_end
    
    return prompt
