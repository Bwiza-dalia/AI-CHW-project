from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def compute_analytics() -> Dict:
    """
    Computes comprehensive CHW engagement metrics, progress, regional comparison, and predictive analytics.
    Returns a dict with enhanced analytics results including trends, distributions, and insights.
    """
    event_file = os.path.join(os.path.dirname(__file__), '../../data/simulated_chw_events.csv')
    df = pd.read_csv(event_file)
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.isocalendar().week
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.day_name()
    
    today = df['date'].max()
    last_7 = today - pd.Timedelta(days=7)
    last_30 = today - pd.Timedelta(days=30)
    
    # === CORE METRICS ===
    total_chws = df['chw_id'].nunique()
    total_sessions = len(df)
    daily_active = df.groupby('date')['chw_id'].nunique().to_dict()
    weekly_active = df[df['date'] >= last_7]['chw_id'].nunique()
    monthly_active = df[df['date'] >= last_30]['chw_id'].nunique()
    
    # === ENGAGEMENT METRICS ===
    avg_session = df['session_length_min'].mean()
    median_session = df['session_length_min'].median()
    total_quiz_attempts = df['quiz_attempts'].sum()
    avg_score = df['avg_score'].mean()
    median_score = df['avg_score'].median()
    
    # === COMPLETION & RETENTION ===
    sessions_per_chw = df.groupby('chw_id').size()
    completion_rate = (sessions_per_chw >= 10).mean()
    high_engagement = (sessions_per_chw >= 20).mean()
    retention_7d = (df[df['date'] >= last_7]['chw_id'].nunique() / total_chws) if total_chws > 0 else 0
    
    # === REGIONAL ANALYSIS ===
    regional_stats = df.groupby('region').agg({
        'chw_id': 'nunique',
        'avg_score': ['mean', 'std'],
        'quiz_attempts': ['sum', 'mean'],
        'session_length_min': ['mean', 'std'],
        'date': 'count'
    }).round(2)
    
    regional_stats.columns = ['total_chws', 'avg_score_mean', 'avg_score_std', 
                             'total_quiz_attempts', 'avg_quiz_attempts', 
                             'avg_session_length', 'session_length_std', 'total_sessions']
    regional_comparison = regional_stats.reset_index().to_dict(orient='records')
    
    # === TREND ANALYSIS ===
    daily_trends = df.groupby('date').agg({
        'chw_id': 'nunique',
        'session_length_min': 'mean',
        'avg_score': 'mean',
        'quiz_attempts': 'sum'
    }).reset_index()
    daily_trends.columns = ['date', 'active_chws', 'avg_session_length', 'avg_score', 'total_quiz_attempts']
    
    weekly_trends = df.groupby('week').agg({
        'chw_id': 'nunique',
        'session_length_min': 'mean',
        'avg_score': 'mean',
        'quiz_attempts': 'sum'
    }).reset_index()
    weekly_trends.columns = ['week', 'active_chws', 'avg_session_length', 'avg_score', 'total_quiz_attempts']
    
    # === PERFORMANCE DISTRIBUTION ===
    score_distribution = df['avg_score'].dropna()
    score_ranges = {
        'excellent': (score_distribution >= 4.5).sum(),
        'good': ((score_distribution >= 3.5) & (score_distribution < 4.5)).sum(),
        'average': ((score_distribution >= 2.5) & (score_distribution < 3.5)).sum(),
        'needs_improvement': (score_distribution < 2.5).sum()
    }
    
    session_length_ranges = {
        'short': (df['session_length_min'] < 30).sum(),
        'medium': ((df['session_length_min'] >= 30) & (df['session_length_min'] < 60)).sum(),
        'long': (df['session_length_min'] >= 60).sum()
    }
    
    # === PREDICTIVE ANALYTICS ===
    recent_data = df[df['date'] >= last_7]
    chw_performance = recent_data.groupby('chw_id').agg({
        'session_length_min': 'mean',
        'avg_score': 'mean',
        'quiz_attempts': 'sum',
        'date': 'count'
    }).round(2)
    chw_performance.columns = ['avg_session_length', 'avg_score', 'total_quiz_attempts', 'session_count']
    
    # Flag CHWs at risk
    at_risk_chws = chw_performance[
        (chw_performance['session_count'] < 2) | 
        (chw_performance['avg_score'] < 3.0) |
        (chw_performance['avg_session_length'] < 20)
    ].index.tolist()
    
    # Top performers
    top_performers = chw_performance[
        (chw_performance['avg_score'] >= 4.0) & 
        (chw_performance['session_count'] >= 3)
    ].sort_values('avg_score', ascending=False).head(10).index.tolist()
    
    # === INSIGHTS & RECOMMENDATIONS ===
    insights = generate_insights(df, regional_comparison, score_distribution, daily_trends)
    
    # === WEEKLY PATTERNS ===
    day_of_week_activity = df.groupby('day_of_week')['chw_id'].nunique().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ]).fillna(0).to_dict()
    
    return {
        # Core metrics
        'total_chws': total_chws,
        'total_sessions': total_sessions,
        'daily_active': daily_active,
        'weekly_active': weekly_active,
        'monthly_active': monthly_active,
        
        # Engagement metrics
        'avg_session_length': round(avg_session, 2),
        'median_session_length': round(median_session, 2),
        'total_quiz_attempts': int(total_quiz_attempts),
        'avg_score': round(avg_score, 2) if pd.notnull(avg_score) else 0,
        'median_score': round(median_score, 2) if pd.notnull(median_score) else 0,
        
        # Completion & retention
        'completion_rate': round(completion_rate, 3),
        'high_engagement_rate': round(high_engagement, 3),
        'retention_7d': round(retention_7d, 3),
        
        # Regional analysis
        'regional_comparison': regional_comparison,
        
        # Trends
        'daily_trends': daily_trends.to_dict(orient='records'),
        'weekly_trends': weekly_trends.to_dict(orient='records'),
        
        # Distributions
        'score_distribution': score_ranges,
        'session_length_distribution': session_length_ranges,
        
        # Predictive analytics
        'at_risk_chws': at_risk_chws,
        'top_performers': top_performers,
        'chw_performance': chw_performance.to_dict(orient='index'),
        
        # Patterns
        'day_of_week_activity': day_of_week_activity,
        
        # Insights
        'insights': insights
    }

def generate_insights(df: pd.DataFrame, regional_data: List[Dict], score_dist: pd.Series, daily_trends: pd.DataFrame) -> List[str]:
    """Generate actionable insights from the analytics data."""
    insights = []
    
    # Score insights
    if score_dist.mean() < 3.0:
        insights.append("⚠️ Average scores are below 3.0 - consider additional training support")
    elif score_dist.mean() > 4.0:
        insights.append("✅ Excellent performance! Average scores above 4.0")
    
    # Regional insights
    if regional_data:
        best_region = max(regional_data, key=lambda x: x['avg_score_mean'])
        worst_region = min(regional_data, key=lambda x: x['avg_score_mean'])
        insights.append(f"🏆 {best_region['region']} is the top-performing region (avg score: {best_region['avg_score_mean']:.1f})")
        insights.append(f"📈 {worst_region['region']} needs support (avg score: {worst_region['avg_score_mean']:.1f})")
    
    # Trend insights
    if len(daily_trends) >= 7:
        recent_avg = daily_trends['active_chws'].tail(7).mean()
        previous_avg = daily_trends['active_chws'].head(7).mean() if len(daily_trends) >= 14 else recent_avg
        if recent_avg > previous_avg * 1.1:
            insights.append("📈 Daily active CHWs are increasing - great engagement!")
        elif recent_avg < previous_avg * 0.9:
            insights.append("📉 Daily active CHWs are decreasing - consider engagement strategies")
    
    # Session length insights
    avg_session = df['session_length_min'].mean()
    if avg_session < 30:
        insights.append("⏱️ Average session length is short - consider content optimization")
    elif avg_session > 60:
        insights.append("⏱️ Long session lengths suggest high engagement")
    
    return insights
