"""
Reconciliation Dashboard - Streamlit App
==========================================
Complete dashboard for discrepancy tracking and revenue analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

import os

# =============================================================================
# CONFIGURATION
# =============================================================================

# Page config
st.set_page_config(
    page_title="Reconciliation Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
# =============================================================================
# DATABASE CONNECTION
# =============================================================================

@st.cache_resource
def get_connection():
    """Create database connection"""
    try:
        engine = create_engine(DATABASE_URL)
        return engine
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_data(ttl=300)
def load_discrepancies():
    """Load discrepancy data from database"""
    query = """
    SELECT 
        id,
        reconciliation_date,
        TO_CHAR(reconciliation_date, 'YYYY-MM') as year_month,
        TO_CHAR(reconciliation_date, 'Mon YYYY') as month_display,
        TO_CHAR(reconciliation_date, 'Day') as day_of_week,
        practitioner,
        site,
        session_slot,
        terminal,
        type as discrepancy_type,
        CASE 
            WHEN type = 'missing_in_bank' THEN 'Missing from Bank'
            WHEN type = 'extra_in_bank' THEN 'Extra in Bank'
            ELSE type
        END as type_display,
        COALESCE(booking_total, 0) as booking_total,
        COALESCE(booking_count, 0) as booking_count,
        COALESCE(transaction_total, 0) as transaction_total,
        COALESCE(transaction_count, 0) as transaction_count,
        COALESCE(difference, 0) as difference,
        status,
        notes,
        COALESCE(is_reception, false) as is_reception,
        created_at,
        resolved_at,
        resolved_by
    FROM reconciliation_discrepancies
    ORDER BY reconciliation_date DESC, practitioner
    """
    engine = get_connection()
    if engine:
        return pd.read_sql(query, engine)
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_revenue_by_practitioner():
    """Load revenue data grouped by practitioner"""
    query = """
    SELECT 
        practitioner,
        TO_CHAR(appointment_date, 'YYYY-MM') as year_month,
        TO_CHAR(appointment_date, 'Mon YYYY') as month_display,
        SUM(appointments) as total_appointments,
        SUM(net_appointments) as net_appointments,
        SUM(revenue) as total_revenue,
        SUM(cash) as total_cash,
        SUM(card) as total_card,
        CASE 
            WHEN SUM(net_appointments) > 0 
            THEN ROUND(SUM(revenue)::numeric / SUM(net_appointments), 2)
            ELSE 0
        END as avg_per_appointment,
        COUNT(DISTINCT appointment_date) as sessions,
        CASE 
            WHEN COUNT(DISTINCT appointment_date) > 0 
            THEN ROUND(SUM(revenue)::numeric / COUNT(DISTINCT appointment_date), 2)
            ELSE 0
        END as avg_per_session,
        SUM(no_wax) as no_wax,
        SUM(no_show) as no_show,
        SUM(follow_up) as follow_up,
        CASE 
            WHEN SUM(appointments) > 0 
            THEN ROUND((SUM(no_wax)::numeric / SUM(appointments)) * 100, 2)
            ELSE 0
        END as no_wax_ratio,
        SUM(session_length) as total_hours,
        CASE 
            WHEN SUM(session_length) > 0 
            THEN ROUND(SUM(appointments)::numeric / SUM(session_length), 2)
            ELSE 0
        END as bookings_per_hour,
        CASE 
            WHEN SUM(session_length) > 0 
            THEN ROUND((SUM(appointments)::numeric / (SUM(session_length) * 4)) * 100, 2)
            ELSE 0
        END as occupancy_rate
    FROM revenue_data
    WHERE appointment_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY practitioner, TO_CHAR(appointment_date, 'YYYY-MM'), TO_CHAR(appointment_date, 'Mon YYYY')
    ORDER BY year_month DESC, total_revenue DESC
    """
    engine = get_connection()
    if engine:
        return pd.read_sql(query, engine)
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_revenue_by_site():
    """Load revenue data grouped by site"""
    query = """
    SELECT 
        site,
        TO_CHAR(appointment_date, 'YYYY-MM') as year_month,
        TO_CHAR(appointment_date, 'Mon YYYY') as month_display,
        SUM(appointments) as total_appointments,
        SUM(net_appointments) as net_appointments,
        SUM(revenue) as total_revenue,
        SUM(cash) as total_cash,
        SUM(card) as total_card,
        CASE 
            WHEN SUM(net_appointments) > 0 
            THEN ROUND(SUM(revenue)::numeric / SUM(net_appointments), 2)
            ELSE 0
        END as avg_per_appointment,
        COUNT(DISTINCT appointment_date || practitioner) as sessions,
        CASE 
            WHEN COUNT(DISTINCT appointment_date || practitioner) > 0 
            THEN ROUND(SUM(revenue)::numeric / COUNT(DISTINCT appointment_date || practitioner), 2)
            ELSE 0
        END as avg_per_session,
        SUM(no_wax) as no_wax,
        CASE 
            WHEN SUM(appointments) > 0 
            THEN ROUND((SUM(no_wax)::numeric / SUM(appointments)) * 100, 2)
            ELSE 0
        END as no_wax_ratio,
        SUM(session_length) as total_hours,
        CASE 
            WHEN SUM(session_length) > 0 
            THEN ROUND(SUM(appointments)::numeric / SUM(session_length), 2)
            ELSE 0
        END as bookings_per_hour,
        CASE 
            WHEN SUM(session_length) > 0 
            THEN ROUND((SUM(appointments)::numeric / (SUM(session_length) * 4)) * 100, 2)
            ELSE 0
        END as occupancy_rate
    FROM revenue_data
    WHERE appointment_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY site, TO_CHAR(appointment_date, 'YYYY-MM'), TO_CHAR(appointment_date, 'Mon YYYY')
    ORDER BY year_month DESC, total_revenue DESC
    """
    engine = get_connection()
    if engine:
        return pd.read_sql(query, engine)
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_monthly_summary():
    """Load monthly summary data"""
    query = """
    SELECT 
        TO_CHAR(appointment_date, 'YYYY-MM') as year_month,
        TO_CHAR(appointment_date, 'Mon YYYY') as month_display,
        COUNT(DISTINCT practitioner) as total_practitioners,
        COUNT(DISTINCT site) as total_sites,
        SUM(appointments) as total_appointments,
        SUM(revenue) as total_revenue,
        SUM(cash) as total_cash,
        SUM(card) as total_card,
        SUM(no_wax) as total_no_wax,
        SUM(no_show) as total_no_show,
        SUM(session_length) as total_hours
    FROM revenue_data
    WHERE appointment_date >= CURRENT_DATE - INTERVAL '180 days'
    GROUP BY TO_CHAR(appointment_date, 'YYYY-MM'), TO_CHAR(appointment_date, 'Mon YYYY')
    ORDER BY year_month DESC
    """
    engine = get_connection()
    if engine:
        return pd.read_sql(query, engine)
    return pd.DataFrame()

# =============================================================================
# STYLING FUNCTIONS
# =============================================================================

def highlight_reception(row):
    """Highlight reception terminal rows"""
    if row.get('is_reception', False):
        return ['background-color: #FFE4B5'] * len(row)
    return [''] * len(row)

def highlight_discrepancy_type(row):
    """Color code by discrepancy type"""
    if row.get('discrepancy_type') == 'missing_in_bank':
        return ['background-color: #FFCCCB'] * len(row)
    elif row.get('discrepancy_type') == 'extra_in_bank':
        return ['background-color: #90EE90'] * len(row)
    return [''] * len(row)

def color_occupancy(val):
    """Color code occupancy rate"""
    if pd.isna(val):
        return ''
    if val >= 60:
        return 'background-color: #008000'  # Green
    elif val >= 40:
        return 'background-color: #FFAA33'   # Light Yellow
    elif val >= 30:
        return 'background-color: #FFA500'  # Orange
    else:
        return 'background-color: #FF0000'  # Red

# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render sidebar with navigation and filters"""
    st.sidebar.title("📊 Dashboard")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigate",
        ["🔍 Discrepancies", "💰 Revenue - Practitioners", "🏥 Revenue - Sites", "📈 Monthly Overview"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Filters")
    
    return page

# =============================================================================
# DISCREPANCY DASHBOARD
# =============================================================================

def render_discrepancy_dashboard():
    """Render the discrepancy dashboard page"""
    st.title("🔍 Reconciliation Discrepancy Dashboard")
    
    # Load data
    df = load_discrepancies()
    
    if df.empty:
        st.warning("No discrepancy data found. Please check database connection.")
        return
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        months = ['All'] + sorted(df['year_month'].dropna().unique().tolist(), reverse=True)
        selected_month = st.selectbox("📅 Month", months, key="disc_month")
    
    with col2:
        practitioners = ['All'] + sorted(df['practitioner'].dropna().unique().tolist())
        selected_practitioner = st.selectbox("👤 Practitioner", practitioners, key="disc_pract")
    
    with col3:
        sites = ['All'] + sorted(df['site'].dropna().unique().tolist())
        selected_site = st.selectbox("📍 Site", sites, key="disc_site")
    
    with col4:
        types = ['All', 'Missing from Bank', 'Extra in Bank']
        selected_type = st.selectbox("🏷️ Type", types, key="disc_type")
    
    # Additional filters
    col5, col6, col7 = st.columns(3)
    
    with col5:
        statuses = ['All', 'pending', 'resolved']
        selected_status = st.selectbox("📋 Status", statuses, key="disc_status")
    
    with col6:
        show_reception = st.checkbox("🏢 Show Reception Only", value=False)
    
    with col7:
        hide_reception = st.checkbox("🚫 Hide Reception", value=False)
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_month != 'All':
        filtered_df = filtered_df[filtered_df['year_month'] == selected_month]
    
    if selected_practitioner != 'All':
        filtered_df = filtered_df[filtered_df['practitioner'] == selected_practitioner]
    
    if selected_site != 'All':
        filtered_df = filtered_df[filtered_df['site'] == selected_site]
    
    if selected_type != 'All':
        filtered_df = filtered_df[filtered_df['type_display'] == selected_type]
    
    if selected_status != 'All':
        filtered_df = filtered_df[filtered_df['status'] == selected_status]
    
    if show_reception:
        filtered_df = filtered_df[filtered_df['is_reception'] == True]
    
    if hide_reception:
        filtered_df = filtered_df[filtered_df['is_reception'] == False]
    
    # Metrics
    st.markdown("---")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📊 Total Discrepancies", len(filtered_df))
    
    with col2:
        missing = filtered_df[filtered_df['discrepancy_type'] == 'missing_in_bank']['difference'].sum()
        st.metric("❌ Total Missing", f"£{missing:,.2f}")
    
    with col3:
        extra = filtered_df[filtered_df['discrepancy_type'] == 'extra_in_bank']['difference'].sum()
        st.metric("💰 Total Extra", f"£{extra:,.2f}")
    
    with col4:
        pending = len(filtered_df[filtered_df['status'] == 'pending'])
        st.metric("⏳ Pending", pending)
    
    with col5:
        reception_count = len(filtered_df[filtered_df['is_reception'] == True])
        st.metric("🏢 Reception", reception_count)
    
    # Charts
    st.markdown("---")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("📊 By Type")
        if not filtered_df.empty:
            type_counts = filtered_df['type_display'].value_counts()
            fig = px.pie(
                values=type_counts.values, 
                names=type_counts.index,
                color=type_counts.index,
                color_discrete_map={
                    'Missing from Bank': '#FF6B6B',
                    'Extra in Bank': '#4ECDC4'
                }
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        st.subheader("📊 By Practitioner (Top 10)")
        if not filtered_df.empty:
            pract_sum = filtered_df.groupby('practitioner')['difference'].sum().sort_values(ascending=True).tail(10)
            fig = px.bar(
                x=pract_sum.values,
                y=pract_sum.index,
                orientation='h',
                color=pract_sum.values,
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Time series
    st.subheader("📈 Discrepancies Over Time")
    if not filtered_df.empty:
        daily = filtered_df.groupby('reconciliation_date').agg({
            'difference': 'sum',
            'id': 'count'
        }).reset_index()
        daily.columns = ['Date', 'Amount', 'Count']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily['Date'], y=daily['Amount'], name='Amount (£)'))
        fig.add_trace(go.Scatter(x=daily['Date'], y=daily['Count'], name='Count', yaxis='y2', line=dict(color='red')))
        fig.update_layout(
            yaxis=dict(title='Amount (£)'),
            yaxis2=dict(title='Count', overlaying='y', side='right'),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.markdown("---")
    st.subheader("📋 Discrepancy Details")
    
    # Select columns to display
    display_cols = [
        'reconciliation_date', 'practitioner', 'site', 'session_slot',
        'type_display', 'booking_total', 'transaction_total',
        'difference', 'status', 'is_reception', 'terminal'
    ]
    
    # Filter columns that exist
    display_cols = [col for col in display_cols if col in filtered_df.columns]
    
    if not filtered_df.empty:
        # Style the dataframe
        styled_df = filtered_df[display_cols].copy()
        styled_df = styled_df.rename(columns={
            'reconciliation_date': 'Date',
            'practitioner': 'Practitioner',
            'site': 'Site',
            'session_slot': 'Session',
            'type_display': 'Type',
            'booking_total': 'Bookings £',
            'transaction_total': 'Transaction £',
            'difference': 'Diff £',
            'status': 'Status',
            'is_reception': 'Reception',
            'terminal': 'Terminal'
        })
        
        st.dataframe(
            styled_df.style.apply(
                lambda row: ['background-color: #FF0000' if row.get('Reception', False) else '' for _ in row],
                axis=1
            ),
            use_container_width=True,
            height=400
        )
    
    # Export button
    if not filtered_df.empty:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"discrepancies_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# =============================================================================
# REVENUE BY PRACTITIONER DASHBOARD
# =============================================================================

def render_revenue_practitioner_dashboard():
    """Render revenue by practitioner dashboard"""
    st.title("💰 Revenue Dashboard - Practitioners")
    
    # Load data
    df = load_revenue_by_practitioner()
    
    if df.empty:
        st.warning("No revenue data found. Please check database connection.")
        return
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        months = ['All'] + sorted(df['year_month'].dropna().unique().tolist(), reverse=True)
        selected_month = st.selectbox("📅 Month", months, key="rev_pract_month")
    
    with col2:
        practitioners = ['All'] + sorted(df['practitioner'].dropna().unique().tolist())
        selected_practitioner = st.selectbox("👤 Practitioner", practitioners, key="rev_pract_pract")
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_month != 'All':
        filtered_df = filtered_df[filtered_df['year_month'] == selected_month]
    
    if selected_practitioner != 'All':
        filtered_df = filtered_df[filtered_df['practitioner'] == selected_practitioner]
    
    # Metrics
    st.markdown("---")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("💰 Total Revenue", f"£{filtered_df['total_revenue'].sum():,.2f}")
    
    with col2:
        st.metric("📅 Total Appointments", f"{filtered_df['total_appointments'].sum():,}")
    
    with col3:
        avg = filtered_df['total_revenue'].sum() / max(filtered_df['net_appointments'].sum(), 1)
        st.metric("📊 Avg per Appointment", f"£{avg:.2f}")
    
    with col4:
        st.metric("🗓️ Total Sessions", f"{filtered_df['sessions'].sum():,}")
    
    with col5:
        avg_occ = filtered_df['occupancy_rate'].mean()
        st.metric("📈 Avg Occupancy", f"{avg_occ:.1f}%")
    
    # Charts
    st.markdown("---")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("💰 Revenue by Practitioner")
        if not filtered_df.empty:
            pract_rev = filtered_df.groupby('practitioner')['total_revenue'].sum().sort_values(ascending=True)
            fig = px.bar(
                x=pract_rev.values,
                y=pract_rev.index,
                orientation='h',
                color=pract_rev.values,
                color_continuous_scale='Greens'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        st.subheader("📈 Occupancy by Practitioner")
        if not filtered_df.empty:
            pract_occ = filtered_df.groupby('practitioner')['occupancy_rate'].mean().sort_values(ascending=True)
            colors = ['#FF6B6B' if v < 30 else '#FFE66D' if v < 40 else '#4ECDC4' if v < 60 else '#2ECC71' for v in pract_occ.values]
            fig = px.bar(
                x=pract_occ.values,
                y=pract_occ.index,
                orientation='h',
                color=pract_occ.values,
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.markdown("---")
    st.subheader("📋 Practitioner Summary")
    
    # Aggregate for display
    summary_df = filtered_df.groupby('practitioner').agg({
        'total_appointments': 'sum',
        'net_appointments': 'sum',
        'total_revenue': 'sum',
        'total_cash': 'sum',
        'sessions': 'sum',
        'no_wax': 'sum',
        'total_hours': 'sum'
    }).reset_index()
    
    summary_df['avg_per_appt'] = (summary_df['total_revenue'] / summary_df['net_appointments'].replace(0, 1)).round(2)
    summary_df['avg_per_session'] = (summary_df['total_revenue'] / summary_df['sessions'].replace(0, 1)).round(2)
    summary_df['no_wax_ratio'] = ((summary_df['no_wax'] / summary_df['total_appointments'].replace(0, 1)) * 100).round(2)
    
    summary_df = summary_df.rename(columns={
        'practitioner': 'Practitioner',
        'total_appointments': 'Appointments',
        'total_revenue': 'Revenue',
        'total_cash': 'Cash',
        'avg_per_appt': 'Avg/Appt',
        'sessions': 'Sessions',
        'avg_per_session': 'Avg/Session',
        'no_wax': 'No Wax',
        'no_wax_ratio': 'No Wax %'
    })
    
    display_cols = ['Practitioner', 'Appointments', 'Revenue', 'Cash', 'Avg/Appt', 'Sessions', 'Avg/Session', 'No Wax', 'No Wax %']
    
    st.dataframe(
        summary_df[display_cols].sort_values('Revenue', ascending=False),
        use_container_width=True,
        height=400
    )
    
    # Export
    if not filtered_df.empty:
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"revenue_practitioners_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# =============================================================================
# REVENUE BY SITE DASHBOARD
# =============================================================================

def render_revenue_site_dashboard():
    """Render revenue by site dashboard"""
    st.title("🏥 Revenue Dashboard - Sites")
    
    # Load data
    df = load_revenue_by_site()
    
    if df.empty:
        st.warning("No revenue data found. Please check database connection.")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        months = ['All'] + sorted(df['year_month'].dropna().unique().tolist(), reverse=True)
        selected_month = st.selectbox("📅 Month", months, key="rev_site_month")
    
    with col2:
        sites = ['All'] + sorted(df['site'].dropna().unique().tolist())
        selected_site = st.selectbox("📍 Site", sites, key="rev_site_site")
    
    with col3:
        occ_filter = st.selectbox("📊 Occupancy Filter", ['All', '60%+ (Excellent)', '40-59% (Good)', '30-39% (Average)', 'Under 30% (Low)'])
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_month != 'All':
        filtered_df = filtered_df[filtered_df['year_month'] == selected_month]
    
    if selected_site != 'All':
        filtered_df = filtered_df[filtered_df['site'] == selected_site]
    
    if occ_filter != 'All':
        if occ_filter == '60%+ (Excellent)':
            filtered_df = filtered_df[filtered_df['occupancy_rate'] >= 60]
        elif occ_filter == '40-59% (Good)':
            filtered_df = filtered_df[(filtered_df['occupancy_rate'] >= 40) & (filtered_df['occupancy_rate'] < 60)]
        elif occ_filter == '30-39% (Average)':
            filtered_df = filtered_df[(filtered_df['occupancy_rate'] >= 30) & (filtered_df['occupancy_rate'] < 40)]
        else:
            filtered_df = filtered_df[filtered_df['occupancy_rate'] < 30]
    
    # Metrics
    st.markdown("---")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("💰 Total Revenue", f"£{filtered_df['total_revenue'].sum():,.2f}")
    
    with col2:
        st.metric("🏥 Total Sites", f"{filtered_df['site'].nunique()}")
    
    with col3:
        avg_occ = filtered_df['occupancy_rate'].mean()
        st.metric("📈 Avg Occupancy", f"{avg_occ:.1f}%")
    
    with col4:
        st.metric("⏱️ Total Hours", f"{filtered_df['total_hours'].sum():,.1f}")
    
    with col5:
        st.metric("📅 Total Appointments", f"{filtered_df['total_appointments'].sum():,}")
    
    # Charts
    st.markdown("---")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("💰 Revenue by Site")
        if not filtered_df.empty:
            site_rev = filtered_df.groupby('site')['total_revenue'].sum().sort_values(ascending=True).tail(15)
            fig = px.bar(
                x=site_rev.values,
                y=site_rev.index,
                orientation='h',
                color=site_rev.values,
                color_continuous_scale='Greens'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        st.subheader("📈 Occupancy by Site")
        if not filtered_df.empty:
            site_occ = filtered_df.groupby('site')['occupancy_rate'].mean().sort_values(ascending=True).tail(15)
            fig = px.bar(
                x=site_occ.values,
                y=site_occ.index,
                orientation='h',
                color=site_occ.values,
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Occupancy distribution
    st.subheader("📊 Occupancy Distribution")
    if not filtered_df.empty:
        site_occ = filtered_df.groupby('site')['occupancy_rate'].mean().reset_index()
        site_occ['category'] = site_occ['occupancy_rate'].apply(
            lambda x: '60%+ (Excellent)' if x >= 60 else '40-59% (Good)' if x >= 40 else '30-39% (Average)' if x >= 30 else 'Under 30% (Low)'
        )
        cat_counts = site_occ['category'].value_counts()
        fig = px.pie(
            values=cat_counts.values,
            names=cat_counts.index,
            color=cat_counts.index,
            color_discrete_map={
                '60%+ (Excellent)': '#2ECC71',
                '40-59% (Good)': '#F1C40F',
                '30-39% (Average)': '#E67E22',
                'Under 30% (Low)': '#E74C3C'
            }
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.markdown("---")
    st.subheader("📋 Site Summary")
    
    # Aggregate for display
    summary_df = filtered_df.groupby('site').agg({
        'total_appointments': 'sum',
        'net_appointments': 'sum',
        'total_revenue': 'sum',
        'total_cash': 'sum',
        'sessions': 'sum',
        'total_hours': 'sum',
        'no_wax': 'sum',
        'occupancy_rate': 'mean'
    }).reset_index()
    
    summary_df['avg_per_appt'] = (summary_df['total_revenue'] / summary_df['net_appointments'].replace(0, 1)).round(2)
    summary_df['bookings_per_hour'] = (summary_df['total_appointments'] / summary_df['total_hours'].replace(0, 1)).round(2)
    summary_df['occupancy_category'] = summary_df['occupancy_rate'].apply(
        lambda x: '60%+' if x >= 60 else '40-59%' if x >= 40 else '30-39%' if x >= 30 else '<30%'
    )
    
    summary_df = summary_df.rename(columns={
        'site': 'Site',
        'total_appointments': 'Appointments',
        'total_revenue': 'Revenue',
        'total_cash': 'Cash',
        'avg_per_appt': 'Avg/Appt',
        'sessions': 'Sessions',
        'total_hours': 'Hours',
        'bookings_per_hour': 'Book/Hr',
        'occupancy_rate': 'Occ %',
        'occupancy_category': 'Category'
    })
    
    display_cols = ['Site', 'Appointments', 'Revenue', 'Cash', 'Avg/Appt', 'Sessions', 'Hours', 'Book/Hr', 'Occ %', 'Category']
    
    st.dataframe(
        summary_df[display_cols].sort_values('Revenue', ascending=False).style.applymap(
            color_occupancy, subset=['Occ %']
        ),
        use_container_width=True,
        height=400
    )
    
    # Export
    if not filtered_df.empty:
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"revenue_sites_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# =============================================================================
# MONTHLY OVERVIEW DASHBOARD
# =============================================================================

def render_monthly_overview():
    """Render monthly overview dashboard"""
    st.title("📈 Monthly Overview")
    
    # Load data
    df = load_monthly_summary()
    
    if df.empty:
        st.warning("No monthly data found. Please check database connection.")
        return
    
    # Metrics for latest month
    if not df.empty:
        latest = df.iloc[0]
        
        st.markdown("### 📅 " + str(latest.get('month_display', 'Current Month')))
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("💰 Revenue", f"£{latest.get('total_revenue', 0):,.2f}")
        
        with col2:
            st.metric("📅 Appointments", f"{latest.get('total_appointments', 0):,}")
        
        with col3:
            st.metric("👥 Practitioners", f"{latest.get('total_practitioners', 0)}")
        
        with col4:
            st.metric("🏥 Sites", f"{latest.get('total_sites', 0)}")
        
        with col5:
            st.metric("⏱️ Total Hours", f"{latest.get('total_hours', 0):,.1f}")
    
    # Trend charts
    st.markdown("---")
    
    st.subheader("📈 Revenue Trend")
    if not df.empty:
        fig = px.line(
            df.sort_values('year_month'),
            x='month_display',
            y='total_revenue',
            markers=True
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Appointments Trend")
        if not df.empty:
            fig = px.bar(
                df.sort_values('year_month'),
                x='month_display',
                y='total_appointments',
                color='total_appointments',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⏱️ Hours Worked Trend")
        if not df.empty:
            fig = px.bar(
                df.sort_values('year_month'),
                x='month_display',
                y='total_hours',
                color='total_hours',
                color_continuous_scale='Greens'
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Monthly comparison table
    st.markdown("---")
    st.subheader("📋 Monthly Comparison")
    
    display_df = df.copy()
    display_df = display_df.rename(columns={
        'month_display': 'Month',
        'total_practitioners': 'Practitioners',
        'total_sites': 'Sites',
        'total_appointments': 'Appointments',
        'total_revenue': 'Revenue',
        'total_cash': 'Cash',
        'total_hours': 'Hours'
    })
    
    display_cols = ['Month', 'Practitioners', 'Sites', 'Appointments', 'Revenue', 'Cash', 'Hours']
    display_cols = [col for col in display_cols if col in display_df.columns]
    
    st.dataframe(
        display_df[display_cols],
        use_container_width=True,
        height=300
    )

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application"""
    
    # Sidebar
    page = render_sidebar()
    
    # Render selected page
    if page == "🔍 Discrepancies":
        render_discrepancy_dashboard()
    elif page == "💰 Revenue - Practitioners":
        render_revenue_practitioner_dashboard()
    elif page == "🏥 Revenue - Sites":
        render_revenue_site_dashboard()
    elif page == "📈 Monthly Overview":
        render_monthly_overview()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <small>Reconciliation Dashboard v1.0 | Last updated: {}</small>
        </div>
        """.format(datetime.now().strftime('%Y-%m-%d %H:%M')),
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()