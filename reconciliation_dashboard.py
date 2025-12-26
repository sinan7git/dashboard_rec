"""
Reconciliation Dashboard - Streamlit App v2
=============================================
Features:
- Discrepancy tracking with detail view
- Invoice & Transaction matching
- Mark as Resolved functionality (updates revenue_data)
- Archive/Resolved discrepancies
- Clinician ID Mapping management
- Revenue analytics
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
def get_engine():
    """Create database engine"""
    try:
        engine = create_engine(DATABASE_URL)
        return engine
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

def execute_query(query, params=None):
    """Execute a query and commit"""
    engine = get_engine()
    if engine:
        with engine.connect() as conn:
            if params:
                result = conn.execute(text(query), params)
            else:
                result = conn.execute(text(query))
            conn.commit()
            return result
    return None

def fetch_dataframe(query, params=None):
    """Fetch query results as DataFrame"""
    engine = get_engine()
    if engine:
        if params:
            return pd.read_sql(text(query), engine, params=params)
        return pd.read_sql(query, engine)
    return pd.DataFrame()

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_data(ttl=60)
def load_discrepancies(status_filter='all'):
    """Load discrepancy data from database"""
    query = """
    SELECT 
        id,
        reconciliation_date,
        TO_CHAR(reconciliation_date, 'YYYY-MM') as year_month,
        TO_CHAR(reconciliation_date, 'DD/MM/YYYY') as date_display,
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
        COALESCE(resolution_notes, '') as resolution_notes,
        COALESCE(is_reception, false) as is_reception,
        created_at,
        resolved_at,
        resolved_by
    FROM reconciliation_discrepancies
    """
    
    if status_filter == 'pending':
        query += " WHERE status = 'pending'"
    elif status_filter == 'resolved':
        query += " WHERE status = 'resolved'"
    
    query += " ORDER BY reconciliation_date DESC, practitioner"
    
    return fetch_dataframe(query)

def load_invoices_for_discrepancy(practitioner, date, site):
    """Load invoices matching a discrepancy"""
    query = """
    SELECT 
        id,
        invoice_number,
        practitioner,
        clinic,
        total_amount,
        payment_taken,
        invoice_date,
        service_provided,
        contact_name,
        contact_email,
        source,
        payment_method,
        invoice_link
    FROM clearearwax_finance_invoice_data
    WHERE invoice_date::date = :date
      AND payment_method = 'card'
      AND (
        LOWER(SPLIT_PART(practitioner, ' ', 1)) = LOWER(:practitioner)
        OR LOWER(practitioner) = LOWER(:practitioner)
      )
    ORDER BY invoice_date
    """
    return fetch_dataframe(query, {
        'date': date, 
        'practitioner': practitioner.split()[0].lower() if practitioner else ''
    })

def load_transactions_for_practitioner(practitioner, date):
    """Load transaction data for practitioner on date"""
    query = """
    SELECT 
        id,
        practitioner,
        appointment_date,
        site,
        session_slot,
        appointments,
        revenue,
        card,
        cash,
        is_reception
    FROM revenue_data
    WHERE appointment_date = :date
      AND (
        LOWER(SPLIT_PART(practitioner, ' ', 1)) = LOWER(:practitioner)
        OR LOWER(practitioner) = LOWER(:practitioner)
      )
    ORDER BY session_slot
    """
    return fetch_dataframe(query, {
        'date': date, 
        'practitioner': practitioner.split()[0].lower() if practitioner else ''
    })


@st.cache_data(ttl=60)
def load_clinician_mappings():
    """Load clinician ID mappings"""
    query = """
    SELECT id, crm_id, practitioner_name, created_at, created_by
    FROM practitioner_id_mapping
    ORDER BY practitioner_name
    """
    try:
        return fetch_dataframe(query)
    except:
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
    return fetch_dataframe(query)

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
    return fetch_dataframe(query)

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
    return fetch_dataframe(query)

# =============================================================================
# DATABASE UPDATE FUNCTIONS
# =============================================================================

def mark_discrepancy_resolved(discrepancy_id, resolved_by, resolution_notes='', update_revenue=True):
    """Mark discrepancy as resolved and optionally update revenue_data"""
    
    # Convert numpy.int64 to Python int
    discrepancy_id = int(discrepancy_id)
    
    # First, get the discrepancy details
    disc_query = """
    SELECT * FROM reconciliation_discrepancies WHERE id = :id
    """
    disc_df = fetch_dataframe(disc_query, {'id': discrepancy_id})
    
    if disc_df.empty:
        return False
    
    disc = disc_df.iloc[0]
    
    # Update the discrepancy status
    update_disc_query = """
    UPDATE reconciliation_discrepancies 
    SET status = 'resolved',
        resolved_at = NOW(),
        resolved_by = :resolved_by,
        resolution_notes = :notes
    WHERE id = :id
    """
    execute_query(update_disc_query, {
        'id': discrepancy_id, 
        'resolved_by': resolved_by,
        'notes': resolution_notes
    })
    
    # Optionally update revenue_data to reconcile the amounts
    if update_revenue and disc['type'] == 'extra_in_bank':
        # Extra in bank - update card to match actual transaction amount
        update_revenue_query = """
        UPDATE revenue_data 
        SET card = :new_card,
            revenue = :new_card + cash
        WHERE LOWER(SPLIT_PART(practitioner, ' ', 1)) = LOWER(SPLIT_PART(:practitioner, ' ', 1))
          AND appointment_date = :date
          AND LOWER(site) = LOWER(:site)
          AND (
            session_slot = :session_slot 
            OR (:session_slot = '' AND (session_slot IS NULL OR session_slot = ''))
            OR session_slot IS NULL
          )
        """
        
        execute_query(update_revenue_query, {
            'new_card': float(disc['transaction_total']),
            'practitioner': str(disc['practitioner'] or ''),
            'date': disc['reconciliation_date'],
            'site': str(disc['site'] or ''),
            'session_slot': str(disc['session_slot'] or '')
        })
    
    # Clear cache to refresh data
    st.cache_data.clear()
    return True


def mark_discrepancy_pending(discrepancy_id):
    """Reopen a resolved discrepancy"""
    discrepancy_id = int(discrepancy_id)  # Add this line
    
    query = """
    UPDATE reconciliation_discrepancies 
    SET status = 'pending',
        resolved_at = NULL,
        resolved_by = NULL,
        resolution_notes = NULL
    WHERE id = :id
    """
    execute_query(query, {'id': discrepancy_id})
    st.cache_data.clear()

def add_clinician_mapping(crm_id, practitioner_name, created_by='Admin'):
    """Add a new clinician ID mapping"""
    query = """
    INSERT INTO practitioner_id_mapping (crm_id, practitioner_name, created_by)
    VALUES (:crm_id, :practitioner_name, :created_by)
    ON CONFLICT (crm_id) DO UPDATE SET 
        practitioner_name = EXCLUDED.practitioner_name,
        created_by = EXCLUDED.created_by
    """
    execute_query(query, {
        'crm_id': crm_id, 
        'practitioner_name': practitioner_name,
        'created_by': created_by
    })
    st.cache_data.clear()

def delete_clinician_mapping(mapping_id):
    """Delete a clinician mapping"""
    mapping_id = int(mapping_id)  # Add this line
    
    query = "DELETE FROM practitioner_id_mapping WHERE id = :id"
    execute_query(query, {'id': mapping_id})
    st.cache_data.clear()

def update_booking_practitioner_names():
    """Update booking data with mapped practitioner names"""
    query = """
    UPDATE clearearwax_finance_invoice_data f
    SET practitioner = m.practitioner_name
    FROM practitioner_id_mapping m
    WHERE f.practitioner = m.crm_id
      OR f.clinician_name = m.crm_id
    """
    result = execute_query(query)
    st.cache_data.clear()
    return result

# =============================================================================
# STYLING FUNCTIONS
# =============================================================================

def color_occupancy(val):
    """Color code occupancy rate"""
    if pd.isna(val):
        return ''
    if val >= 60:
        return 'background-color: #008000'
    elif val >= 40:
        return 'background-color: #FFAA33'
    elif val >= 30:
        return 'background-color: #FFA500'
    else:
        return 'background-color: #FF0000'

# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render sidebar with navigation and filters"""
    st.sidebar.title("📊 Dashboard")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigate",
        [
            "🔍 Pending Discrepancies",
            "✅ Resolved Discrepancies",
            "👤 Clinician Mapping",
            "💰 Revenue - Practitioners", 
            "🏥 Revenue - Sites", 
            "📈 Monthly Overview"
        ]
    )
    
    st.sidebar.markdown("---")
    
    # Quick stats
    pending_df = load_discrepancies('pending')
    resolved_df = load_discrepancies('resolved')
    
    st.sidebar.subheader("📊 Quick Stats")
    st.sidebar.metric("⏳ Pending", len(pending_df))
    st.sidebar.metric("✅ Resolved", len(resolved_df))
    
    if len(pending_df) > 0:
        missing = pending_df[pending_df['discrepancy_type'] == 'missing_in_bank']['difference'].sum()
        extra = pending_df[pending_df['discrepancy_type'] == 'extra_in_bank']['difference'].sum()
        st.sidebar.metric("❌ Missing", f"£{missing:,.2f}")
        st.sidebar.metric("💰 Extra", f"£{extra:,.2f}")
    
    return page

# =============================================================================
# DISCREPANCY DETAIL COMPONENT
# =============================================================================

def render_discrepancy_detail(discrepancy):
    """Render detailed view of a discrepancy with resolution options"""
    
    # Info display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"**Practitioner:** {discrepancy['practitioner']}")
        st.markdown(f"**Date:** {discrepancy['date_display']}")
    
    with col2:
        st.markdown(f"**Site:** {discrepancy['site']}")
        st.markdown(f"**Session:** {discrepancy['session_slot'] or 'N/A'}")
    
    with col3:
        st.markdown(f"**Terminal:** {discrepancy['terminal']}")
        st.markdown(f"**Type:** {discrepancy['type_display']}")
    
    with col4:
        st.markdown(f"**Bookings:** £{discrepancy['booking_total']:.2f} ({int(discrepancy['booking_count'])})")
        st.markdown(f"**Transactions:** £{discrepancy['transaction_total']:.2f} ({int(discrepancy['transaction_count'])})")
        st.markdown(f"**Difference:** £{discrepancy['difference']:.2f}")
    
    # Load and display related invoices
    st.markdown("#### 📄 Related Invoices")
    invoices = load_invoices_for_discrepancy(
        discrepancy['practitioner'],
        discrepancy['reconciliation_date'],
        discrepancy['site']
    )
    
    if not invoices.empty:
        display_cols = ['invoice_number','clinic', 'contact_name', 'service_provided', 
                       'total_amount', 'payment_taken', 'payment_method', 'source']
        display_cols = [c for c in display_cols if c in invoices.columns]
        st.dataframe(invoices[display_cols], use_container_width=True, height=150)
    else:
        st.info("No invoices found. This might be due to clinician name mismatch (CRM ID vs Name).")
    
    # Load and display transactions
    st.markdown("#### 💳 Related Transactions (from revenue_data)")
    transactions = load_transactions_for_practitioner(
        discrepancy['practitioner'],
        discrepancy['reconciliation_date']
    )
    
    if not transactions.empty:
        display_cols = ['site', 'session_slot', 'appointments', 'revenue', 'card', 'cash']
        display_cols = [c for c in display_cols if c in transactions.columns]
        st.dataframe(transactions[display_cols], use_container_width=True, height=150)
    else:
        st.info("No transaction data found in revenue_data.")
    
    # Resolution form
    st.markdown("#### ✅ Resolve This Discrepancy")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        resolution_notes = st.text_area(
            "Resolution Notes",
            placeholder="Enter notes about how this was resolved...",
            key=f"notes_{discrepancy['id']}"
        )
    
    with col2:
        resolved_by = st.text_input(
            "Resolved By",
            placeholder="Your name",
            key=f"resolver_{discrepancy['id']}"
        )
    
    # Resolution type options
    st.markdown("**Resolution Type:**")
    resolution_type = st.radio(
        "How to resolve this discrepancy?",
        [
            "✅ Accept & Update Revenue Data",
            "📝 Mark Reviewed Only (No Data Change)",
            "🗑️ Write Off (No Action Needed)"
        ],
        key=f"res_type_{discrepancy['id']}",
        horizontal=True
    )
    
    # Explain what will happen
    if discrepancy['discrepancy_type'] == 'missing_in_bank':
        st.caption(f"""
        **Missing from Bank:** Bookings show £{discrepancy['booking_total']:.2f} but only £{discrepancy['transaction_total']:.2f} in transactions.
        If you "Accept & Update", revenue_data.transaction_total will be updated to £{discrepancy['booking_total']:.2f}
        """)
    else:
        st.caption(f"""
        **Extra in Bank:** Transactions show £{discrepancy['transaction_total']:.2f} but only £{discrepancy['booking_total']:.2f} in bookings.
        If you "Accept & Update", revenue_data.card will be updated to £{discrepancy['transaction_total']:.2f}
        """)
    
    # Resolve button
    if st.button("✅ Resolve Discrepancy", key=f"resolve_{discrepancy['id']}", type="primary"):
        if resolved_by:
            update_revenue = "Accept & Update" in resolution_type
            full_notes = f"[{resolution_type}] {resolution_notes}"
            mark_discrepancy_resolved(
                discrepancy['id'], 
                resolved_by, 
                full_notes,
                update_revenue=update_revenue
            )
            st.success("✅ Discrepancy resolved!")
            st.rerun()
        else:
            st.error("Please enter your name")

# =============================================================================
# PAGE: PENDING DISCREPANCIES
# =============================================================================

def render_pending_discrepancies():
    """Render pending discrepancies page"""
    st.title("🔍 Pending Discrepancies")
    
    df = load_discrepancies('pending')
    
    if df.empty:
        st.success("🎉 No pending discrepancies! All reconciled.")
        return
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        months = ['All'] + sorted(df['year_month'].dropna().unique().tolist(), reverse=True)
        selected_month = st.selectbox("📅 Month", months, key="pend_month")
    
    with col2:
        practitioners = ['All'] + sorted(df['practitioner'].dropna().unique().tolist())
        selected_practitioner = st.selectbox("👤 Practitioner", practitioners, key="pend_pract")
    
    with col3:
        sites = ['All'] + sorted(df['site'].dropna().unique().tolist())
        selected_site = st.selectbox("📍 Site", sites, key="pend_site")
    
    with col4:
        types = ['All', 'Missing from Bank', 'Extra in Bank']
        selected_type = st.selectbox("🏷️ Type", types, key="pend_type")
    
    # Additional filters
    col5, col6 = st.columns(2)
    with col5:
        hide_reception = st.checkbox("🚫 Hide Reception Terminals", value=False)
    with col6:
        show_reception_only = st.checkbox("🏢 Show Reception Only", value=False)
    
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
    if hide_reception:
        filtered_df = filtered_df[filtered_df['is_reception'] == False]
    if show_reception_only:
        filtered_df = filtered_df[filtered_df['is_reception'] == True]
    
    # Metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total", len(filtered_df))
    with col2:
        missing = filtered_df[filtered_df['discrepancy_type'] == 'missing_in_bank']['difference'].sum()
        st.metric("❌ Missing", f"£{missing:,.2f}")
    with col3:
        extra = filtered_df[filtered_df['discrepancy_type'] == 'extra_in_bank']['difference'].sum()
        st.metric("💰 Extra", f"£{extra:,.2f}")
    with col4:
        reception = len(filtered_df[filtered_df['is_reception'] == True])
        st.metric("🏢 Reception", reception)
    
    # Discrepancy list with expandable details
    # Discrepancy list - TABLE VIEW (faster)
    st.markdown("---")
    st.subheader("📋 Discrepancies")
    
    # Show table
    display_cols = ['id', 'date_display', 'practitioner', 'site', 'type_display', 'difference', 'is_reception']
    display_df = filtered_df[display_cols].copy()
    display_df.columns = ['ID', 'Date', 'Practitioner', 'Site', 'Type', 'Diff £', 'Reception']
    
    st.dataframe(display_df, use_container_width=True, height=300)
    
    # Select one to view details
    st.markdown("---")
    st.subheader("🔍 View & Resolve Discrepancy")
    
    disc_ids = filtered_df['id'].tolist()
    if disc_ids:
        selected_id = st.selectbox(
            "Select Discrepancy ID to view details:",
            options=disc_ids,
            format_func=lambda x: f"#{x} - {filtered_df[filtered_df['id']==x]['practitioner'].values[0]} | {filtered_df[filtered_df['id']==x]['site'].values[0]} | £{filtered_df[filtered_df['id']==x]['difference'].values[0]:.2f}"
        )
        
        if selected_id:
            selected_row = filtered_df[filtered_df['id'] == selected_id].iloc[0]
            render_discrepancy_detail(selected_row)
    
    with col3:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Export to CSV",
            data=csv,
            file_name=f"pending_discrepancies_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# =============================================================================
# PAGE: RESOLVED DISCREPANCIES
# =============================================================================

def render_resolved_discrepancies():
    """Render resolved discrepancies archive"""
    st.title("✅ Resolved Discrepancies (Archive)")
    
    df = load_discrepancies('resolved')
    
    if df.empty:
        st.info("No resolved discrepancies yet.")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        months = ['All'] + sorted(df['year_month'].dropna().unique().tolist(), reverse=True)
        selected_month = st.selectbox("📅 Month", months, key="res_month")
    
    with col2:
        practitioners = ['All'] + sorted(df['practitioner'].dropna().unique().tolist())
        selected_practitioner = st.selectbox("👤 Practitioner", practitioners, key="res_pract")
    
    with col3:
        resolvers = ['All'] + sorted(df['resolved_by'].dropna().unique().tolist())
        selected_resolver = st.selectbox("👤 Resolved By", resolvers, key="res_resolver")
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_month != 'All':
        filtered_df = filtered_df[filtered_df['year_month'] == selected_month]
    if selected_practitioner != 'All':
        filtered_df = filtered_df[filtered_df['practitioner'] == selected_practitioner]
    if selected_resolver != 'All':
        filtered_df = filtered_df[filtered_df['resolved_by'] == selected_resolver]
    
    # Metrics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("✅ Total Resolved", len(filtered_df))
    with col2:
        total_amount = filtered_df['difference'].sum()
        st.metric("💰 Total Amount", f"£{total_amount:,.2f}")
    with col3:
        unique_resolvers = filtered_df['resolved_by'].nunique()
        st.metric("👥 Resolved By", f"{unique_resolvers} people")
    
    # Display table
    st.markdown("---")
    
    display_cols = ['date_display', 'practitioner', 'site', 'type_display',
                    'difference', 'resolved_by', 'resolution_notes']
    display_cols = [c for c in display_cols if c in filtered_df.columns]
    
    st.dataframe(
        filtered_df[display_cols].rename(columns={
            'date_display': 'Date',
            'practitioner': 'Practitioner',
            'site': 'Site',
            'type_display': 'Type',
            'difference': 'Amount',
            'resolved_by': 'Resolved By',
            'resolution_notes': 'Notes'
        }),
        use_container_width=True,
        height=400
    )
    
    # Reopen functionality
    st.markdown("---")
    st.subheader("🔄 Reopen Discrepancy")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        reopen_id = st.number_input("Discrepancy ID", min_value=1, step=1, key="reopen_id")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reopen", type="secondary"):
            mark_discrepancy_pending(reopen_id)
            st.success(f"✅ Discrepancy #{reopen_id} reopened")
            st.rerun()
    
    # Export
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        "📥 Export Archive to CSV",
        data=csv,
        file_name=f"resolved_discrepancies_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# =============================================================================
# PAGE: CLINICIAN MAPPING
# =============================================================================

def render_clinician_mapping():
    """Render clinician ID mapping management page"""
    st.title("👤 Clinician ID Mapping")
    
    st.markdown("""
    Map CRM IDs (like `CPinuJuGPgGIkDDKsvnV`) to actual practitioner names.
    This fixes reconciliation when the booking system uses internal IDs instead of names.
    """)
    
    # Add new mapping form
    st.markdown("---")
    st.subheader("➕ Add New Mapping")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        new_crm_id = st.text_input("CRM ID", placeholder="e.g., CPinuJuGPgGIkDDKsvnV")
    
    with col2:
        new_name = st.text_input("Practitioner Name", placeholder="e.g., Mugunth")
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Add", type="primary"):
            if new_crm_id and new_name:
                add_clinician_mapping(new_crm_id.strip(), new_name.strip())
                st.success(f"✅ Added: {new_crm_id} → {new_name}")
                st.rerun()
            else:
                st.error("Please fill in both fields")
    
    # Current mappings
    st.markdown("---")
    st.subheader("📋 Current Mappings")
    
    mappings = load_clinician_mappings()
    
    if mappings.empty:
        st.info("No mappings configured yet. Add your first mapping above.")
    else:
        for idx, row in mappings.iterrows():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                st.code(row['crm_id'], language=None)
            with col2:
                st.markdown(f"**→ {row['practitioner_name']}**")
            with col3:
                if pd.notna(row.get('created_at')):
                    st.caption(f"Added: {row['created_at']}")
            with col4:
                if st.button("🗑️", key=f"del_{row['id']}"):
                    delete_clinician_mapping(row['id'])
                    st.rerun()
    
    # Apply mappings
    st.markdown("---")
    st.subheader("🔄 Apply Mappings to Booking Data")
    
    st.warning("""
    This will update the booking data table to replace CRM IDs with actual practitioner names.
    Run this after adding new mappings, then re-run the reconciliation workflow.
    """)
    
    if st.button("🔄 Apply All Mappings", type="primary"):
        update_booking_practitioner_names()
        st.success("✅ Mappings applied! Now re-run the reconciliation workflow.")
    
    # Find unmapped IDs
    st.markdown("---")
    st.subheader("🔍 Find Unmapped CRM IDs")
    
    if st.button("🔍 Scan Booking Data"):
        query = """
        SELECT DISTINCT practitioner, COUNT(*) as count
        FROM clearearwax_finance_invoice_data
        WHERE LENGTH(practitioner) > 15
          AND practitioner ~ '^[a-zA-Z0-9]+$'
          AND practitioner NOT IN (SELECT crm_id FROM practitioner_id_mapping)
        GROUP BY practitioner
        ORDER BY count DESC
        LIMIT 20
        """
        try:
            unmapped = fetch_dataframe(query)
            
            if unmapped.empty:
                st.success("✅ All CRM IDs are mapped!")
            else:
                st.warning(f"Found {len(unmapped)} unmapped CRM IDs:")
                st.dataframe(unmapped, use_container_width=True)
        except Exception as e:
            st.error(f"Error scanning: {e}")

# =============================================================================
# PAGE: REVENUE BY PRACTITIONER
# =============================================================================

def render_revenue_practitioner_dashboard():
    """Render revenue by practitioner dashboard"""
    st.title("💰 Revenue Dashboard - Practitioners")
    
    df = load_revenue_by_practitioner()
    
    if df.empty:
        st.warning("No revenue data found.")
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
        st.metric("📅 Appointments", f"{filtered_df['total_appointments'].sum():,}")
    with col3:
        avg = filtered_df['total_revenue'].sum() / max(filtered_df['net_appointments'].sum(), 1)
        st.metric("📊 Avg/Appt", f"£{avg:.2f}")
    with col4:
        st.metric("🗓️ Sessions", f"{filtered_df['sessions'].sum():,}")
    with col5:
        avg_occ = filtered_df['occupancy_rate'].mean()
        st.metric("📈 Occupancy", f"{avg_occ:.1f}%")
    
    # Charts
    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("💰 Revenue by Practitioner")
        if not filtered_df.empty:
            pract_rev = filtered_df.groupby('practitioner')['total_revenue'].sum().sort_values(ascending=True)
            fig = px.bar(x=pract_rev.values, y=pract_rev.index, orientation='h',
                        color=pract_rev.values, color_continuous_scale='Greens')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        st.subheader("📈 Occupancy by Practitioner")
        if not filtered_df.empty:
            pract_occ = filtered_df.groupby('practitioner')['occupancy_rate'].mean().sort_values(ascending=True)
            fig = px.bar(x=pract_occ.values, y=pract_occ.index, orientation='h',
                        color=pract_occ.values, color_continuous_scale='RdYlGn')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.markdown("---")
    st.subheader("📋 Practitioner Summary")
    
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
    summary_df['no_wax_ratio'] = ((summary_df['no_wax'] / summary_df['total_appointments'].replace(0, 1)) * 100).round(2)
    
    st.dataframe(
        summary_df.rename(columns={
            'practitioner': 'Practitioner', 'total_appointments': 'Appointments',
            'total_revenue': 'Revenue', 'total_cash': 'Cash', 'avg_per_appt': 'Avg/Appt',
            'sessions': 'Sessions', 'no_wax': 'No Wax', 'no_wax_ratio': 'No Wax %'
        }).sort_values('Revenue', ascending=False),
        use_container_width=True, height=400
    )
    
    csv = summary_df.to_csv(index=False)
    st.download_button("📥 Download CSV", data=csv,
                      file_name=f"revenue_practitioners_{datetime.now().strftime('%Y%m%d')}.csv",
                      mime="text/csv")

# =============================================================================
# PAGE: REVENUE BY SITE
# =============================================================================

def render_revenue_site_dashboard():
    """Render revenue by site dashboard"""
    st.title("🏥 Revenue Dashboard - Sites")
    
    df = load_revenue_by_site()
    
    if df.empty:
        st.warning("No revenue data found.")
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
        occ_filter = st.selectbox("📊 Occupancy", ['All', '60%+', '40-59%', '30-39%', '<30%'])
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_month != 'All':
        filtered_df = filtered_df[filtered_df['year_month'] == selected_month]
    if selected_site != 'All':
        filtered_df = filtered_df[filtered_df['site'] == selected_site]
    if occ_filter != 'All':
        if occ_filter == '60%+':
            filtered_df = filtered_df[filtered_df['occupancy_rate'] >= 60]
        elif occ_filter == '40-59%':
            filtered_df = filtered_df[(filtered_df['occupancy_rate'] >= 40) & (filtered_df['occupancy_rate'] < 60)]
        elif occ_filter == '30-39%':
            filtered_df = filtered_df[(filtered_df['occupancy_rate'] >= 30) & (filtered_df['occupancy_rate'] < 40)]
        else:
            filtered_df = filtered_df[filtered_df['occupancy_rate'] < 30]
    
    # Metrics
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("💰 Revenue", f"£{filtered_df['total_revenue'].sum():,.2f}")
    with col2:
        st.metric("🏥 Sites", f"{filtered_df['site'].nunique()}")
    with col3:
        avg_occ = filtered_df['occupancy_rate'].mean()
        st.metric("📈 Occupancy", f"{avg_occ:.1f}%")
    with col4:
        st.metric("⏱️ Hours", f"{filtered_df['total_hours'].sum():,.1f}")
    with col5:
        st.metric("📅 Appointments", f"{filtered_df['total_appointments'].sum():,}")
    
    # Charts
    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("💰 Revenue by Site")
        if not filtered_df.empty:
            site_rev = filtered_df.groupby('site')['total_revenue'].sum().sort_values(ascending=True).tail(15)
            fig = px.bar(x=site_rev.values, y=site_rev.index, orientation='h',
                        color=site_rev.values, color_continuous_scale='Greens')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        st.subheader("📈 Occupancy by Site")
        if not filtered_df.empty:
            site_occ = filtered_df.groupby('site')['occupancy_rate'].mean().sort_values(ascending=True).tail(15)
            fig = px.bar(x=site_occ.values, y=site_occ.index, orientation='h',
                        color=site_occ.values, color_continuous_scale='RdYlGn')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.markdown("---")
    st.subheader("📋 Site Summary")
    
    summary_df = filtered_df.groupby('site').agg({
        'total_appointments': 'sum', 'total_revenue': 'sum', 'total_cash': 'sum',
        'sessions': 'sum', 'total_hours': 'sum', 'occupancy_rate': 'mean'
    }).reset_index()
    
    summary_df['avg_per_appt'] = (summary_df['total_revenue'] / summary_df['total_appointments'].replace(0, 1)).round(2)
    
    st.dataframe(
        summary_df.rename(columns={
            'site': 'Site', 'total_appointments': 'Appointments', 'total_revenue': 'Revenue',
            'total_cash': 'Cash', 'avg_per_appt': 'Avg/Appt', 'sessions': 'Sessions',
            'total_hours': 'Hours', 'occupancy_rate': 'Occ %'
        }).sort_values('Revenue', ascending=False).style.applymap(color_occupancy, subset=['Occ %']),
        use_container_width=True, height=400
    )
    
    csv = summary_df.to_csv(index=False)
    st.download_button("📥 Download CSV", data=csv,
                      file_name=f"revenue_sites_{datetime.now().strftime('%Y%m%d')}.csv",
                      mime="text/csv")

# =============================================================================
# PAGE: MONTHLY OVERVIEW
# =============================================================================

def render_monthly_overview():
    """Render monthly overview dashboard"""
    st.title("📈 Monthly Overview")
    
    df = load_monthly_summary()
    
    if df.empty:
        st.warning("No monthly data found.")
        return
    
    # Latest month metrics
    if not df.empty:
        latest = df.iloc[0]
        
        st.markdown(f"### 📅 {latest.get('month_display', 'Current Month')}")
        
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
            st.metric("⏱️ Hours", f"{latest.get('total_hours', 0):,.1f}")
    
    # Trend charts
    st.markdown("---")
    st.subheader("📈 Revenue Trend")
    if not df.empty:
        fig = px.line(df.sort_values('year_month'), x='month_display', y='total_revenue', markers=True)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Appointments Trend")
        if not df.empty:
            fig = px.bar(df.sort_values('year_month'), x='month_display', y='total_appointments',
                        color='total_appointments', color_continuous_scale='Blues')
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⏱️ Hours Trend")
        if not df.empty:
            fig = px.bar(df.sort_values('year_month'), x='month_display', y='total_hours',
                        color='total_hours', color_continuous_scale='Greens')
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Monthly table
    st.markdown("---")
    st.subheader("📋 Monthly Comparison")
    
    st.dataframe(
        df.rename(columns={
            'month_display': 'Month', 'total_practitioners': 'Practitioners',
            'total_sites': 'Sites', 'total_appointments': 'Appointments',
            'total_revenue': 'Revenue', 'total_cash': 'Cash', 'total_hours': 'Hours'
        })[['Month', 'Practitioners', 'Sites', 'Appointments', 'Revenue', 'Cash', 'Hours']],
        use_container_width=True, height=300
    )

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application entry point"""
    
    # Sidebar navigation
    page = render_sidebar()
    
    # Route to page
    if page == "🔍 Pending Discrepancies":
        render_pending_discrepancies()
    elif page == "✅ Resolved Discrepancies":
        render_resolved_discrepancies()
    elif page == "👤 Clinician Mapping":
        render_clinician_mapping()
    elif page == "💰 Revenue - Practitioners":
        render_revenue_practitioner_dashboard()
    elif page == "🏥 Revenue - Sites":
        render_revenue_site_dashboard()
    elif page == "📈 Monthly Overview":
        render_monthly_overview()
    
    # Footer
    st.markdown("---")
    st.caption(f"Reconciliation Dashboard v2.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")

if __name__ == "__main__":
    main()