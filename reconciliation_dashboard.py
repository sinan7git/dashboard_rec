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
        clinic as site,
        total_amount,
        payment_taken,
        invoice_date,
        created_at,  
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
    ORDER BY created_at  -- Changed from invoice_date
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

def load_bank_transactions_for_discrepancy(terminal, date):
    """Load raw bank transactions for a specific terminal and date"""
    query = """
    SELECT 
        TO_CHAR(trans_local_time, 'HH24:MI:SS') as time,
        trans_amount as amount,
        result as status,
        card_type,
        brand,
        CONCAT('****', masked_pan) as card_number
    FROM merchant_transactions
    WHERE terminal_sn = :terminal
      AND trans_local_time::date = :date
    ORDER BY trans_local_time ASC
    """
    return fetch_dataframe(query, {'terminal': str(terminal), 'date': date})


@st.cache_data(ttl=60)
def load_bank_feed_data(start_date, end_date):
    """Load raw transactions from the merchant_transactions table"""
    query = """
    SELECT 
        id,
        trans_local_time as "Time",
        terminal_sn as "Terminal",
        trans_amount as "Amount",
        result as "Status",
        card_type as "Card Type",
        brand as "Brand",
        masked_pan as "Card Last 4",
        merchant as "Merchant",
        created_at as "Ingested At"
    FROM merchant_transactions
    WHERE trans_local_time::date BETWEEN :start_date AND :end_date
    ORDER BY trans_local_time DESC
    """
    return fetch_dataframe(query, {'start_date': start_date, 'end_date': end_date})


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


def lookup_practitioner_info(terminal_sn, date_obj):
    """
    Finds the practitioner assigned to a terminal (using terminal_assignments)
    and their scheduled site (using practitioner_schedule) for context.
    """
    # 1. Get Name from Terminal Assignment
    # We look for the assignment that was active on or matches the terminal SN
    query_assign = """
    SELECT practitioner_name 
    FROM terminal_assignments 
    WHERE terminal_sn = :terminal
    LIMIT 1
    """
    # Note: We cast terminal to string to ensure matching works against varchar column
    df_assign = fetch_dataframe(query_assign, {'terminal': str(terminal_sn)})
    
    if df_assign.empty:
        return None, None
        
    practitioner_name = df_assign.iloc[0]['practitioner_name']
    
    # 2. Get Site from Schedule (For context/verification)
    query_schedule = """
    SELECT site 
    FROM practitioner_schedule
    WHERE practitioner = :name 
      AND schedule_date = :date
    LIMIT 1
    """
    df_schedule = fetch_dataframe(query_schedule, {'name': practitioner_name, 'date': date_obj})
    
    site_name = "Unknown Site"
    if not df_schedule.empty:
        site_name = df_schedule.iloc[0]['site']
        
    return practitioner_name, site_name

def load_system_invoices(practitioner_name, date_obj):
    """
    Fetch raw invoice rows from the database for comparison.
    Matches practitioner name loosely (first name or full name) to catch all variations.
    """
    query = """
    SELECT 
        invoice_number AS "Invoice #",
        TO_CHAR(created_at, 'HH24:MI') AS "Time Logged",
        contact_name AS "Patient",
        service_provided AS "Service",
        payment_method AS "Method",
        payment_taken AS "Paid £",
        clinic AS "Clinic",
        is_deposit AS "Deposit?"
    FROM clearearwax_finance_invoice_data
    WHERE invoice_date = :date
      AND (
        LOWER(practitioner) = LOWER(:name)
        OR LOWER(SPLIT_PART(practitioner, ' ', 1)) = LOWER(SPLIT_PART(:name, ' ', 1))
        OR LOWER(clinician_name) = LOWER(:name)
      )
    ORDER BY created_at ASC
    """
    return fetch_dataframe(query, {'date': date_obj, 'name': practitioner_name})

def render_manual_audit():
    st.title("🕵️ Manual Audit Tool")
    st.markdown("""
    **Compare the 'Bank' vs the 'System'.**
    Upload a transaction CSV to see exactly what money arrived and match it against the invoices logged in the system.
    """)

    # --- STEP 1: UPLOAD BANK FILE ---
    st.subheader("1. Upload Bank Data")
    uploaded_file = st.file_uploader("📂 Upload Transaction CSV/Excel", type=["csv", "xlsx"])

    if not uploaded_file:
        st.info("👆 Please upload a file to begin the audit.")
        return

    # Load Data
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file, dtype=str)
        else:
            df = pd.read_excel(uploaded_file, dtype=str)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    # --- STEP 2: MAP COLUMNS ---
    with st.expander("⚙️ Column Mapping (Click to Edit)", expanded=False):
        c1, c2, c3 = st.columns(3)
        cols = df.columns.tolist()
        
        # Smart defaults
        date_idx = next((i for i, c in enumerate(cols) if 'TIME' in c.upper() or 'DATE' in c.upper()), 0)
        term_idx = next((i for i, c in enumerate(cols) if 'TERMINAL' in c.upper() or 'SN' in c.upper()), 0)
        amt_idx = next((i for i, c in enumerate(cols) if 'AMOUNT' in c.upper()), 0)

        col_date = c1.selectbox("📅 Event Time/Date", cols, index=date_idx)
        col_term = c2.selectbox("📟 Terminal SN", cols, index=term_idx)
        col_amt = c3.selectbox("💷 Amount", cols, index=amt_idx)

    # Process Data
    df[col_date] = pd.to_datetime(df[col_date], errors='coerce')
    df[col_amt] = pd.to_numeric(df[col_amt], errors='coerce')

    # --- STEP 3: FILTER VIEW ---
    st.divider()
    st.subheader("2. Select Day & Terminal")
    f1, f2 = st.columns(2)
    
    # Date Filter
    unique_dates = sorted(df[col_date].dt.date.dropna().unique(), reverse=True)
    selected_date = f1.selectbox("📅 Select Date", unique_dates)
    
    # Terminal Filter (Dependent on Date)
    day_df = df[df[col_date].dt.date == selected_date]
    unique_terminals = sorted(day_df[col_term].dropna().unique())
    
    if not unique_terminals:
        st.warning("No transactions found for this date.")
        return
        
    selected_terminal = f2.selectbox("📟 Select Terminal", unique_terminals)

    # Filter DataFrame for Terminal
    audit_df = day_df[day_df[col_term] == selected_terminal].sort_values(col_date)

    # --- STEP 4: FIND THE DOCTOR ---
    st.divider()
    
    practitioner_name, site_name = lookup_practitioner_info(selected_terminal, selected_date)
    
    if practitioner_name:
        st.success(f"✅ **Identity Match:** Terminal `{selected_terminal}` belongs to **{practitioner_name}** ({site_name})")
    else:
        st.warning(f"⚠️ **Unknown Terminal:** Could not find assignment for `{selected_terminal}` in database.")
        practitioner_name = st.text_input("Manually enter Practitioner Name to force check:")

    # --- STEP 5: THE SIDE-BY-SIDE AUDIT ---
    st.subheader("3. Side-by-Side Comparison")
    
    col_left, col_right = st.columns(2)

    # 👈 LEFT SIDE: RAW BANK DATA
    with col_left:
        st.markdown(f"### 🏦 Bank CSV (Left Side)")
        st.caption(f"Raw transactions from your uploaded file")
        
        bank_total = audit_df[col_amt].sum()
        bank_count = len(audit_df)
        
        st.metric("Total Money In", f"£{bank_total:,.2f}", f"{bank_count} Transactions")
        
        # Display simplified table
        display_cols = [col_date, col_amt]
        # Add 'RESULT' if it exists to show Approved/Declined
        result_col = next((c for c in df.columns if 'RESULT' in c.upper()), None)
        if result_col: display_cols.append(result_col)
        
        # Format time for display
        view_df = audit_df[display_cols].copy()
        view_df[col_date] = view_df[col_date].dt.strftime('%H:%M:%S')
        
        st.dataframe(view_df, use_container_width=True, hide_index=True)

    # 👉 RIGHT SIDE: SYSTEM DATA
    with col_right:
        st.markdown(f"### 💻 System Data (Right Side)")
        
        if practitioner_name:
            st.caption(f"Invoices logged in database for **{practitioner_name}**")
            
            # Fetch Data
            sys_df = load_system_invoices(practitioner_name, selected_date)
            
            if not sys_df.empty:
                # Calculate Card Total (To compare apples to apples)
                # We look for 'card' in payment method column
                card_df = sys_df[sys_df['Method'].astype(str).str.lower().str.contains('card', na=False)]
                sys_card_total = card_df['Paid £'].sum()
                sys_count = len(card_df)
                
                # Difference Calculation
                diff = bank_total - sys_card_total
                
                if diff == 0:
                    delta_color = "normal"
                    delta_msg = "✅ Perfect Match"
                elif diff > 0:
                    delta_color = "inverse" # Red
                    delta_msg = f"💰 Bank has £{diff:.2f} MORE (Extra)"
                else:
                    delta_color = "inverse" # Red
                    delta_msg = f"❌ Bank has £{abs(diff):.2f} LESS (Missing)"

                st.metric("Expected (Card Only)", f"£{sys_card_total:,.2f}", delta_msg, delta_color=delta_color)
                
                # Display Table
                st.dataframe(
                    sys_df[['Time Logged', 'Patient', 'Method', 'Paid £', 'Invoice #']], 
                    use_container_width=True, 
                    hide_index=True
                )
            else:
                st.warning("No invoices found in the system for this doctor/date.")
                st.metric("Expected (Card Only)", "£0.00")
        else:
            st.info("Waiting for practitioner identification...")

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
def load_invoice_revenue_by_practitioner():
    """
    Load revenue from invoices grouped by practitioner using strict total_amount logic.
    Logic: If duplicate invoice_id_ref exists, prioritize the row where is_deposit=False
    to capture the single source of truth for total_amount.
    """
    query = """
    WITH RankedInvoices AS (
        SELECT 
            *,
            -- LOGIC: Partition by invoice ID. 
            -- Order by is_deposit ASC (False comes first).
            -- If an invoice has a Deposit row and a Normal row, the Normal row becomes rn=1.
            -- If it only has one row, that row becomes rn=1.
            ROW_NUMBER() OVER (
                PARTITION BY invoice_id_ref 
                ORDER BY is_deposit ASC, id DESC
            ) as rn
        FROM clearearwax_finance_invoice_data
        WHERE invoice_date >= CURRENT_DATE - INTERVAL '6 months' -- Adjust timeframe as needed
    )
    SELECT 
        practitioner,
        TO_CHAR(invoice_date, 'YYYY-MM') as year_month,
        TO_CHAR(invoice_date, 'Mon YYYY') as month_display,
        
        -- Count unique invoices (only count row 1s)
        COUNT(CASE WHEN rn = 1 THEN 1 END) as total_invoices,
        
        -- REVENUE CALCULATION:
        -- Only take total_amount from the #1 ranked row (Non-deposit preferred)
        SUM(CASE 
            WHEN rn = 1 THEN total_amount 
            ELSE 0 
        END) as total_revenue,
        
        -- Breakdown Metrics (Optional: keeps payment_taken for accurate cash/card split)
        SUM(CASE WHEN payment_method = 'card' THEN payment_taken ELSE 0 END) as card_payments,
        SUM(CASE WHEN payment_method = 'cash' THEN payment_taken ELSE 0 END) as cash_payments,
        
        -- Deposit Tracking
        SUM(CASE WHEN is_deposit = true THEN payment_taken ELSE 0 END) as deposit_amount,
        
        -- Source Counts (Unique Invoices)
        COUNT(CASE WHEN rn = 1 AND source = 'Clear Earwax' THEN 1 END) as clearearwax_count,
        COUNT(CASE WHEN rn = 1 AND source = 'Hearing Expert' THEN 1 END) as hearingexpert_count
        
    FROM RankedInvoices
    GROUP BY practitioner, TO_CHAR(invoice_date, 'YYYY-MM'), TO_CHAR(invoice_date, 'Mon YYYY')
    ORDER BY year_month DESC, total_revenue DESC
    """
    return fetch_dataframe(query)


@st.cache_data(ttl=300)
def load_invoice_revenue_by_site():
    """Load revenue from invoices grouped by clinic/site"""
    query = """
    SELECT 
        clinic as site,
        TO_CHAR(invoice_date, 'YYYY-MM') as year_month,
        TO_CHAR(invoice_date, 'Mon YYYY') as month_display,
        
        -- Count unique invoices
        COUNT(DISTINCT invoice_id_ref) as total_invoices,
        
        -- Total revenue (avoid duplicates)
        SUM(CASE 
            WHEN rn = 1 THEN total_amount 
            ELSE 0 
        END) as total_revenue,
        
        -- Payments by method
        SUM(CASE WHEN payment_method = 'card' THEN payment_taken ELSE 0 END) as card_payments,
        SUM(CASE WHEN payment_method = 'cash' THEN payment_taken ELSE 0 END) as cash_payments,
        SUM(CASE WHEN payment_method = 'other' THEN payment_taken ELSE 0 END) as other_payments,
        
        -- Deposit tracking
        SUM(CASE WHEN is_deposit THEN payment_taken ELSE 0 END) as deposit_amount,
        COUNT(DISTINCT CASE WHEN is_deposit THEN invoice_id_ref END) as deposit_count,
        
        -- Average
        CASE 
            WHEN COUNT(DISTINCT invoice_id_ref) > 0 
            THEN ROUND((SUM(CASE WHEN rn = 1 THEN total_amount ELSE 0 END)::numeric / COUNT(DISTINCT invoice_id_ref)), 2)
            ELSE 0
        END as avg_per_invoice,
        
        -- Practitioners working at this site
        COUNT(DISTINCT practitioner) as practitioner_count
        
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY invoice_id_ref ORDER BY id) as rn
        FROM clearearwax_finance_invoice_data
        WHERE invoice_date >= CURRENT_DATE - INTERVAL '90 days'
    ) ranked
    GROUP BY clinic, TO_CHAR(invoice_date, 'YYYY-MM'), TO_CHAR(invoice_date, 'Mon YYYY')
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

def fetch_all_clinic_names():
    """
    Fetch ALL distinct clinic names found in the database (no exclusions).
    """
    query = """
    SELECT DISTINCT clinic 
    FROM clearearwax_finance_invoice_data 
    WHERE clinic IS NOT NULL 
      AND clinic != ''
    ORDER BY clinic
    """
    df = fetch_dataframe(query)
    return df['clinic'].tolist() if not df.empty else []

def fetch_invoices_by_clinic_text(clinic_name):
    """
    Fetch invoices matching a specific text string in the clinic column.
    """
    query = """
    SELECT 
        id,
        invoice_number,
        invoice_date,
        contact_name,
        total_amount,
        clinic as current_clinic_name,
        practitioner
    FROM clearearwax_finance_invoice_data
    WHERE clinic = :clinic_name
    ORDER BY invoice_date DESC
    """
    return fetch_dataframe(query, {'clinic_name': clinic_name})

def bulk_update_clinic_name(invoice_ids, new_clinic_name):
    """
    Update the clinic name for a list of invoice IDs.
    """
    if not invoice_ids:
        return False
        
    # REMOVED 'updated_at = NOW()' because the column does not exist
    query = """
    UPDATE clearearwax_finance_invoice_data
    SET clinic = :new_name
    WHERE id = ANY(:ids)
    """
    
    # Pass IDs as a list
    result = execute_query(query, {'new_name': new_clinic_name, 'ids': list(invoice_ids)})
    st.cache_data.clear()
    return True

def mark_discrepancy_resolved(discrepancy_id, resolved_by, resolution_notes='', update_revenue=True, manual_site=None):
    """Mark discrepancy as resolved and optionally update revenue_data"""
    
    discrepancy_id = int(discrepancy_id)
    
    # Get discrepancy details
    disc_query = "SELECT * FROM reconciliation_discrepancies WHERE id = :id"
    disc_df = fetch_dataframe(disc_query, {'id': discrepancy_id})
    
    if disc_df.empty:
        return False
    
    disc = disc_df.iloc[0]
    
    # =====================================================================
    # STEP 0: Handle NULL/Unknown Site
    # =====================================================================
    site_value = disc['site']
    session_value = disc['session_slot']


    
    # If site is NULL/Unknown - MUST have manual selection
    site_is_invalid = not site_value or site_value.lower() in ['unknown', 'null', 'none', '', 'no schedule']
    session_is_invalid = not session_value or session_value.lower() in ['n/a', 'null', 'none', '']

    if site_is_invalid or session_is_invalid:
        if not manual_site:
            st.error("❌ **Site is missing or no session scheduled! Please select a site manually before resolving.**")
            return False
        else:
            site_value = manual_site
            st.info(f"📍 Using manually selected site: **{site_value}**")

    
    # =====================================================================
    # STEP 1: Get ACTUAL bank transaction total
    # =====================================================================
    if update_revenue:
        bank_query = """
        SELECT COALESCE(SUM(trans_amount), 0) as actual_bank_total
        FROM merchant_transactions
        WHERE terminal_sn = :terminal
          AND trans_local_time::date = :date
          AND result = 'APPROVED'
        """
        bank_df = fetch_dataframe(bank_query, {
            'terminal': str(disc['terminal']),
            'date': disc['reconciliation_date']
        })
        
        actual_bank_total = float(bank_df.iloc[0]['actual_bank_total']) if not bank_df.empty else 0
        st.info(f"🏦 Actual Bank Total: £{actual_bank_total:.2f}")
    
    # =====================================================================
    # STEP 2: UPSERT to revenue_data
    # =====================================================================
    if update_revenue:
        upsert_query = """
        INSERT INTO revenue_data (
            practitioner, appointment_date, site, session_slot,
            card, revenue, appointments, net_appointments, day_of_week, created_at
        )
        VALUES (
            :practitioner, :date, :site, :session_slot,
            :card_amount, :card_amount, 
            COALESCE(:booking_count, 0), COALESCE(:booking_count, 0),
            TO_CHAR(:date::date, 'Day'), NOW()
        )
        ON CONFLICT (practitioner, appointment_date, site, session_slot)
        DO UPDATE SET
            card = :card_amount,
            revenue = EXCLUDED.card + revenue_data.cash
        """
        
        execute_query(upsert_query, {
            'practitioner': str(disc['practitioner']),
            'date': disc['reconciliation_date'],
            'site': site_value,  # Manual selection or existing value
            'session_slot': str(disc['session_slot']) if disc['session_slot'] else '',
            'card_amount': actual_bank_total,
            'booking_count': int(disc['booking_count']) if pd.notna(disc['booking_count']) else 0
        })
        
        st.success(f"✅ Updated: {disc['practitioner']} | {site_value} | £{actual_bank_total:.2f}")
    
    # =====================================================================
    # STEP 3: Mark as resolved
    # =====================================================================
    update_disc_query = """
    UPDATE reconciliation_discrepancies 
    SET status = 'resolved', resolved_at = NOW(), resolved_by = :resolved_by, resolution_notes = :notes
    WHERE id = :id
    """
    execute_query(update_disc_query, {
        'id': discrepancy_id, 
        'resolved_by': resolved_by,
        'notes': resolution_notes
    })
    
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

def render_invoice_revenue_practitioner_dashboard():
    """Render invoice-based revenue by practitioner dashboard (Total Amount Logic)"""
    st.markdown("## 💰 Invoiced Revenue - Practitioner View")
    
    # Load the new data
    df = load_invoice_revenue_by_practitioner()
    
    if df.empty:
        st.warning("No invoice data found.")
        return
    
    # --- FILTERS ---
    col1, col2 = st.columns(2)
    
    with col1:
        months = ['All'] + sorted(df['year_month'].dropna().unique().tolist(), reverse=True)
        selected_month = st.selectbox("📅 Month", months, key="inv_pract_month_v2")
    
    with col2:
        practitioners = ['All'] + sorted(df['practitioner'].dropna().unique().tolist())
        selected_practitioner = st.selectbox("👤 Practitioner", practitioners, key="inv_pract_name_v2")
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_month != 'All':
        filtered_df = filtered_df[filtered_df['year_month'] == selected_month]
    if selected_practitioner != 'All':
        filtered_df = filtered_df[filtered_df['practitioner'] == selected_practitioner]
    
    # --- METRICS ---
    st.markdown("---")
    
    # Calculate totals
    total_rev = filtered_df['total_revenue'].sum()
    total_inv = filtered_df['total_invoices'].sum()
    
    m1,  m3,  = st.columns(2)
    
    with m1:
        st.metric("💰 Total Invoiced Revenue", f"£{total_rev:,.2f}", help="Sum of Total Amount (deduplicated)")

    with m3:
        avg = total_rev / max(total_inv, 1)
        st.metric("📊 Avg Invoice Value", f"£{avg:.2f}")

    
    # --- CHARTS ---
    st.markdown("---")
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("💰 Revenue by Practitioner")
        if not filtered_df.empty:
            # Grouping again in case 'All' months selected to aggregate practitioner totals
            pract_rev = filtered_df.groupby('practitioner')['total_revenue'].sum().sort_values(ascending=True).tail(15)
            
            fig = px.bar(
                x=pract_rev.values, 
                y=pract_rev.index, 
                orientation='h',
                text_auto='.2s',
                color=pract_rev.values, 
                color_continuous_scale='Greens',
                labels={'x': 'Total Amount (£)', 'y': 'Practitioner'}
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.subheader("📄 Volume by Practitioner")
        if not filtered_df.empty:
            pract_inv = filtered_df.groupby('practitioner')['total_invoices'].sum().sort_values(ascending=True).tail(15)
            
            fig = px.bar(
                x=pract_inv.values, 
                y=pract_inv.index, 
                orientation='h',
                text_auto=True,
                color=pract_inv.values, 
                color_continuous_scale='Blues',
                labels={'x': 'Count', 'y': 'Practitioner'}
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
    # --- TABLE ---
    st.markdown("---")
    st.subheader("📋 Detailed Breakdown")
    
    summary_df = filtered_df.groupby('practitioner').agg({
        'total_invoices': 'sum',
        'total_revenue': 'sum',
        'card_payments': 'sum', # Note: This is still payment_taken sum
        'cash_payments': 'sum', # Note: This is still payment_taken sum
        'deposit_amount': 'sum',
        'clearearwax_count': 'sum',
        'hearingexpert_count': 'sum'
    }).reset_index()
    
    # Sort by Revenue
    summary_df = summary_df.sort_values('total_revenue', ascending=False)
    
    st.dataframe(
        summary_df.rename(columns={
            'practitioner': 'Practitioner',
            'total_invoices': 'Invoices',
            'total_revenue': 'Total Amount (£)',
            'card_payments': 'Card Paid (£)',
            'cash_payments': 'Cash Paid (£)',
            'deposit_amount': 'Deposits (£)',
            'clearearwax_count': 'CEW Count',
            'hearingexpert_count': 'HE Count'
        }),
        use_container_width=True,
        hide_index=True
    )

# =============================================================================
# PAGE: INVOICE REVENUE BY SITE
# =============================================================================

def render_invoice_revenue_site_dashboard():
    """Render invoice-based revenue by site/clinic dashboard"""
    st.title("🏥 Invoice Revenue - By Site")
    st.caption("📄 Based on actual invoices from clearearwax_finance_invoice_data")
    
    df = load_invoice_revenue_by_site()
    
    if df.empty:
        st.warning("No invoice data found.")
        return
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        months = ['All'] + sorted(df['year_month'].dropna().unique().tolist(), reverse=True)
        selected_month = st.selectbox("📅 Month", months, key="inv_site_month")
    
    with col2:
        sites = ['All'] + sorted(df['site'].dropna().unique().tolist())
        selected_site = st.selectbox("📍 Site", sites, key="inv_site_name")
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_month != 'All':
        filtered_df = filtered_df[filtered_df['year_month'] == selected_month]
    if selected_site != 'All':
        filtered_df = filtered_df[filtered_df['site'] == selected_site]
    
    # Metrics
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("💰 Revenue", f"£{filtered_df['total_revenue'].sum():,.2f}")
    with col2:
        st.metric("🏥 Sites", f"{filtered_df['site'].nunique()}")
    with col3:
        st.metric("📄 Invoices", f"{filtered_df['total_invoices'].sum():,}")
    with col4:
        st.metric("💳 Card", f"£{filtered_df['card_payments'].sum():,.2f}")
    with col5:
        st.metric("👥 Practitioners", f"{filtered_df['practitioner_count'].sum()}")
    
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
        st.subheader("📄 Invoice Count by Site")
        if not filtered_df.empty:
            site_inv = filtered_df.groupby('site')['total_invoices'].sum().sort_values(ascending=True).tail(15)
            fig = px.bar(x=site_inv.values, y=site_inv.index, orientation='h',
                        color=site_inv.values, color_continuous_scale='Blues')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.markdown("---")
    st.subheader("📋 Site Summary")
    
    summary_df = filtered_df.groupby('site').agg({
        'total_invoices': 'sum',
        'total_revenue': 'sum',
        'card_payments': 'sum',
        'cash_payments': 'sum',
        'other_payments': 'sum',
        'deposit_amount': 'sum',
        'practitioner_count': 'sum'
    }).reset_index()
    
    summary_df['avg_per_invoice'] = (summary_df['total_revenue'] / summary_df['total_invoices'].replace(0, 1)).round(2)
    
    st.dataframe(
        summary_df.rename(columns={
            'site': 'Site',
            'total_invoices': 'Invoices',
            'total_revenue': 'Revenue',
            'card_payments': 'Card',
            'cash_payments': 'Cash',
            'other_payments': 'Other',
            'deposit_amount': 'Deposits',
            'avg_per_invoice': 'Avg/Invoice',
            'practitioner_count': 'Practitioners'
        }).sort_values('Revenue', ascending=False),
        use_container_width=True,
        height=400
    )
    
    csv = summary_df.to_csv(index=False)
    st.download_button("📥 Download CSV", data=csv,
                      file_name=f"invoice_revenue_sites_{datetime.now().strftime('%Y%m%d')}.csv",
                      mime="text/csv")

def render_bank_feed_page():
    """Render the Raw Bank Feed page"""
    st.title("💳 Bank Feed (Merchant Transactions)")    
    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.date_input("Start Date", datetime.now() - timedelta(days=7))
    end_date = col_d2.date_input("End Date", datetime.now())

    if start_date > end_date:
        st.error("Start Date must be before End Date")
        return

    df = load_bank_feed_data(start_date, end_date)

    with st.expander("🔎 Filter Transactions", expanded=True):
        col1, col2 = st.columns(2)
        
        # Status Filter
        status_filter = col1.selectbox("Status", ["All", "APPROVED", "DECLINED"])

        # Terminal Filter (Populated from the loaded data)
        if not df.empty:
            # Get unique terminals present in the selected date range
            available_terminals = sorted(df["Terminal"].unique().tolist())
            terminal_options = ["All"] + available_terminals
        else:
            terminal_options = ["All"]
            
        terminal_filter = col2.selectbox("Terminal SN", terminal_options)

    # --- 3. Apply Filters ---
    if df.empty:
        st.info("No transactions found for this date range.")
        return

    # Filter by Status
    if status_filter != "All":
        df = df[df["Status"] == status_filter]

    # Filter by Terminal
    if terminal_filter != "All":
        df = df[df["Terminal"] == terminal_filter]

    # --- 4. Metrics ---
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    
    total_vol = df['Amount'].sum()
    tx_count = len(df)
    terminals_active = df['Terminal'].nunique()
    latest_ingest = df['Ingested At'].max().strftime('%d/%m %H:%M') if not df.empty else "N/A"

    m1.metric("💰 Total Volume", f"£{total_vol:,.2f}")
    m2.metric("💳 Transactions", tx_count)
    # m3.metric("Pp Active Terminals", terminals_active)
    # m4.metric("📥 Latest Import", latest_ingest)

    # --- 5. Data Table ---
    st.markdown(f"### 📋 Transaction Log ({len(df)} records)")
    
    # Style the Status column
    def color_status(val):
        color = '#d4edda' if val == 'APPROVED' else '#f8d7da' # Green / Red
        return f'background-color: {color}; color: black'

    st.dataframe(
        df.style.applymap(color_status, subset=['Status']),
        use_container_width=True,
        height=600
    )

def render_clinic_cleanup_page():
    """Render the Clinic Name Cleanup / Override tool"""
    st.title("🏥 Clinic Name Cleanup")
    st.markdown("""
    **Manage Clinic Names.** Select a clinic name from the list below (Source), select the invoices you want to move, and choose the correct name (Target).
    """)
    
    # Fetch ALL names found in the DB
    all_clinics = fetch_all_clinic_names()
    
    if not all_clinics:
        st.warning("No clinic names found in the database.")
        return

    # --- Step 1: Source Selection (OUTSIDE FORM so it updates immediately) ---
    st.subheader("1. Source: Select Clinic Name to Fix")
    
    selected_source = st.selectbox(
        "Select the name currently on the invoices:", 
        all_clinics,
        key="source_clinic_select"
    )
    
    # Load invoices for the selected source
    df = fetch_invoices_by_clinic_text(selected_source)
    
    if df.empty:
        st.warning(f"No invoices found under the name '{selected_source}'.")
        return
        
    # Add a 'Select' column for the user to check
    # Check if 'Select' already exists to avoid errors, otherwise add it
    if "Select" not in df.columns:
        df.insert(0, "Select", False)
    
    st.markdown("---")
    st.subheader(f"2. Select Invoices & Rename")

    # --- START FORM ---
    # Everything inside this 'with' block will NOT trigger a reload until the button is clicked
    with st.form(key="cleanup_form"):
        
        # 1. The Table
        edited_df = st.data_editor(
            df,
            column_config={
                "Select": st.column_config.CheckboxColumn("Select", default=False),
                "id": st.column_config.NumberColumn("ID", disabled=True),
                "invoice_number": st.column_config.TextColumn("Invoice #", disabled=True),
                "invoice_date": st.column_config.DateColumn("Date", disabled=True),
                "total_amount": st.column_config.NumberColumn("Total", format="£%.2f", disabled=True),
                "current_clinic_name": st.column_config.TextColumn("Current Name", disabled=True),
            },
            hide_index=True,
            use_container_width=True,
            key="clinic_cleanup_editor"
        )
        
        st.markdown("### 3. Choose Target Name")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            target_options = all_clinics + ["-- New / Custom Name --"]
            target_selection = st.selectbox("Select the correct name:", target_options)
            
            # Note: We can't conditionally show a text input effectively INSIDE a form 
            # based on a selection made INSIDE the same form (because it requires a rerun).
            # So we show the text input always, but only use it if "Custom" is picked.
            custom_name_input = st.text_input("If 'New/Custom', type name here:")
        
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            # This is the form submit button
            submit_button = st.form_submit_button("🚀 Rename Selected Invoices", type="primary")

    # --- HANDLE SUBMISSION (OUTSIDE FORM) ---
    if submit_button:
        # Determine the final target name
        final_target_name = target_selection
        if target_selection == "-- New / Custom Name --":
            final_target_name = custom_name_input
        
        # Get selected IDs from the edited dataframe
        selected_rows = edited_df[edited_df["Select"] == True]
        selected_ids = selected_rows['id'].tolist()
        count_selected = len(selected_ids)

        # Validation Logic
        if count_selected == 0:
            st.error("❌ You didn't select any invoices in the table.")
        elif not final_target_name:
            st.error("❌ Target name is empty.")
        elif final_target_name == selected_source:
            st.warning("⚠️ Source and Target names are the same. No changes made.")
        else:
            # Execute Update
            success = bulk_update_clinic_name(selected_ids, final_target_name)
            if success:
                st.success(f"✅ Successfully moved {count_selected} invoices from **'{selected_source}'** to **'{final_target_name}'**!")
                # Optional: Rerun manually to refresh the table and remove the moved rows
                st.rerun()
            else:
                st.error("Failed to update database.")

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
            "💳 Bank Feed (Merchant Transactions)",
            "🕵️ Manual Audit",
            "🏥 Clinic Name Cleanup",
            "✅ Resolved Discrepancies",
            "👤 Clinician Mapping",
            "💰 Revenue - Practitioners", 
            "🏥 Revenue - Sites", 
            "📈 Monthly Overview",
            "📄 Invoice Revenue - Practitioners", 
            "🏪 Invoice Revenue - Sites"
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
    
    # ========================================================================
    # SIDE-BY-SIDE COMPARISON: INVOICES vs BANK TRANSACTIONS
    # ========================================================================
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    # LEFT SIDE: System Invoices (What we EXPECTED)
    with col_left:
        st.markdown("#### 📄 System Invoices")
        st.caption(f"Card payments logged for {discrepancy['practitioner']}")
        
        invoices = load_invoices_for_discrepancy(
            discrepancy['practitioner'],
            discrepancy['reconciliation_date'],
            discrepancy['site']
        )
        
        if not invoices.empty:
            # Filter for card payments only
            card_invoices = invoices[invoices['payment_method'].str.lower().str.contains('card', na=False)]
            
            # Calculate totals
            invoice_total = card_invoices['payment_taken'].sum()
            invoice_count = len(card_invoices)
            
            # Show metrics
            st.metric("Expected Total (Card)", f"£{invoice_total:.2f}", f"{invoice_count} Invoices")
            
            # Show table - USE created_at instead of invoice_date
            display_cols = ['invoice_date', 'site', 'contact_name', 'service_provided', 'payment_taken', 'invoice_number','total_amount']
            display_cols = [c for c in display_cols if c in card_invoices.columns]
            
            # Format time from created_at
            display_df = card_invoices[display_cols].copy()
            if 'created_at' in display_df.columns:
                display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%H:%M')
            
            st.dataframe(
                display_df.rename(columns={
                    'invoice_date': 'Date',
                    'site': 'Site',
                    'contact_name': 'Patient',
                    'service_provided': 'Service',
                    'payment_taken': 'Amount £',
                    'invoice_number': 'Invoice #',
                    'total_amount': 'Total £'
                }),
                use_container_width=True,
                height=300,
                hide_index=True
            )
        else:
            st.warning("⚠️ No invoices found")
            st.caption("This might be due to clinician name mismatch (CRM ID vs Name)")
            st.metric("Expected Total (Card)", "£0.00")
    # RIGHT SIDE: Raw Bank Transactions (What ACTUALLY happened)
    with col_right:
        st.markdown("#### 🏦 Bank Transaction Log ")
        st.caption(f"Raw transactions from Terminal {discrepancy['terminal']}")
        
        bank_transactions = load_bank_transactions_for_discrepancy(
            discrepancy['terminal'],
            discrepancy['reconciliation_date']
        )
        
        if not bank_transactions.empty:
            # Filter approved only
            approved_tx = bank_transactions[bank_transactions['status'] == 'APPROVED']
            
            # Calculate totals
            bank_total = approved_tx['amount'].sum()
            bank_count = len(approved_tx)
            
            # Calculate difference
            if not invoices.empty:
                card_invoices = invoices[invoices['payment_method'].str.lower().str.contains('card', na=False)]
                expected_total = card_invoices['payment_taken'].sum()
                difference = bank_total - expected_total
                
                if difference == 0:
                    delta_msg = "✅ Perfect Match"
                    delta_color = "normal"
                elif difference > 0:
                    delta_msg = f"💰 +£{difference:.2f} Extra"
                    delta_color = "inverse"
                else:
                    delta_msg = f"❌ -£{abs(difference):.2f} Missing"
                    delta_color = "inverse"
            else:
                delta_msg = "No invoices to compare"
                delta_color = "off"
            
            # Show metrics
            st.metric("Actual Total (Bank)", f"£{bank_total:.2f}", delta_msg, delta_color=delta_color)
            
            # Show table
            st.dataframe(
                approved_tx.rename(columns={
                    'time': 'Time',
                    'amount': 'Amount £',
                    'status': 'Status',
                    'card_type': 'Card Type',
                    'card_number': 'Card #'
                }),
                use_container_width=True,
                height=300,
                hide_index=True
            )
            
            # Show declined transactions if any
            declined_tx = bank_transactions[bank_transactions['status'] != 'APPROVED']
            if not declined_tx.empty:
                with st.expander(f"⚠️ {len(declined_tx)} Declined Transactions (Not Counted)", expanded=False):
                    st.dataframe(
                        declined_tx.rename(columns={
                            'time': 'Time',
                            'amount': 'Amount £',
                            'status': 'Status'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
        else:
            st.warning("⚠️ No bank transactions found")
            st.caption(f"No transactions logged for Terminal {discrepancy['terminal']} on this date")
            st.metric("Actual Total (Bank)", "£0.00")
    
    # Additional context from revenue_data
    st.markdown("---")
    st.markdown("#### 📊 Additional Context (Revenue Data Summary)")
    transactions = load_transactions_for_practitioner(
        discrepancy['practitioner'],
        discrepancy['reconciliation_date']
    )
    
    if not transactions.empty:
        display_cols = ['site', 'session_slot', 'appointments', 'revenue', 'card', 'cash']
        display_cols = [c for c in display_cols if c in transactions.columns]
        st.dataframe(
            transactions[display_cols].rename(columns={
                'site': 'Site',
                'session_slot': 'Session',
                'appointments': 'Appts',
                'revenue': 'Revenue £',
                'card': 'Card £',
                'cash': 'Cash £'
            }),
            use_container_width=True,
            height=100,
            hide_index=True
        )
    
    # Resolution form
    st.markdown("---")
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

    site_override = None
    site_is_invalid = (not discrepancy['site'] or 
                   discrepancy['site'].lower() in ['unknown', 'null', 'none', '', 'no schedule'])

# Check if session is invalid (N/A means no schedule)
    session_is_invalid = (not discrepancy['session_slot'] or 
                        discrepancy['session_slot'].lower() in ['n/a', 'null', 'none', ''])

    if site_is_invalid or session_is_invalid:
        if site_is_invalid:
            st.warning(f"⚠️ **Site is REQUIRED but shows: '{discrepancy['site']}'**")
        if session_is_invalid:
            st.warning(f"⚠️ **Session is '{discrepancy['session_slot']}' (No Schedule) - Site required!**")
        
        # Get all available sites
        sites_query = "SELECT DISTINCT site FROM revenue_data WHERE site IS NOT NULL AND site != '' ORDER BY site"
        sites_df = fetch_dataframe(sites_query)
        
        if not sites_df.empty:
            available_sites = sites_df['site'].tolist()
            site_override = st.selectbox(
                "📍 Select Site (Required)",
                options=['-- Select Site --'] + available_sites,
                key=f"site_override_{discrepancy['id']}"
            )
            
            if site_override == '-- Select Site --':
                site_override = None
                st.error("👆 Please select a site above before resolving")
        else:
            st.error("❌ No sites found in database!")
    
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
                update_revenue=update_revenue,
                manual_site=site_override
            )
            st.success("✅ Discrepancy resolved!")
            st.rerun()
        else:
            st.error("Please enter your name")


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
        # Get unique dates from the dataframe
        unique_dates = pd.to_datetime(df['reconciliation_date']).dt.date.unique()
        
        # Add "All Dates" option
        show_all = st.checkbox("📅 All Dates", value=True, key="show_all_dates")
        
        if not show_all:
            # Date Range Picker
            st.markdown("**Select Date Range:**")
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                start_date = st.date_input(
                    "Start Date",
                    value=min(unique_dates) if len(unique_dates) > 0 else datetime.now().date() - timedelta(days=7),
                    min_value=min(unique_dates) if len(unique_dates) > 0 else None,
                    max_value=max(unique_dates) if len(unique_dates) > 0 else None,
                    key="pend_start_date"
                )
            
            with col_d2:
                end_date = st.date_input(
                    "End Date",
                    value=max(unique_dates) if len(unique_dates) > 0 else datetime.now().date(),
                    min_value=min(unique_dates) if len(unique_dates) > 0 else None,
                    max_value=max(unique_dates) if len(unique_dates) > 0 else None,
                    key="pend_end_date"
                )
            
            # Validate date range
            if start_date > end_date:
                st.error("⚠️ Start Date must be before End Date")
                return
        else:
            start_date = None
            end_date = None
    
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
    
    # Date range filter
    if start_date is not None and end_date is not None:
        filtered_df = filtered_df[
            (pd.to_datetime(filtered_df['reconciliation_date']).dt.date >= start_date) &
            (pd.to_datetime(filtered_df['reconciliation_date']).dt.date <= end_date)
        ]
    
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
    
    # Discrepancy list - TABLE VIEW
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
    
    # Export
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
    **Add CRM ID → Name mapping here.**  
    When you add a mapping, it automatically:
    1. ✅ Saves to `practitioner_id_mapping`
    2. ✅ Updates `terminal_assignments` (fills crm_id)
    3. ✅ Updates `practitioner_schedule` (fills crm_id)
    4. ✅ Updates booking data (replaces CRM ID with name)
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
        if st.button("➕ Add & Sync", type="primary"):
            if new_crm_id and new_name:
                add_clinician_mapping(new_crm_id.strip(), new_name.strip())
                st.success(f"""
                ✅ Added: {new_crm_id} → {new_name}
                
                Also updated:
                - terminal_assignments
                - practitioner_schedule  
                - booking data
                """)
                st.rerun()
            else:
                st.error("Please fill in both fields")
    
    # Show sync status
    st.markdown("---")
    st.subheader("📊 Sync Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Count terminal_assignments without crm_id
        query = "SELECT COUNT(*) as count FROM terminal_assignments WHERE crm_id IS NULL"
        result = fetch_dataframe(query)
        count = result['count'].iloc[0] if not result.empty else 0
        st.metric("🔌 Terminals Missing CRM ID", count)
    
    with col2:
        # Count schedules without crm_id
        query = "SELECT COUNT(*) as count FROM practitioner_schedule WHERE crm_id IS NULL"
        result = fetch_dataframe(query)
        count = result['count'].iloc[0] if not result.empty else 0
        st.metric("📅 Schedules Missing CRM ID", count)
    
    with col3:
        # Count bookings with CRM ID as name
        query = """
        SELECT COUNT(*) as count FROM clearearwax_finance_invoice_data 
        WHERE LENGTH(practitioner) > 15 AND practitioner ~ '^[a-zA-Z0-9]+$'
        """
        result = fetch_dataframe(query)
        count = result['count'].iloc[0] if not result.empty else 0
        st.metric("📄 Bookings with CRM ID", count)
    
    # Current mappings (rest of the existing code...)
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
                    delete_clinician_mapping(int(row['id']))
                    st.rerun()
    
    # Bulk sync button
    st.markdown("---")
    st.subheader("🔄 Bulk Sync All Mappings")
    
    st.info("This will update ALL existing mappings across all tables.")
    
    if st.button("🔄 Sync All Mappings Now", type="secondary"):
        sync_all_mappings()
        st.success("✅ All mappings synced!")
        st.rerun()


def sync_all_mappings():
    """Sync all mappings to terminal_assignments, schedule, and bookings"""
    
    # Sync terminal_assignments
    query1 = """
    UPDATE terminal_assignments ta
    SET crm_id = m.crm_id
    FROM practitioner_id_mapping m
    WHERE ta.crm_id IS NULL
      AND LOWER(SPLIT_PART(ta.practitioner_name, ' ', 1)) = LOWER(SPLIT_PART(m.practitioner_name, ' ', 1))
    """
    execute_query(query1)
    
    # Sync practitioner_schedule
    query2 = """
    UPDATE practitioner_schedule ps
    SET crm_id = m.crm_id
    FROM practitioner_id_mapping m
    WHERE ps.crm_id IS NULL
      AND LOWER(SPLIT_PART(ps.practitioner, ' ', 1)) = LOWER(SPLIT_PART(m.practitioner_name, ' ', 1))
    """
    execute_query(query2)
    
    # Sync booking data
    query3 = """
    UPDATE clearearwax_finance_invoice_data f
    SET practitioner = m.practitioner_name,
        clinician_name = m.practitioner_name
    FROM practitioner_id_mapping m
    WHERE f.practitioner = m.crm_id OR f.clinician_name = m.crm_id
    """
    execute_query(query3)
    
    st.cache_data.clear()

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
    elif page == "💳 Bank Feed (Merchant Transactions)":
        render_bank_feed_page()
    elif page == "🕵️ Manual Audit":
        render_manual_audit()
    elif page == "🏥 Clinic Name Cleanup":  
        render_clinic_cleanup_page()
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
    elif page == "📄 Invoice Revenue - Practitioners":
        render_invoice_revenue_practitioner_dashboard()
    elif page == "🏪 Invoice Revenue - Sites":
        render_invoice_revenue_site_dashboard()
    
    # Footer
    st.markdown("---")
    st.caption(f"Reconciliation Dashboard v2.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")

if __name__ == "__main__":
    main()