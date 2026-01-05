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