import streamlit as st
import pandas as pd
import os
from pathlib import Path

# --- CONFIGURATION ---
WORKING_LEDGER = 'EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_WORKING.csv'
CHARTS_DIR = 'manual_trade_chartpacks/fast_signal'
SCHEMA_FILE = 'EURUSD_MANUAL_ANNOTATION_SCHEMA.md'

st.set_page_config(page_title="EURUSD Fast Signal Annotation Studio", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    if not os.path.exists(WORKING_LEDGER):
        st.error(f"Ledger file not found: {WORKING_LEDGER}")
        return None
    df = pd.read_csv(WORKING_LEDGER)
    # Convert human annotation fields to object dtype to allow string values
    human_fields = ['liquidity_source', 'trigger_type', 'confirmation_type', 
                    'operational_context', 'entry_motive', 'quality_rating', 'comment']
    for field in human_fields:
        df[field] = df[field].astype(object)
    return df

def save_data(df):
    try:
        abs_path = os.path.abspath(WORKING_LEDGER)
        df.to_csv(WORKING_LEDGER, index=False)
        mtime = datetime.fromtimestamp(os.path.getmtime(WORKING_LEDGER)).strftime("%Y-%m-%d %H:%M:%S")
        return True, f"Saved to {abs_path} at {mtime}"
    except Exception as e:
        return False, f"Save failed: {str(e)}"

# Taxonomy from Schema (Static for speed, but could be parsed)
TAXONOMY = {
    'liquidity_source': ['previous_day_high', 'previous_day_low', 'asia_high', 'asia_low', 'london_high', 'london_low', 'none_unclear'],
    'trigger_type': ['sweep_reclaim', 'sweep_displacement', 'continuation_after_break', 'reversal_after_sweep', 'breakout_from_compression', 'none_unclear'],
    'confirmation_type': ['close_back_inside', 'strong_displacement_bar', 'structure_break', 'reclaim_then_go', 'immediate_rejection', 'none_unclear'],
    'operational_context': ['london_open_drive', 'london_continuation', 'london_reversal', 'pre_ny_transition', 'early_ny_followthrough', 'none_unclear'],
    'entry_motive': ['liquidity', 'displacement', 'reclaim', 'time_window', 'confluence', 'none_unclear'],
    'quality_rating': ['A', 'B', 'C']
}

# --- UI STYLING ---
st.title("🎯 EURUSD Fast Signal Annotation Studio")
st.markdown("---")

df = load_data()

if df is not None:
    # --- SIDEBAR NAVIGATION ---
    st.sidebar.header("Navigation")
    
    # Progress
    annotated_count = df['liquidity_source'].notna().sum()
    total_count = len(df)
    progress = annotated_count / total_count
    st.sidebar.progress(progress)
    st.sidebar.write(f"**Progress**: {annotated_count} / {total_count} trades")
    
    # Selection
    current_rank_idx = st.sidebar.number_input("Select Rank", min_value=1, max_value=total_count, value=1) - 1
    
    st.sidebar.markdown("---")
    
    # Status display
    row = df.iloc[current_rank_idx]
    st.sidebar.write(f"**Trade ID**: `{row['trade_id']}`")
    st.sidebar.write(f"**Outcome**: {row['outcome']}")
    st.sidebar.write(f"**Side**: {row['side']}")
    st.sidebar.write(f"**Status**: {row['annotation_status']}")
    
    # Navigation Buttons
    col_prev, col_next = st.sidebar.columns(2)
    with col_prev:
        if st.button("⬅ Previous"):
            if current_rank_idx > 0:
                st.session_state.current_idx = current_rank_idx - 1
                st.rerun()
    with col_next:
        if st.button("Next ➡"):
            if current_rank_idx < total_count - 1:
                st.session_state.current_idx = current_rank_idx + 1
                st.rerun()

    # --- MAIN CONTENT ---
    col1, col2 = columns = st.columns([2, 1])

    with col1:
        st.subheader(f"Rank {row['rank']} | Trade {row['trade_id']}")
        
        # Build image path
        # Filename pattern: FS_001_183239883.png
        img_filename = f"FS_{str(row['rank']).zfill(3)}_{row['trade_id']}.png"
        img_path = os.path.join(CHARTS_DIR, img_filename)
        
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.warning(f"Chart Not Found: {img_path}")
            # Try searching by trade_id in the folder
            files = [f for f in os.listdir(CHARTS_DIR) if str(row['trade_id']) in f]
            if files:
                st.image(os.path.join(CHARTS_DIR, files[0]), use_container_width=True)

    with col2:
        st.subheader("Human Annotation")
        
        with st.form("annotation_form"):
            # Prepare default values (handle NaN)
            def get_default(field):
                val = row[field]
                if pd.isna(val) or val not in TAXONOMY[field]:
                    return None
                return TAXONOMY[field].index(val)

            l_source = st.selectbox("Liquidity Source", TAXONOMY['liquidity_source'], index=None, placeholder="Select...", key=f"ls_{current_rank_idx}")
            t_type = st.selectbox("Trigger Type", TAXONOMY['trigger_type'], index=None, placeholder="Select...", key=f"tt_{current_rank_idx}")
            c_type = st.selectbox("Confirmation Type", TAXONOMY['confirmation_type'], index=None, placeholder="Select...", key=f"ct_{current_rank_idx}")
            o_context = st.selectbox("Operational Context", TAXONOMY['operational_context'], index=None, placeholder="Select...", key=f"oc_{current_rank_idx}")
            e_motive = st.selectbox("Entry Motive", TAXONOMY['entry_motive'], index=None, placeholder="Select...", key=f"em_{current_rank_idx}")
            q_rating = st.selectbox("Quality Rating", TAXONOMY['quality_rating'], index=None, placeholder="Select...", key=f"qr_{current_rank_idx}")
            
            comment = st.text_area("Comment", value=row['comment'] if pd.notna(row['comment']) else "", max_chars=200, key=f"cm_{current_rank_idx}")
            
            # Form submission
            submitted = st.form_submit_button("Save Annotation")
            if submitted:
                # Update DataFrame
                df.iloc[current_rank_idx, df.columns.get_loc('liquidity_source')] = l_source
                df.iloc[current_rank_idx, df.columns.get_loc('trigger_type')] = t_type
                df.iloc[current_rank_idx, df.columns.get_loc('confirmation_type')] = c_type
                df.iloc[current_rank_idx, df.columns.get_loc('operational_context')] = o_context
                df.iloc[current_rank_idx, df.columns.get_loc('entry_motive')] = e_motive
                df.iloc[current_rank_idx, df.columns.get_loc('quality_rating')] = q_rating
                df.iloc[current_rank_idx, df.columns.get_loc('comment')] = comment
                
                # Recalculate status
                missing_count = sum([1 for f in TAXONOMY.keys() if pd.isna(df.iloc[current_rank_idx][f])])
                df.iloc[current_rank_idx, df.columns.get_loc('missing_human_fields_count')] = missing_count
                if missing_count == 0:
                    df.iloc[current_rank_idx, df.columns.get_loc('annotation_status')] = 'READY'
                else:
                    df.iloc[current_rank_idx, df.columns.get_loc('annotation_status')] = 'PENDING'
                
                # Save with hardened function
                success, message = save_data(df)
                if success:
                    st.success(f"Trade {row['trade_id']} saved! {message}")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(f"Save failed: {message}")

    # --- FOOTER CONTROLS ---
    st.markdown("---")
    col_val, col_ana = st.columns(2)
    
    with col_val:
        if st.button("🔍 Run Global Validation", use_container_width=True):
            with st.spinner("Validating..."):
                import subprocess
                result = subprocess.run(["python", "scripts/validate_manual_annotations.py"], capture_output=True, text=True)
                st.code(result.stdout)
                if result.returncode == 0:
                    st.success("Ledger is READY for analysis!")
                else:
                    st.error("Ledger has errors or missing fields.")

    with col_ana:
        if st.button("📊 Run Final Analysis", use_container_width=True):
            with st.spinner("Analyzing..."):
                import subprocess
                result = subprocess.run(["python", "scripts/analyze_manual_annotations.py"], capture_output=True, text=True)
                st.code(result.stdout)
                if result.returncode == 0:
                    st.success("Analysis complete!")
                else:
                    st.error("Analysis failed. Ensure validation passes first.")

st.sidebar.markdown("---")
st.sidebar.info("Antigravity Lab - ETAPA 1 Fast Signal")
