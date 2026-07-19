import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Page Global Setup
st.set_page_config(
    page_title="Fintech Lead Intelligence Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Premium CSS Injecting
st.markdown("""
    <style>
    .main-title { font-size: 40px; color: #1E3A8A; font-weight: 800; text-align: center; margin-bottom: 2px; }
    .subtitle { font-size: 16px; color: #64748B; text-align: center; margin-bottom: 35px; }
    .metric-card { background-color: #FFFFFF; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); border-left: 6px solid #0EA5E9; margin-bottom: 20px; }
    .metric-value { font-size: 28px; font-weight: bold; color: #0F172A; }
    .metric-label { font-size: 14px; color: #64748B; text-transform: uppercase; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏦 Credit Card Lead Intelligence AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enterprise Voting Ensemble Architecture Engine • Robust Production Studio</div>', unsafe_allow_html=True)

# 📥 Safe Model & Scaler Loader Setup
@st.cache_resource
def load_production_artifacts():
    model = joblib.load('final_trio_ensemble_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    ai_engine, production_scaler = load_production_artifacts()
    st.sidebar.success("🤖 Robust Core AI Engine Active")
except Exception as e:
    st.sidebar.error(f"❌ Core System Failure: {e}")
    st.stop()

# 🛠️ ROBUST PREPROCESSING PIPELINE (Matching Training Process Exactly)
def transform_scale_and_align(df, is_bulk_mode=False):
    processed_df = df.copy()
    
    # 1. Handling the Dangerous "Unknown" bias
    if 'Credit_Product' in processed_df.columns:
        processed_df['Credit_Product'] = processed_df['Credit_Product'].replace('Unknown', 'No')
    
    # 2. Log Transformation for Balance
    if 'Avg_Account_Balance' in processed_df.columns:
        processed_df['Avg_Account_Balance_log'] = np.log1p(processed_df['Avg_Account_Balance'])
    
    # 3. Region Code Handling
    if 'Region_Code' in processed_df.columns:
        # Perform categorical encoding during batch mode processing
        processed_df['Region_Code'] = processed_df['Region_Code'].astype('category').cat.codes
    else:
        # Assign a safe fallback default value (0) for single entry profiling to bypass user input
        processed_df['Region_Code'] = 0

    # 4. One-Hot Encoding Structure Mapping
    processed_df['Gender_Male'] = np.where(processed_df['Gender'] == 'Male', 1, 0)
    
    processed_df['Occupation_Other'] = np.where(processed_df['Occupation'] == 'Other', 1, 0)
    processed_df['Occupation_Salaried'] = np.where(processed_df['Occupation'] == 'Salaried', 1, 0)
    processed_df['Occupation_Self_Employed'] = np.where(processed_df['Occupation'] == 'Self_Employed', 1, 0)
    
    processed_df['Channel_Code_X2'] = np.where(processed_df['Channel_Code'] == 'X2', 1, 0)
    processed_df['Channel_Code_X3'] = np.where(processed_df['Channel_Code'] == 'X3', 1, 0)
    processed_df['Channel_Code_X4'] = np.where(processed_df['Channel_Code'] == 'X4', 1, 0)
    
    processed_df['Credit_Product_Unknown'] = np.where(processed_df['Credit_Product'] == 'Unknown', 1, 0)
    processed_df['Credit_Product_Yes'] = np.where(processed_df['Credit_Product'] == 'Yes', 1, 0)
    
    processed_df['Is_Active_Yes'] = np.where(processed_df['Is_Active'] == 'Yes', 1, 0)
    
    # Expected Training Feature Sequence
    expected_features = [
        'Age', 'Region_Code', 'Vintage', 'Gender_Male', 
        'Occupation_Other', 'Occupation_Salaried', 'Occupation_Self_Employed', 
        'Channel_Code_X2', 'Channel_Code_X3', 'Channel_Code_X4', 
        'Credit_Product_Unknown', 'Credit_Product_Yes', 'Is_Active_Yes', 
        'Avg_Account_Balance_log'
    ]
    
    # Fallback missing check
    for feature in expected_features:
        if feature not in processed_df.columns:
            processed_df[feature] = 0
            
    final_features_df = processed_df[expected_features]
    
    # 5. Production Mathematical Scaling Layer (Crucial for Warning/Error Fix)
    scaled_array = production_scaler.transform(final_features_df)
    scaled_dataframe = pd.DataFrame(scaled_array, columns=expected_features)
    
    return scaled_dataframe

# 🎛️ Navigation Control Panel
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Operational Mode")
mode_selector = st.sidebar.radio(
    "Select Workflow Pattern:",
    ["👤 Live Single Customer Profiling", "📊 Batch File Processing (Bulk Upload)"]
)

# -------------------------------------------------------------
# WORKFLOW 1: SINGLE CUSTOMER PREDICTION
# -------------------------------------------------------------
if mode_selector == "👤 Live Single Customer Profiling":
    st.markdown("### 📝 Customer Demographics & Financial Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        customer_age = st.slider("Target Age Spectrum", 18, 95, 38)
        customer_gender = st.selectbox("Identified Gender", ["Male", "Female"])
        # 🎯 SMART BYPASS: Geographical Region Code selectbox has been completely removed from UI layout
    with col2:
        customer_job = st.selectbox("Employment Structure", ["Salaried", "Self_Employed", "Other", "Entrepreneur"])
        
        # User-friendly dictionary mapping for descriptive channel interpretation
        channel_mapping = {
            "X1 - Branch Walk-in (Direct Visit)": "X1",
            "X2 - Telemarketing (Phone Calls)": "X2",
            "X3 - Digital Channels (Website/App)": "X3",
            "X4 - Third-Party Agents (DSA)": "X4"
        }
        
        # Display the full descriptive labels to the user in the UI
        selected_channel_label = st.selectbox("Lead Sourcing Channel", list(channel_mapping.keys()))
        
        # Extract and pass only the core structural code ("X1", "X2", etc.) to the background model
        acquisition_channel = channel_mapping[selected_channel_label]
        vintage_months = st.number_input("Account Longevity/Vintage (Months)", min_value=0, max_value=240, value=24)
    with col3:
        credit_history = st.selectbox("Prior Active Credit History Status", ["No", "Yes"]) # "Unknown" removed
        avg_balance = st.number_input("Average Monthly Account Liquidity (INR)", min_value=0, value=350000)
        active_status = st.selectbox("Recent Engagement Flag (Last 90 Days)", ["Yes", "No"])

    if st.button("🚀 Execute Smart Conversion Probability Analysis"):
        raw_payload = {
            'Gender': [customer_gender], 'Age': [customer_age],
            'Occupation': [customer_job], 'Channel_Code': [acquisition_channel], 'Vintage': [vintage_months],
            'Credit_Product': [credit_history], 'Avg_Account_Balance': [avg_balance], 'Is_Active': [active_status]
        }
        raw_df = pd.DataFrame(raw_payload)
        pipelined_df = transform_scale_and_align(raw_df)
        
        # Real-time Scoring
        prediction_flag = ai_engine.predict(pipelined_df)[0]
        confidence_metric = ai_engine.predict_proba(pipelined_df)[0][1]
        
        st.markdown("---")
        st.markdown("### 🎯 Real-Time Target Acquisition Output")
        out_col1, out_col2 = st.columns([2, 1])
        
        with out_col1:
            if prediction_flag == 1:
                st.balloons()
                st.success(f"## 🏆 HIGH-POTENTIAL TARGET AUDIENCE \n\n **System Confidence Score: {confidence_metric*100:.2f}% Chance of positive conversion.**")
            else:
                st.info(f"## 🔕 LOWER CONVERSION PROBABILITY \n\n **System Confidence Score: {confidence_metric*100:.2f}% Chance of immediate lead transition.**")
                
        with out_col2:
            advice_html = f"""
            <div class="metric-card" style="border-left-color: {'#10B981' if prediction_flag==1 else '#64748B'};">
                <div class="metric-label">Operational Recommendation</div>
                <div class="metric-value" style="font-size: 16px; margin-top: 10px; color:#1E293B;">
                    {"✅ Route instantly to Premium Relationship Desk for high-value priority calls." if prediction_flag==1 else "❌ Exclude from direct active phone campaigns. Save marketing costs."}
                </div>
            </div>
            """
            st.markdown(advice_html, unsafe_allow_html=True)

# -------------------------------------------------------------
# WORKFLOW 2: BULK FILES DATASETS PROCESSING
# -------------------------------------------------------------
else:
    st.markdown("### 📊 Bulk Dataset Batch Deployment Center")
    bulk_uploader_widget = st.file_uploader("Select Target Financial Dataset File", type=["csv"])
    
    if bulk_uploader_widget is not None:
        raw_bulk_data = pd.read_csv(bulk_uploader_widget)
        st.dataframe(raw_bulk_data.head(5), use_container_width=True)
        
        if st.button("🔥 Initialize Bulk Intelligence Compute Cycle"):
            with st.spinner("AI Processing backend pipeline..."):
                processed_bulk = transform_scale_and_align(raw_bulk_data, is_bulk_mode=True)
                
                vector_predictions = ai_engine.predict(processed_bulk)
                vector_probabilities = ai_engine.predict_proba(processed_bulk)[:, 1]
                
                raw_bulk_data['Lead_Conversion_Classification'] = np.where(vector_predictions == 1, "🎯 High Priority Lead", "❌ Low Priority Lead")
                raw_bulk_data['Target_Score_Percentage'] = np.round(vector_probabilities * 100, 2)
                
                sorted_output_report = raw_bulk_data.sort_values(by='Target_Score_Percentage', ascending=False)
                
                st.markdown("---")
                dash1, dash2, dash3 = st.columns(3)
                with dash1:
                    total_leads_identified = int(np.sum(vector_predictions))
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Total Valid Targets Found</div><div class="metric-value">{total_leads_identified} Leads</div></div>', unsafe_allow_html=True)
                with dash2:
                    global_conversion_ratio = (total_leads_identified / len(raw_bulk_data)) * 100
                    st.markdown(f'<div class="metric-card" style="border-left-color: #10B981;"><div class="metric-label">Projected Campaign Success %</div><div class="metric-value">{global_conversion_ratio:.2f}%</div></div>', unsafe_allow_html=True)
                with dash3:
                    st.markdown(f'<div class="metric-card" style="border-left-color: #F59E0B;"><div class="metric-label">Total Records Scanned</div><div class="metric-value">{len(raw_bulk_data)} Customers</div></div>', unsafe_allow_html=True)
                
                st.dataframe(sorted_output_report, use_container_width=True)
                
                stream_csv_data = sorted_output_report.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Optimized Campaign Spreadsheet File (.CSV)",
                    data=stream_csv_data,
                    file_name="Premium_Credit_Card_Leads_Priority_List.csv",
                    mime="text/csv"
                )
                st.success("Analysis Cycle Finished!")