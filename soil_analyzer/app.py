import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from PIL import Image
import io
import requests
import json
import base64
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY and GROQ_API_KEY.startswith('"') and GROQ_API_KEY.endswith('"'):
    GROQ_API_KEY = GROQ_API_KEY[1:-1]  # Remove quotes if present

# Set page config
st.set_page_config(
    page_title="Soil Health Analyzer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Display plant image at the top (if exists)
if os.path.exists("plant.png"):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("plant.png", use_column_width=True)

# Custom CSS with mobile responsiveness
st.markdown("""
<style>
    .main-header {
        font-size: min(2.5rem, 10vw);
        color: #2e7d32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: min(1.5rem, 7vw);
        color: #388e3c;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #4caf50;
    }
    .metric-container {
        background-color: #f1f8e9;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .result-container {
        background-color: #f9fbe7;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        border-left: 5px solid #8bc34a;
    }
    .info-box {
        background-color: #e8f5e9;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        border-left: 3px solid #4caf50;
    }
    .download-btn {
        display: inline-block;
        background-color: #4caf50;
        color: white;
        padding: 10px 15px;
        border-radius: 5px;
        text-decoration: none;
        margin-top: 10px;
        text-align: center;
        transition: background-color 0.3s, transform 0.2s;
    }
    .download-btn:hover {
        background-color: #2e7d32;
        transform: translateY(-2px);
    }
    .share-btn {
        display: inline-block;
        background-color: #2196f3;
        color: white;
        padding: 10px 15px;
        border-radius: 5px;
        text-decoration: none;
        margin-top: 10px;
        margin-left: 10px;
        text-align: center;
        transition: background-color 0.3s, transform 0.2s;
    }
    .share-btn:hover {
        background-color: #0d47a1;
        transform: translateY(-2px);
    }
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .stHorizontalBlock {
            flex-direction: column;
        }
        .stColumn {
            width: 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Display app title with enhanced styling
st.markdown("<h1 class='main-header'>🌱 Soil Health Analyzer</h1>", unsafe_allow_html=True)

# Initialize session state for active tab
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "analyze"

# Load dataset
@st.cache_data
def load_data():
    try:
        return pd.read_csv("soil_report_dataset_500.csv")
    except FileNotFoundError:
        st.error("Dataset file 'soil_report_dataset_500.csv' not found. Please ensure the file exists in the app directory.")
        # Create a sample dataset for demonstration
        sample_data = {
            'Crop': ['Wheat', 'Rice', 'Corn', 'Tomato', 'Potato'],
            'pH': [6.5, 6.0, 6.8, 6.2, 5.8],
            'N': [80, 90, 85, 75, 70],
            'P': [35, 30, 40, 45, 35],
            'K': [45, 40, 50, 55, 45],
            'Report': ['Sample report for demonstration'] * 5
        }
        return pd.DataFrame(sample_data)

data = load_data()

# Define range values for soil parameters
soil_ranges = {
    "pH": {
        "very_acidic": (0, 5.5),
        "acidic": (5.5, 6.5),
        "neutral": (6.5, 7.5),
        "alkaline": (7.5, 8.5),
        "very_alkaline": (8.5, 14)
    },
    "Nitrogen": {
        "very_low": (0, 50),
        "low": (50, 70),
        "medium": (70, 90),
        "high": (90, 110),
        "very_high": (110, 200)
    },
    "N": {
        "very_low": (0, 50),
        "low": (50, 70),
        "medium": (70, 90),
        "high": (90, 110),
        "very_high": (110, 200)
    },
    "P": {
        "very_low": (0, 20),
        "low": (20, 30),
        "medium": (30, 40),
        "high": (40, 50),
        "very_high": (50, 100)
    },
    "K": {
        "very_low": (0, 30),
        "low": (30, 40),
        "medium": (40, 50),
        "high": (50, 60),
        "very_high": (60, 100)
    }
}

# Function to classify soil parameters
def classify_soil_parameter(value, parameter):
    if parameter not in soil_ranges:
        return "unknown"
    
    for category, (min_val, max_val) in soil_ranges[parameter].items():
        if min_val <= value < max_val:
            return category
    return "out_of_range"

# Function to get color based on category
def get_color_for_category(category):
    colors = {
        "very_acidic": "#d32f2f",
        "acidic": "#f57c00",
        "neutral": "#4caf50",
        "alkaline": "#f57c00",
        "very_alkaline": "#d32f2f",
        "very_low": "#d32f2f",
        "low": "#f57c00",
        "medium": "#4caf50",
        "high": "#2196f3",
        "very_high": "#673ab7"
    }
    return colors.get(category, "#9e9e9e")

# Function to create gauge charts
def create_gauge_chart(value, parameter, min_val=0, max_val=100):
    try:
        category = classify_soil_parameter(value, parameter)
        color = get_color_for_category(category)
        
        # Initialize steps with a default empty list
        steps = []
        # Determine appropriate ranges for gauge
        if parameter == "pH":
            min_val, max_val = 0, 14
            steps = [
                {'range': [soil_ranges[parameter]["very_acidic"][0], soil_ranges[parameter]["very_acidic"][1]], 'color': '#ffcdd2'},
                {'range': [soil_ranges[parameter]["acidic"][0], soil_ranges[parameter]["acidic"][1]], 'color': '#ffecb3'},
                {'range': [soil_ranges[parameter]["neutral"][0], soil_ranges[parameter]["neutral"][1]], 'color': '#c8e6c9'},
                {'range': [soil_ranges[parameter]["alkaline"][0], soil_ranges[parameter]["alkaline"][1]], 'color': '#bbdefb'},
                {'range': [soil_ranges[parameter]["very_alkaline"][0], soil_ranges[parameter]["very_alkaline"][1]], 'color': '#d1c4e9'}
            ]
        elif parameter in ["N", "P", "K"]:
            min_val, max_val = 0, 150
            steps = [
                {'range': [soil_ranges[parameter]["very_low"][0], soil_ranges[parameter]["very_low"][1]], 'color': '#ffcdd2'},
                {'range': [soil_ranges[parameter]["low"][0], soil_ranges[parameter]["low"][1]], 'color': '#ffecb3'},
                {'range': [soil_ranges[parameter]["medium"][0], soil_ranges[parameter]["medium"][1]], 'color': '#c8e6c9'},
                {'range': [soil_ranges[parameter]["high"][0], soil_ranges[parameter]["high"][1]], 'color': '#bbdefb'},
                {'range': [soil_ranges[parameter]["very_high"][0], soil_ranges[parameter]["very_high"][1]], 'color': '#d1c4e9'}
            ]
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': parameter},
            gauge={
                'axis': {'range': [min_val, max_val]},
                'bar': {'color': color},
                'steps': steps,
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': value
                }
            }
        ))
        
        fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
        return fig
    except Exception as e:
        st.error(f"Error creating gauge chart: {str(e)}")
        return go.Figure()

# Function to generate a soil report using Groq
def generate_soil_report(ph, n, p, k):
    try:
        # Ensure we have the required columns
        required_columns = ['pH', 'N', 'P', 'K', 'Crop']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            # Try alternative column names
            column_mapping = {
                'pH-level': 'pH',
                'N-level': 'N',
                'P-level': 'P',
                'K-level': 'K'
            }
            
            for old_name, new_name in column_mapping.items():
                if old_name in data.columns and new_name in missing_columns:
                    data[new_name] = data[old_name]
                    missing_columns.remove(new_name)
        
        if missing_columns:
            error_msg = f"Missing required columns in dataset: {missing_columns}"
            st.error(error_msg)
            return error_msg, "Unknown", pd.DataFrame()
        
        # Find the most suitable crop based on soil parameters
        def weighted_distance(row):
            ph_diff = abs(row['pH'] - ph) / 14  # Normalize pH (0-14 range)
            n_diff = abs(row['N'] - n) / 150    # Normalize N (assuming 0-150 range)
            p_diff = abs(row['P'] - p) / 100    # Normalize P
            k_diff = abs(row['K'] - k) / 100    # Normalize K
            return ph_diff * 0.4 + n_diff * 0.2 + p_diff * 0.2 + k_diff * 0.2

        data['distance'] = data.apply(weighted_distance, axis=1)
        closest_matches = data.sort_values('distance').head(3)
        
        # Get the most suitable crop and its ideal values
        best_match = closest_matches.iloc[0]
        suitable_crop = best_match['Crop']
        
        # If no API key available, use existing report from dataset or generate basic report
        if not GROQ_API_KEY or GROQ_API_KEY == "Your Groq API Key Here":
            st.warning("⚠️ No Groq API key found. Using basic soil analysis.")
            
            # Generate basic report
            ph_status = classify_soil_parameter(ph, "pH")
            n_status = classify_soil_parameter(n, "N")
            p_status = classify_soil_parameter(p, "P")
            k_status = classify_soil_parameter(k, "K")
            
            basic_report = f"""
            <strong>Soil Analysis Report</strong><br>
            <br>
            <strong>Soil Parameters:</strong><br>
            - pH Level: {ph} ({ph_status})<br>
            - Nitrogen: {n} kg/ha ({n_status})<br>
            - Phosphorus: {p} kg/ha ({p_status})<br>
            - Potassium: {k} kg/ha ({k_status})<br>
            <br>
            <strong>Recommended Crop:</strong> {suitable_crop}<br>
            <br>
            <strong>Basic Recommendations:</strong><br>
            - Monitor pH levels and adjust if needed<br>
            - Consider soil amendments based on nutrient levels<br>
            - Regular soil testing recommended<br>
            """
            
            if 'Report' in best_match and pd.notna(best_match['Report']):
                report = best_match['Report']
            else:
                report = basic_report
                
            return report, suitable_crop, closest_matches
        
        # Create prompt for Groq LLM
        prompt = f"""
        Generate a comprehensive soil health report based on the following soil test results:
        
        pH: {ph} (Classification: {classify_soil_parameter(ph, "pH")})
        Nitrogen (N): {n} kg/ha (Classification: {classify_soil_parameter(n, "N")})
        Phosphorus (P): {p} kg/ha (Classification: {classify_soil_parameter(p, "P")})
        Potassium (K): {k} kg/ha (Classification: {classify_soil_parameter(k, "K")})
        
        Most suitable crop based on similar soil profiles: {suitable_crop}
        
        The report should include the following sections:
        1. Soil Deficiency Analysis: Analyze each parameter (pH, N, P, K) stating whether it's optimal, too low, or too high for general plant growth and specifically for {suitable_crop}.
        2. Detailed Recommendations: Suggest specific fertilizers or amendments to address any deficiencies or excesses.
        3. Soil Type Context: Indicate what soil type might have these characteristics and how suitable it is for {suitable_crop}.
        4. Irrigation Recommendation: Suggest appropriate irrigation methods.
        5. Additional Suggestions: Provide 2-3 practical tips for improving soil health for {suitable_crop} cultivation.
        
        Keep the report professional but easy to understand. Use specific measurements and product recommendations where appropriate.
        Format the report with clear section headings and proper paragraph breaks.
        """
        
        try:
            # Call Groq API for report generation
            messages = [
                {
                    "role": "system",
                    "content": "You are a soil science expert specializing in agricultural soil health analysis. Provide informative, accurate and actionable soil reports."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            chat_completion = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama3-70b-8192",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1024,
                    "top_p": 0.9
                },
                timeout=30
            )
            
            if chat_completion.status_code != 200:
                st.error(f"API Error: {chat_completion.status_code} - {chat_completion.text}")
                # Fallback to basic report
                basic_report = f"Basic soil analysis: pH={ph}, N={n}, P={p}, K={k}. Recommended crop: {suitable_crop}"
                return basic_report, suitable_crop, closest_matches
                
            report = chat_completion.json()["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            st.error(f"Error calling Groq API: {str(e)}")
            # Fallback to basic report
            basic_report = f"Basic soil analysis: pH={ph}, N={n}, P={p}, K={k}. Recommended crop: {suitable_crop}"
            return basic_report, suitable_crop, closest_matches
        
        return report, suitable_crop, closest_matches
        
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        return f"Error generating report: {str(e)}", "Unknown", pd.DataFrame()

# Function to create radar chart for soil comparison
def create_radar_chart(ph, n, p, k, best_match):
    try:
        # Normalize values for radar chart
        max_n, max_p, max_k = 150, 100, 100
        
        user_values = [ph/14, n/max_n, p/max_p, k/max_k]
        ideal_values = [best_match['pH']/14, best_match['N']/max_n, best_match['P']/max_p, best_match['K']/max_k]
        
        categories = ['pH', 'Nitrogen', 'Phosphorus', 'Potassium']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=user_values,
            theta=categories,
            fill='toself',
            name='Your Soil'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=ideal_values,
            theta=categories,
            fill='toself',
            name=f'Ideal for {best_match["Crop"]}'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            height=400
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating radar chart: {str(e)}")
        return go.Figure()

# Function to export report as HTML
def get_html_download_link(report_html, filename="soil_report.html"):
    """Generates a link to download the report as HTML"""
    b64 = base64.b64encode(report_html.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}" class="download-btn">Download Report</a>'
    return href

# Function to save report history
def save_report_to_history(ph, n, p, k, report, suitable_crop):
    """Saves the current report to session state history"""
    if 'report_history' not in st.session_state:
        st.session_state.report_history = []
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.report_history.append({
        "timestamp": timestamp,
        "ph": ph,
        "n": n,
        "p": p,
        "k": k,
        "suitable_crop": suitable_crop,
        "report": report
    })
    
    # Keep only the last 10 reports
    if len(st.session_state.report_history) > 10:
        st.session_state.report_history = st.session_state.report_history[-10:]

# Navigation
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
with nav_col1:
    if st.button("🧪 Soil Analysis", key="nav_analyze", help="Analyze soil parameters", use_container_width=True):
        st.session_state.active_tab = "analyze"

with nav_col2:
    if st.button("🌾 Crop Recommendations", key="nav_crops", help="View recommended crops", use_container_width=True):
        st.session_state.active_tab = "crops"

with nav_col3:
    if st.button("📊 Historical Reports", key="nav_history", help="View past reports", use_container_width=True):
        st.session_state.active_tab = "history"

with nav_col4:
    if st.button("⚙️ Help & Settings", key="nav_settings", help="App settings and help", use_container_width=True):
        st.session_state.active_tab = "settings"

# Analyze Section
if st.session_state.active_tab == "analyze":
    # Info box
    st.markdown("""
    <div class="info-box">
    Analyze your soil's health based on pH and NPK values. Get detailed deficiency reports and recommendations to improve your soil quality.
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for desktop layout
    col1, col2 = st.columns([1, 1])
    
    # Input parameters in first column
    with col1:
        st.markdown('<h2 class="sub-header">Soil Parameters</h2>', unsafe_allow_html=True)
        
        ph = st.slider("pH Level (acidity/alkalinity)", 0.0, 14.0, 7.0, 0.1,
                       help="pH below 7 is acidic, pH 7 is neutral, pH above 7 is alkaline")
        
        st.markdown("#### NPK Values (kg/ha)")
        
        n_value = st.slider("Nitrogen (N)", 0, 150, 75, 
                           help="Nitrogen is essential for leaf growth and green vegetation")
        
        p_value = st.slider("Phosphorus (P)", 0, 100, 40, 
                           help="Phosphorus is important for root growth and flower/fruit development")
        
        k_value = st.slider("Potassium (K)", 0, 100, 50, 
                           help="Potassium improves overall plant health and disease resistance")
    
        analyze_button = st.button("🔍 Analyze Soil", type="primary", use_container_width=True)
    
    # Display gauge charts in second column
    with col2:
        st.markdown('<h2 class="sub-header">Current Readings</h2>', unsafe_allow_html=True)
        
        # Create 2x2 grid for gauge charts
        gauge_row1_col1, gauge_row1_col2 = st.columns(2)
        gauge_row2_col1, gauge_row2_col2 = st.columns(2)
        
        with gauge_row1_col1:
            st.plotly_chart(create_gauge_chart(ph, "pH", 0, 14), use_container_width=True)
        
        with gauge_row1_col2:
            st.plotly_chart(create_gauge_chart(n_value, "N"), use_container_width=True)
        
        with gauge_row2_col1:
            st.plotly_chart(create_gauge_chart(p_value, "P"), use_container_width=True)
        
        with gauge_row2_col2:
            st.plotly_chart(create_gauge_chart(k_value, "K"), use_container_width=True)
    
    if analyze_button:
        with st.spinner("Analyzing soil samples and generating report..."):
            report, suitable_crop, similar_soils = generate_soil_report(ph, n_value, p_value, k_value)
            
            # Save to history
            save_report_to_history(ph, n_value, p_value, k_value, report, suitable_crop)
        
        if suitable_crop and suitable_crop != "Unknown" and isinstance(similar_soils, pd.DataFrame) and not similar_soils.empty:
            st.markdown('<h2 class="sub-header">Soil Analysis Results</h2>', unsafe_allow_html=True)
            
            # Create tabs for different parts of the report
            tab1, tab2, tab3 = st.tabs(["📋 Soil Report", "📈 Comparative Analysis", "🌱 Similar Soil Profiles"])
            
            with tab1:
                report_html = f'<div class="result-container">{report}</div>'
                st.markdown(report_html, unsafe_allow_html=True)
                
                # Download options
                col_download, col_share = st.columns(2)
                with col_download:
                    st.markdown(get_html_download_link(report), unsafe_allow_html=True)
                with col_share:
                    st.markdown(f'<a href="#" class="share-btn">Share Report</a>', unsafe_allow_html=True)
            
            with tab2:
                st.subheader("Your Soil vs. Ideal Soil")
                best_match = similar_soils.iloc[0]
                radar_chart = create_radar_chart(ph, n_value, p_value, k_value, best_match)
                st.plotly_chart(radar_chart, use_container_width=True)
                
                # Show differences
                st.markdown("#### Differences from Ideal Values")
                diff_col1, diff_col2, diff_col3, diff_col4 = st.columns(4)
                
                with diff_col1:
                    ph_diff = ph - best_match['pH']
                    st.metric("pH Difference", f"{ph_diff:.1f}")
                
                with diff_col2:
                    n_diff = n_value - best_match['N']
                    st.metric("N Difference", f"{n_diff:.1f}")
                
                with diff_col3:
                    p_diff = p_value - best_match['P']
                    st.metric("P Difference", f"{p_diff:.1f}")
                
                with diff_col4:
                    k_diff = k_value - best_match['K']
                    st.metric("K Difference", f"{k_diff:.1f}")
            
            with tab3:
                st.subheader("Similar Soil Profiles")
                st.write("These are soil profiles from our database that most closely match your soil parameters:")
                display_cols = ['Crop', 'pH', 'N', 'P', 'K']
                if 'distance' in similar_soils.columns:
                    display_cols.append('distance')
                    similar_soils_display = similar_soils[display_cols].rename({'distance': 'Similarity Score'}, axis=1)
                else:
                    similar_soils_display = similar_soils[display_cols]
                st.dataframe(similar_soils_display)
        else:
            st.warning(f"Analysis completed with limited data. Result: {report}")

# Crops Section
elif st.session_state.active_tab == "crops":
    st.markdown('<h2 class="sub-header">Crop Recommendations</h2>', unsafe_allow_html=True)
    
    st.info("This feature shows crop recommendations based on your soil analysis. Run a soil analysis first to see personalized recommendations.")
    
    # Show general crop information
    if not data.empty:
        st.subheader("Available Crops in Database")
        crop_info = data.groupby('Crop').agg({
            'pH': ['min', 'max', 'mean'],
            'N': ['min', 'max', 'mean'],
            'P': ['min', 'max', 'mean'],
            'K': ['min', 'max', 'mean']
        }).round(1)
        
        st.dataframe(crop_info)

# History Section
elif st.session_state.active_tab == "history":
    st.markdown('<h2 class="sub-header">Report History</h2>', unsafe_allow_html=True)
    
    if 'report_history' not in st.session_state or len(st.session_state.report_history) == 0:
        st.info("No report history yet. Generate a soil report to see it here.")
    else:
        # Display history in reverse order (newest first)
        for i, history_item in enumerate(reversed(st.session_state.report_history)):
            with st.expander(f"Report {i+1}: {history_item['timestamp']} - {history_item['suitable_crop']}"):
                st.markdown(f"**pH:** {history_item['ph']}, **N:** {history_item['n']}, **P:** {history_item['p']}, **K:** {history_item['k']}")
                st.markdown(f"**Suitable Crop:** {history_item['suitable_crop']}")
                st.markdown(f'<div class="result-container">{history_item["report"]}</div>', unsafe_allow_html=True)

# Settings Section
elif st.session_state.active_tab == "settings":
    st.markdown('<h2 class="sub-header">App Settings</h2>', unsafe_allow_html=True)
    
    # API Key settings
    st.subheader("API Configuration")
    current_api_key = GROQ_API_KEY if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here" else ""
    new_api_key = st.text_input("Groq API Key", 
                               value=current_api_key, 
                               type="password",
                               help="Enter your Groq API key to enable custom report generation")
    
    if st.button("Save API Key"):
        try:
            with open(".env", "w") as f:
                f.write(f"GROQ_API_KEY={new_api_key}")
            st.success("API key saved successfully! Please restart the app for changes to take effect.")
        except Exception as e:
            st.error(f"Error saving API key: {str(e)}")
    
    # Data Management
    st.subheader("Data Management")
    if st.button("Clear Report History"):
        if 'report_history' in st.session_state:
            st.session_state.report_history = []
        st.success("Report history cleared successfully!")
    
    # About section
    st.subheader("About")
    st.markdown("""
    <p><strong>Soil Health Analyzer v1.0</strong></p>
    <p>
    This web application uses soil parameter values to generate comprehensive soil deficiency reports
    and provides recommendations for improving soil health.
    </p>
    <p><strong>Features:</strong></p>
    <ul>
        <li>Soil pH and NPK analysis</li>
        <li>Crop recommendations based on soil parameters</li>
        <li>Historical report tracking</li>
        <li>Custom soil reports via Groq API integration</li>
    </ul>
    <p>Developed with Streamlit, Plotly, and modern Python libraries to provide an intuitive and interactive user experience.</p>
    """, unsafe_allow_html=True)

