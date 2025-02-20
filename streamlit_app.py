import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import os
import plotly.express as px  # For creating interactive line graphs

# Helper function to safely format metrics
def format_metric(value):
    """Format a value as a float with 2 decimal places if it's numeric."""
    try:
        return f"{float(value):.2f}"
    except (ValueError, TypeError):
        return "N/A"

# Initialize session state for login persistence and API request tracking
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "device_id" not in st.session_state:
    st.session_state.device_id = ""
if "nidopro_api_key" not in st.session_state:
    st.session_state.nidopro_api_key = ""
if "api_requests" not in st.session_state:
    st.session_state.api_requests = 0

# Function to fetch sensor data from Nidopro API
@st.cache_data(ttl=300)  # Cache data for 5 minutes
def get_sensor_data(device_id, api_key, from_date, to_date, limit=None):
    url = f"https://api.nidopro.com/rest/v1/devices/{device_id}/data"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    params = {
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
    }
    if limit is not None:
        params["limit"] = limit

    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            st.session_state.api_requests += 1  # Track API requests
            data = response.json().get("data", [])
            return data
        else:
            st.error(f"Error fetching sensor data: {response.status_code} - {response.text}")
        return []
    except Exception as e:
        st.error(f"An error occurred while fetching sensor data: {str(e)}")
        return []

# Load OpenRouter API Key
if "OPENROUTER_API_KEY" in st.secrets:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
elif "OPENROUTER_API_KEY" in os.environ:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
else:
    st.error("OPENROUTER_API_KEY is missing. Please configure it in secrets.toml or as an environment variable.")
    st.stop()

# Function to analyze data using DeepSeek AI via OpenRouter
@st.cache_data(ttl=300)  # Cache analysis results for 5 minutes
def analyze_data_deepseek(data, language="English"):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    # Adjust prompt based on language
    if language == "Bahasa Malaysia":
        prompt = (
            f"Analisis data persekitaran berikut dan kenal pasti sebarang masalah yang mungkin:\n"
            f"- EC (Kekonduksian Elektrik): {data.get('EC', 'N/A')} mS/cm\n"
            f"- pH: {data.get('pH', 'N/A')}\n"
            f"- Suhu Air: {data.get('waterTemp', 'N/A')} °C\n"
            f"- Suhu Udara: {data.get('airTemp', 'N/A')} °C\n"
            f"- Kelembapan Udara: {data.get('airHum', 'N/A')} %\n\n"
            f"Berikan cadangan atau tindakan pembetulan jika terdapat sebarang masalah. Jawab dalam Bahasa Malaysia."
        )
    else:
        prompt = (
            f"Analyze the following environmental data and identify any potential problems:\n"
            f"- EC (Electrical Conductivity): {data.get('EC', 'N/A')} mS/cm\n"
            f"- pH: {data.get('pH', 'N/A')}\n"
            f"- Water Temperature: {data.get('waterTemp', 'N/A')} °C\n"
            f"- Air Temperature: {data.get('airTemp', 'N/A')} °C\n"
            f"- Air Humidity: {data.get('airHum', 'N/A')} %\n\n"
            f"Provide recommendations or corrective actions if any issues are detected."
        )

    payload = {
        "model": "deepseek/deepseek-r1-distill-llama-70b:free",
        "messages": [
            {"role": "system", "content": "You are an expert in environmental monitoring and agriculture."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            st.session_state.api_requests += 1  # Track API requests
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            st.error(f"Error fetching AI analysis: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred during AI analysis: {str(e)}")
        return None

# Inject Custom CSS for Stunning Design (Argon Theme)
def inject_custom_css():
    custom_css = """
    <style>
        /* Import Lato Font */
        @import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700&display=swap');

        /* General Styling */
        body {
            font-family: 'Lato', sans-serif;
            background-color: #f8f9fe;
            color: #32325d;
        }
        .stApp {
            background: linear-gradient(135deg, #f8f9fe, #e9ecef);
        }
        .sidebar .css-1d391kg {
            background: linear-gradient(135deg, #6a11cb, #2575fc);
            color: white !important;
        }
        .sidebar .css-1cpxqw2 {
            color: white !important;
        }
        .card {
            margin-bottom: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08);
            padding: 1rem;
            text-align: center;
        }
        .metric-card h3 {
            font-size: 1.25rem;
            margin-bottom: 0.5rem;
            color: #ffffff;
        }
        .metric-card p {
            font-size: 1.5rem;
            font-weight: bold;
            color: #ffffff;
        }

        /* Card Colors */
        .ec-card {
            background: linear-gradient(135deg, #00b4d8, #0077b6);
        }
        .ph-card {
            background: linear-gradient(135deg, #ff9f1c, #f77f00);
        }
        .temp-card {
            background: linear-gradient(135deg, #ef476f, #d90429);
        }
        .humidity-card {
            background: linear-gradient(135deg, #8338ec, #6a0dad);
        }

        /* Chat Styling */
        .chat-container {
            margin-top: 1rem;
            padding: 1rem;
            background-color: #ffffff;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08);
        }
        .chat-header {
            margin-bottom: 1rem;
            font-size: 1.25rem;
            font-weight: bold;
            color: #32325d;
        }
        .chat-response {
            margin-top: 1rem;
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 0.5rem;
            color: #32325d;
        }

        /* Sidebar Footer */
        .sidebar-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 250px; /* Adjust based on sidebar width */
            padding: 10px;
            background-color: #6a11cb;
            color: white;
            text-align: center;
            font-size: 0.9rem;
            font-weight: bold;
            border-top-right-radius: 10px;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Streamlit app
def main():
    # Inject Custom CSS
    inject_custom_css()

    # Title and Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #32325d;">Environmental Monitoring Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)

    # Check for persistent login via query parameters
    query_params = st.query_params
    if "logged_in" in query_params and query_params["logged_in"] == "True":
        st.session_state.logged_in = True
        st.session_state.device_id = query_params.get("device_id", "")
        st.session_state.nidopro_api_key = query_params.get("nidopro_api_key", "")

    # Login Section
    if not st.session_state.logged_in:
        st.sidebar.markdown("""
        <div class="card">
            <div class="card-header">
                <h3 class="text-center">Login</h3>
            </div>
            <div class="card-body">
        """, unsafe_allow_html=True)
        device_id = st.sidebar.text_input("Device ID", placeholder="Enter your Device ID")
        nidopro_api_key = st.sidebar.text_input("Nidopro API Key", type="password", placeholder="Enter your Nidopro API Key")

        if st.sidebar.button("Login"):
            if device_id and nidopro_api_key:
                st.session_state.logged_in = True
                st.session_state.device_id = device_id
                st.session_state.nidopro_api_key = nidopro_api_key

                # Persist login state in query parameters
                st.query_params["logged_in"] = "True"
                st.query_params["device_id"] = device_id
                st.query_params["nidopro_api_key"] = nidopro_api_key
                st.rerun()  # Force rerun to reflect changes
            else:
                st.sidebar.warning("Please fill in all fields.")
        st.sidebar.markdown("</div></div>", unsafe_allow_html=True)
        return

    # Logout Button
    if st.sidebar.button("Logout"):
        # Clear session state and query parameters
        st.session_state.logged_in = False
        st.session_state.device_id = ""
        st.session_state.nidopro_api_key = ""
        st.query_params.clear()
        st.rerun()  # Force rerun to reflect changes

    # Sidebar for Inputs
    st.sidebar.markdown("""
    <div class="card">
        <div class="card-header">
            <h3 class="text-center">Data Retrieval Options</h3>
        </div>
        <div class="card-body">
    """, unsafe_allow_html=True)

    # Date Selection
    today = datetime.today().date()
    date_options = {
        "Today": today,
        "Yesterday": today - timedelta(days=1),
        "Last 7 Days": today - timedelta(days=7),
        "Last 30 Days": today - timedelta(days=30),
    }
    selected_date_option = st.sidebar.selectbox("Select Date Range:", list(date_options.keys()))
    from_date = date_options[selected_date_option] if selected_date_option != "Today" else today
    to_date = today

    # Limits Parameter
    limit = st.sidebar.slider("Limit (25-1000)", min_value=25, max_value=1000, value=100)

    # Language Selection
    language_options = ["English", "Bahasa Malaysia"]
    selected_language = st.sidebar.selectbox("Select Language:", language_options)

    # Display API Usage Statistics
    st.sidebar.write(f"API Requests Made: {st.session_state.api_requests}")

    # Add Sidebar Footer
    st.sidebar.markdown("""
    <div class="sidebar-footer">
        Powered By FAMA Negeri Melaka
    </div>
    """, unsafe_allow_html=True)

    # Fetch Sensor Data
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <h3 class="text-center">Sensor Data</h3>
        </div>
        <div class="card-body">
    """, unsafe_allow_html=True)

    with st.spinner("Fetching sensor data..."):
        sensor_data = get_sensor_data(st.session_state.device_id, st.session_state.nidopro_api_key, from_date, to_date, limit)

    if sensor_data:
        df = pd.DataFrame(sensor_data)
        if not df.empty:
            # Handle missing or misnamed timestamp fields
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'])
            else:
                st.info("No timestamp field found in the API response. Generating timestamps based on data order.")
                df['timestamp'] = pd.date_range(start=from_date, periods=len(df), freq='T')  # Generate timestamps

            # Sort data by timestamp for proper visualization
            df = df.sort_values(by='timestamp')

            # Display Metrics in Beautiful Cards
            latest_data = df.iloc[-1]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="card ec-card metric-card">
                    <h3>EC (mS/cm)</h3>
                    <p>{format_metric(latest_data.get("EC", "N/A"))}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="card ph-card metric-card">
                    <h3>pH</h3>
                    <p>{format_metric(latest_data.get("pH", "N/A"))}</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="card temp-card metric-card">
                    <h3>Air Temp (°C)</h3>
                    <p>{format_metric(latest_data.get("airTemp", "N/A"))}</p>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="card humidity-card metric-card">
                    <h3>Air Humidity (%)</h3>
                    <p>{format_metric(latest_data.get("airHum", "N/A"))}</p>
                </div>
                """, unsafe_allow_html=True)

            # Line Graph Section
            st.subheader("Trend Analysis")
            st.write("Visualize trends over time for key metrics.")

            # Allow users to select metrics for the graph
            metrics = ["EC", "pH", "airTemp", "airHum"]
            selected_metrics = st.multiselect("Select Metrics to Plot:", metrics, default=metrics)

            if selected_metrics:
                # Filter the DataFrame to include only selected metrics
                filtered_df = df[['timestamp'] + selected_metrics]

                # Melt the DataFrame for Plotly compatibility
                melted_df = filtered_df.melt(id_vars='timestamp', var_name='Metric', value_name='Value')

                # Create an interactive line graph using Plotly
                fig = px.line(
                    melted_df,
                    x='timestamp',
                    y='Value',
                    color='Metric',
                    title="Sensor Data Trends Over Time",
                    labels={"timestamp": "Timestamp", "Value": "Value"},
                    template="plotly_white"
                )

                # Customize the layout
                fig.update_layout(
                    legend_title_text="Metrics",
                    xaxis_title="Timestamp",
                    yaxis_title="Value",
                    hovermode="x unified",
                    margin=dict(l=20, r=20, t=50, b=20),
                )

                # Display the graph
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one metric to plot.")

            # AI Analysis Section
            st.subheader("AI-Powered Analysis")
            if st.button("Run Analysis", key="run_analysis"):
                with st.spinner("Analyzing data with DeepSeek AI..."):
                    analysis_result = analyze_data_deepseek(latest_data, language=selected_language)
                    if analysis_result:
                        st.success("Analysis Complete!")
                        st.write(analysis_result)

        else:
            st.warning("No data available for the specified date range.")

    st.markdown("</div></div>", unsafe_allow_html=True)

    # Customized Chat Section
    st.markdown("""
    <div class="card chat-container">
        <div class="card-header">
            <h3 class="chat-header">Agriculture Chat</h3>
        </div>
        <div class="card-body">
    """, unsafe_allow_html=True)
    st.write("Ask questions related to agriculture. Choose a topic below:")

    # Topic selection
    allowed_topics = {
        "irrigation": "Irrigation Techniques",
        "soil_health": "Soil Health",
        "pest_control": "Pest Control",
        "crop_management": "Crop Management",
        "general_agriculture": "General Agriculture"
    }
    topic = st.selectbox("Select a topic:", list(allowed_topics.values()))

    # Map selected topic back to its key
    topic_key = next(key for key, value in allowed_topics.items() if value == topic)

    # User input for chat
    user_message = st.text_input("Ask your question:", placeholder="Type your question here...")
    if st.button("Send", key="send_chat"):
        if not user_message.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating response..."):
                response = chat_with_ai(
                    topic_key,
                    user_message,
                    language=selected_language  # Pass selected language
                )
            if response:
                st.markdown(f"""
                <div class="chat-response">
                    <strong>Response:</strong><br>
                    {response}
                </div>
                """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()