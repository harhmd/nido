import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Helper function to safely format metrics
def format_metric(value):
    """Format a value as a float with 2 decimal places if it's numeric."""
    try:
        return f"{float(value):.2f}"
    except (ValueError, TypeError):
        return "N/A"

# Initialize session state for login persistence
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "device_id" not in st.session_state:
    st.session_state.device_id = ""
if "nidopro_api_key" not in st.session_state:
    st.session_state.nidopro_api_key = ""
if "openrouter_api_key" not in st.session_state:
    st.session_state.openrouter_api_key = ""

# Function to fetch sensor data from Nidopro API
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
            return response.json().get("data", [])
        else:
            st.error(f"Error fetching sensor data: {response.status_code} - {response.text}")
        return []
    except Exception as e:
        st.error(f"An error occurred while fetching sensor data: {str(e)}")
        return []

# Retry logic for API calls
def requests_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 503, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# Function to analyze data using DeepSeek AI via OpenRouter
def analyze_data_deepseek(openrouter_api_key, data, language="English"):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
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
        session = requests_retry_session()
        response = session.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            st.error(f"Error fetching AI analysis: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred during AI analysis: {str(e)}")
        return None

# Function to interact with OpenRouter for chat-based responses
def chat_with_ai(openrouter_api_key, topic_key, user_message, language="English"):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }

    # Adjust prompt based on language
    if language == "Bahasa Malaysia":
        prompt = (
            f"Anda adalah seorang pakar dalam {topic_key.replace('_', ' ')}. "
            f"Jawab soalan berikut dalam Bahasa Malaysia:\n\n{user_message}"
        )
    else:
        prompt = (
            f"You are an expert in {topic_key.replace('_', ' ')}. "
            f"Answer the following question:\n\n{user_message}"
        )

    payload = {
        "model": "deepseek/deepseek-r1-distill-llama-70b:free",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        session = requests_retry_session()
        response = session.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            st.error(f"Error fetching AI response: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred during AI chat: {str(e)}")
        return None

# Inject Custom CSS for Stunning Design
def inject_custom_css():
    custom_css = """
    <style>
        /* Import Lato Font */
        @import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700&display=swap');

        /* General Styling */
        body {
            font-family: 'Lato', sans-serif;
        }
        .stApp {
            background-color: #f8f9fa;
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
 /* Remove "Made with Streamlit" Footer */
        footer {
            visibility: hidden;
        }
        footer:after {
            content: 'Powered By FAMA Negeri Melaka'; /* You can add custom text here if needed */
            visibility: visible;
            display: block;
            position: relative;
            padding: 5px;
            top: 2px;
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
        openrouter_api_key = st.sidebar.text_input("OpenRouter API Key", type="password", placeholder="Enter your OpenRouter API Key")

        if st.sidebar.button("Login"):
            if device_id and nidopro_api_key and openrouter_api_key:
                st.session_state.logged_in = True
                st.session_state.device_id = device_id
                st.session_state.nidopro_api_key = nidopro_api_key
                st.session_state.openrouter_api_key = openrouter_api_key
                st.rerun()  # Updated from st.experimental_rerun()
            else:
                st.sidebar.warning("Please fill in all fields.")
        st.sidebar.markdown("</div></div>", unsafe_allow_html=True)
        return

    # Logout Button
    if st.sidebar.button("Logout"):
        # Clear all session state variables
        for key in list(st.session_state.keys()):
            del st.session_state[key]
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
        if df.empty:
            st.warning("No data available for the specified date range.")
            return

        latest_data = df.iloc[-1]

        # Display Metrics in Beautiful Cards
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

        # AI Analysis Section
        st.subheader("AI-Powered Analysis")
        if st.button("Run Analysis", key="run_analysis"):
            with st.spinner("Analyzing data with DeepSeek AI..."):
                # Use the same date range and limit as configured in the sidebar
                analysis_data = get_sensor_data(
                    st.session_state.device_id,
                    st.session_state.nidopro_api_key,
                    from_date,
                    to_date,
                    limit
                )

                if analysis_data:
                    analysis_df = pd.DataFrame(analysis_data)
                    if not analysis_df.empty:
                        # Use the latest data point for analysis
                        latest_analysis_data = analysis_df.iloc[-1]
                        analysis_result = analyze_data_deepseek(
                            st.session_state.openrouter_api_key,
                            latest_analysis_data,
                            language=selected_language  # Pass selected language
                        )
                        if analysis_result:
                            st.success("Analysis Complete!")
                            st.write(analysis_result)
                    else:
                        st.warning("No data available for the selected analysis period.")
                else:
                    st.error("Failed to fetch data for analysis.")

        # Historical Data Visualization
        st.subheader("Historical Data")
        try:
            numeric_columns = ["EC", "pH", "airTemp", "airHum"]
            df_numeric = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
            df_cleaned = df_numeric.dropna()

            if df_cleaned.empty:
                st.warning("No valid numeric data available for historical visualization.")
            else:
                st.line_chart(df_cleaned)
        except Exception as e:
            st.error(f"An error occurred while preparing the historical data: {str(e)}")

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
                    st.session_state.openrouter_api_key,
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