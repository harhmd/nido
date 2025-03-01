import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import os
import plotly.express as px
from google.oauth2 import service_account
from googleapiclient.discovery import build

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
if "selected_language" not in st.session_state:  # Track selected language
    st.session_state.selected_language = "English"
if "profile_updated" not in st.session_state:
    st.session_state.profile_updated = False
if "device_profiles" not in st.session_state:
    st.session_state.device_profiles = {}

# Load OpenRouter API Key
if "OPENROUTER_API_KEY" in st.secrets:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
elif "OPENROUTER_API_KEY" in os.environ:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
else:
    st.error("OPENROUTER_API_KEY is missing. Please configure it in secrets.toml or as an environment variable.")
    st.stop()

# Load OpenWeatherMap API Key
if "OPENWEATHERMAP_API_KEY" in st.secrets:
    WEATHER_API_KEY = st.secrets["OPENWEATHERMAP_API_KEY"]
elif "OPENWEATHERMAP_API_KEY" in os.environ:
    WEATHER_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
else:
    st.error("OpenWeatherMap API key is missing. Please configure it in secrets.toml or as an environment variable.")
    st.stop()

# Function to initialize Google Sheets API
def init_google_sheets_api():
    try:
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        service = build("sheets", "v4", credentials=creds)
        return service
    except Exception as e:
        st.error(f"Failed to initialize Google Sheets API: {str(e)}")
        return None

# Function to load device profiles from Google Sheets
@st.cache_data(ttl=300)  # Cache data for 5 minutes
def load_profiles_from_sheet(service, sheet_id, range_name):
    try:
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=sheet_id, range=range_name).execute()
        values = result.get("values", [])

        if not values:
            st.warning("No data found in Google Sheet.")
            return {}

        headers = values[0]
        rows = values[1:]
        profiles = {row[0]: dict(zip(headers, row)) for row in rows if row}
        return profiles
    except Exception as e:
        st.error(f"An error occurred while loading profiles from Google Sheets: {str(e)}")
        return {}

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

# Function to analyze data using DeepSeek AI via OpenRouter
@st.cache_data(ttl=300)  # Cache analysis results for 5 minutes
def analyze_data_deepseek(sensor_data, location, language="English"):
    """
    Analyze environmental sensor data and weather data using DeepSeek AI.

    Args:
        sensor_data (dict): Environmental sensor data (e.g., EC, pH, temperature, humidity).
        location (str): The location for which to fetch weather data.
        language (str): Language for the analysis ("English" or "Bahasa Malaysia").

    Returns:
        str: Analysis result from DeepSeek AI.
    """
    # Fetch weather data from OpenWeatherMap API
    weather_data = get_weather_forecast(WEATHER_API_KEY, location)

    if not weather_data:
        st.warning("Failed to fetch weather data. Analysis will proceed without it.")
        weather_info = "No weather data available."
    else:
        current_temp = weather_data["main"]["temp"]
        weather_desc = weather_data["weather"][0]["description"]
        weather_info = (
            f"- Current Temperature: {current_temp}°C\n"
            f"- Weather Description: {weather_desc.capitalize()}\n"
        )

    # Prepare the prompt for DeepSeek AI
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    # Adjust prompt based on language
    if language == "Bahasa Malaysia":
        prompt = (
            f"Analisis data persekitaran dan cuaca berikut untuk mengenal pasti sebarang masalah yang mungkin:\n"
            f"- EC (Kekonduksian Elektrik): {sensor_data.get('EC', 'N/A')} mS/cm\n"
            f"- pH: {sensor_data.get('pH', 'N/A')}\n"
            f"- Suhu Air: {sensor_data.get('waterTemp', 'N/A')} °C\n"
            f"- Suhu Udara: {sensor_data.get('airTemp', 'N/A')} °C\n"
            f"- Kelembapan Udara: {sensor_data.get('airHum', 'N/A')} %\n"
            f"{weather_info}\n"
            f"Lokasi ladang: {location}\n"
            f"Berikan cadangan atau tindakan pembaikan berdasarkan data ini. Jawab dalam Bahasa Malaysia."
        )
    else:
        prompt = (
            f"Analyze the following environmental and weather data to identify any potential problems:\n"
            f"- EC (Electrical Conductivity): {sensor_data.get('EC', 'N/A')} mS/cm\n"
            f"- pH: {sensor_data.get('pH', 'N/A')}\n"
            f"- Water Temperature: {sensor_data.get('waterTemp', 'N/A')} °C\n"
            f"- Air Temperature: {sensor_data.get('airTemp', 'N/A')} °C\n"
            f"- Air Humidity: {sensor_data.get('airHum', 'N/A')} %\n"
            f"{weather_info}\n"
            f"Farm Location: {location}\n"
            f"Provide recommendations or corrective actions based on this data."
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

# Function to fetch weather forecast from OpenWeatherMap API
@st.cache_data(ttl=300)  # Cache weather data for 5 minutes
def get_weather_forecast(api_key, location):
    """
    Fetch weather forecast data for a given location using the OpenWeatherMap API.
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            st.session_state.api_requests += 1  # Track API requests
            weather_data = response.json()
            return weather_data
        elif response.status_code == 404:
            st.error(f"Location '{location}' not found. Please check the spelling and try again.")
            return None
        else:
            st.error(f"Error fetching weather data: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred while fetching weather data: {str(e)}")
        return None

# Streamlit App
def main():
    # Title and Header
    st.markdown("Environmental Monitoring Dashboard", unsafe_allow_html=True)

    # Check for persistent login via query parameters
    query_params = st.query_params
    if "logged_in" in query_params and query_params["logged_in"] == "True":
        st.session_state.logged_in = True
        st.session_state.device_id = query_params.get("device_id", "")
        st.session_state.nidopro_api_key = query_params.get("nidopro_api_key", "")

    # Login Section
    if not st.session_state.logged_in:
        st.sidebar.markdown("Login", unsafe_allow_html=True)
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
    st.sidebar.markdown("Data Retrieval Options", unsafe_allow_html=True)

    # Date Selection
    today = datetime.today().date()
    date_options = {
        "Today": today,
        "Yesterday": today - timedelta(days=1),
        "Last 7 Days": today - timedelta(days=7),
        "Last 30 Days": today - timedelta(days=30),
    }
    selected_date_option = st.sidebar.selectbox("Select Date Range:", list(date_options.keys()), index=2)  # Default to "Last 7 Days"
    from_date = date_options[selected_date_option] if selected_date_option != "Today" else today
    to_date = today

    # Limits Parameter
    limit = st.sidebar.slider("Limit (25-1000)", min_value=25, max_value=1000, value=1000)  # Default to 1000

    # Language Selection
    language_options = ["English", "Bahasa Malaysia"]
    st.session_state.selected_language = st.sidebar.selectbox("Select Language:", language_options)

    # Display API Usage Statistics
    st.sidebar.write(f"API Requests Made: {st.session_state.api_requests}")

    # Profile Update Section
    with st.sidebar.expander("⚙️ Profile Update"):
        st.markdown("Update Profile")
        profile = st.session_state.device_profiles.get(st.session_state.device_id, {})

        size_farm = st.text_input("Size of Farm (in acres)", value=profile.get("size_farm", ""))
        type_of_plant = st.text_input("Type of Plant", value=profile.get("type_of_plant", ""))
        owner_name = st.text_input("Owner Name", value=profile.get("owner_name", ""))
        exact_location = st.text_input("Exact Location (City/Country)", value=profile.get("exact_location", ""))
        telephone_number = st.text_input("Telephone Number", value=profile.get("telephone_number", ""))

        if st.button("Save Profile"):
            updated_profile = {
                "DeviceID": st.session_state.device_id,
                "size_farm": size_farm,
                "type_of_plant": type_of_plant,
                "owner_name": owner_name,
                "exact_location": exact_location,
                "telephone_number": telephone_number,
            }
            st.session_state.device_profiles[st.session_state.device_id] = updated_profile
            st.session_state.profile_updated = True
            st.success("Profile Updated Successfully!")

        # Display updated profile once
        if st.session_state.profile_updated:
            st.markdown("**Updated Profile:**")
            st.write(f"- Size of Farm: {size_farm}")
            st.write(f"- Type of Plant: {type_of_plant}")
            st.write(f"- Owner Name: {owner_name}")
            st.write(f"- Exact Location: {exact_location}")
            st.write(f"- Telephone Number: {telephone_number}")
            st.session_state.profile_updated = False  # Reset flag after displaying

    # Weather Forecast Section
    if exact_location:
        weather_data = get_weather_forecast(WEATHER_API_KEY, exact_location)
        if weather_data:
            current_temp = weather_data["main"]["temp"]
            weather_desc = weather_data["weather"][0]["description"]

            st.markdown("Current Weather", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"Temperature: {current_temp}°C", unsafe_allow_html=True)
            with col2:
                st.markdown(f"Weather: {weather_desc.capitalize()}", unsafe_allow_html=True)

        # Fetch Sensor Data
    st.markdown("Sensor Data", unsafe_allow_html=True)

    with st.spinner("Fetching sensor data... ⏳"):
        sensor_data = get_sensor_data(
            st.session_state.device_id,
            st.session_state.nidopro_api_key,
            from_date,
            to_date,
            limit
        )

    if sensor_data:
        df = pd.DataFrame(sensor_data)
        if not df.empty:
            # Handle missing or misnamed timestamp fields
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            elif "time" in df.columns:
                df["timestamp"] = pd.to_datetime(df["time"])
            else:
                st.info("No timestamp field found in the API response. Generating timestamps based on data order.")
                df["timestamp"] = pd.date_range(start=from_date, periods=len(df), freq="min")  # Generate timestamps

            # Sort data by timestamp for proper visualization
            df = df.sort_values(by="timestamp")

            # Display Metrics in Beautiful Cards
            latest_data = df.iloc[-1]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"EC (mS/cm): {format_metric(latest_data.get('EC', 'N/A'))}", unsafe_allow_html=True)
            with col2:
                st.markdown(f"pH: {format_metric(latest_data.get('pH', 'N/A'))}", unsafe_allow_html=True)
            with col3:
                st.markdown(f"Air Temp (°C): {format_metric(latest_data.get('airTemp', 'N/A'))}", unsafe_allow_html=True)
            with col4:
                st.markdown(f"Air Humidity (%): {format_metric(latest_data.get('airHum', 'N/A'))}", unsafe_allow_html=True)

            # Line Graph Section
            st.subheader("Trend Analysis")
            st.write("Visualize trends over time for key metrics.")

            # Allow users to select metrics for the graph
            metrics = ["EC", "pH", "airTemp", "airHum"]
            selected_metrics = st.multiselect("Select Metrics to Plot:", metrics, default=metrics)

            if selected_metrics:
                # Filter the DataFrame to include only selected metrics
                filtered_df = df[["timestamp"] + selected_metrics]

                # Melt the DataFrame for Plotly compatibility
                melted_df = filtered_df.melt(id_vars="timestamp", var_name="Metric", value_name="Value")

                # Create an interactive line graph using Plotly
                fig = px.line(
                    melted_df,
                    x="timestamp",
                    y="Value",
                    color="Metric",
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

            # Export Data to CSV
            st.subheader("Export Data")
            st.write("Download the fetched sensor data as a CSV file.")
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV 📊",
                data=csv,
                file_name=f"sensor_data_{datetime.now().strftime('%Y-%m-%d')}.csv",
                mime="text/csv"
            )

            # AI Analysis Section
            st.subheader("AI-Powered Analysis")
            if st.button("Run Analysis", key="run_analysis"):
                with st.spinner("Analyzing data with DeepSeek AI... 🤖"):
                    location = profile.get("exact_location", "Unknown Location")
                    analysis_result = analyze_data_deepseek(latest_data, location, language=st.session_state.selected_language)
                    if analysis_result:
                        st.success("Analysis Complete! ✅")
                        st.write(analysis_result)

        else:
            st.warning("No data available for the specified date range. ❌")

if __name__ == "__main__":
    main()