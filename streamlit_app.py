import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# Function to fetch sensor data from Nidopro API
def get_sensor_data(device_id, api_key, from_date, to_date, limit):
    # Actual API endpoint for fetching sensor data
    url = f"https://api.nidopro.com/rest/v1/devices/{device_id}/data"
    headers = {
        "x-api-key": api_key,  # Use the correct header format
        "Content-Type": "application/json"
    }
    params = {
        "from": from_date.isoformat(),  # ISO 8601 format
        "to": to_date.isoformat(),      # ISO 8601 format
        "limit": limit
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json().get("data", [])  # Extract the "data" array
        elif response.status_code == 400:
            st.error(f"Bad Request: {response.json().get('errors', [{}])[0].get('detail', 'Unknown error')}")
        elif response.status_code == 401:
            st.error(f"Unauthorized: {response.json().get('errors', [{}])[0].get('detail', 'Invalid API Key')}")
        elif response.status_code == 404:
            st.error(f"Not Found: {response.json().get('errors', [{}])[0].get('detail', 'Device not found')}")
        else:
            st.error(f"Error fetching sensor data: {response.status_code} - {response.text}")
        return []
    except Exception as e:
        st.error(f"An error occurred while fetching sensor data: {str(e)}")
        return []

# Function to analyze data using DeepSeek AI via OpenRouter
def analyze_data_deepseek(openrouter_api_key, data):
    # OpenRouter API endpoint
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",  # Use your OpenRouter API key
        "Content-Type": "application/json"
    }
    # Prepare the prompt for the AI model
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
        "model": "deepseek/deepseek-r1-distill-llama-70b:free",  # DeepSeek R1 Distill Llama 70B model
        "messages": [
            {"role": "system", "content": "You are an expert in environmental monitoring and agriculture."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]  # Extract the AI's response
        else:
            st.error(f"Error fetching AI analysis: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred during AI analysis: {str(e)}")
        return None

# Function to handle customized chat
def chat_with_ai(openrouter_api_key, topic, user_message):
    # OpenRouter API endpoint
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",  # Use your OpenRouter API key
        "Content-Type": "application/json"
    }
    # Define allowed topics and their descriptions
    allowed_topics = {
        "irrigation": "Discuss irrigation techniques, water usage, and scheduling.",
        "soil_health": "Discuss soil nutrients, pH levels, and soil improvement methods.",
        "pest_control": "Discuss pest management strategies and organic solutions.",
        "crop_management": "Discuss crop rotation, planting schedules, and yield optimization.",
        "general_agriculture": "General questions about agriculture and farming practices."
    }

    # Validate the selected topic
    if topic not in allowed_topics:
        return "Invalid topic selected. Please choose a valid topic."

    # Prepare the prompt for the AI model
    prompt = (
        f"You are an expert in agriculture. Focus on the topic: {allowed_topics[topic]}\n"
        f"User question: {user_message}\n"
        f"Provide a concise and accurate response."
    )
    payload = {
        "model": "deepseek/deepseek-r1-distill-llama-70b:free",  # DeepSeek R1 Distill Llama 70B model
        "messages": [
            {"role": "system", "content": "You are an expert in agriculture."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]  # Extract the AI's response
        else:
            st.error(f"Error fetching AI response: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred during AI chat: {str(e)}")
        return None

# Streamlit app
def main():
    st.title("Environmental Monitoring Dashboard")

    # Input fields for Device ID and API Key
    st.sidebar.header("Authentication")
    device_id = st.sidebar.text_input("Device ID", placeholder="Enter your Device ID")
    nidopro_api_key = st.sidebar.text_input("Nidopro API Key", type="password", placeholder="Enter your Nidopro API Key")
    openrouter_api_key = st.sidebar.text_input("OpenRouter API Key", type="password", placeholder="Enter your OpenRouter API Key")

    # Date range and limit inputs
    st.sidebar.header("Data Retrieval Options")
    from_date = st.sidebar.date_input("From Date", value=datetime(2023, 1, 1))
    to_date = st.sidebar.date_input("To Date", value=datetime.today())
    limit = st.sidebar.slider("Limit (25-1000)", min_value=25, max_value=1000, value=100)

    if not device_id or not nidopro_api_key or not openrouter_api_key:
        st.warning("Please enter your Device ID, Nidopro API Key, and OpenRouter API Key to proceed.")
        return

    # Fetch sensor data
    st.subheader("Sensor Data")
    with st.spinner("Fetching sensor data..."):
        sensor_data = get_sensor_data(device_id, nidopro_api_key, from_date, to_date, limit)

    if sensor_data:
        # Convert sensor data into a DataFrame for easier handling
        df = pd.DataFrame(sensor_data)
        if df.empty:
            st.warning("No data available for the specified date range.")
            return

        # Display metrics for the latest data point
        latest_data = df.iloc[-1]
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("EC (mS/cm)", f"{latest_data.get('EC', 'N/A'):.2f}")
        col2.metric("pH", f"{latest_data.get('pH', 'N/A'):.2f}")
        col3.metric("Water Temp (°C)", f"{latest_data.get('waterTemp', 'N/A'):.2f}")
        col4.metric("Air Temp (°C)", f"{latest_data.get('airTemp', 'N/A'):.2f}")
        col5.metric("Air Humidity (%)", f"{latest_data.get('airHum', 'N/A'):.2f}")

        # AI Analysis Section
        st.subheader("AI-Powered Analysis")
        if st.button("Run Analysis"):
            with st.spinner("Analyzing data with DeepSeek AI..."):
                analysis = analyze_data_deepseek(openrouter_api_key, latest_data)
            if analysis:
                st.success("Analysis Complete!")
                st.write(analysis)

        # Historical Data Visualization
        st.subheader("Historical Data")
        st.line_chart(df[["EC", "pH", "airTemp", "airHum"]])

    # Customized Chat Section
    st.subheader("Agriculture Chat")
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
    if st.button("Send"):
        if not user_message.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating response..."):
                response = chat_with_ai(openrouter_api_key, topic_key, user_message)
            if response:
                st.success("Response:")
                st.write(response)

if __name__ == "__main__":
    main()