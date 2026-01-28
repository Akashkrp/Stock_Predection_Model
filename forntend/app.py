import streamlit as st
import requests
import base64
import pandas as pd
from io import BytesIO
from PIL import Image

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")

def render_footer():
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #0e1117;
            color: #9aa0a6;
            text-align: center;
            padding: 10px;
            font-size: 13px;
            z-index: 9999;
        }
        </style>

        <div class="footer">
            ⚠️ This dashboard is for educational and research purposes only.  
            It is not financial advice. Use at your own risk.
        </div>
        """,
        unsafe_allow_html=True,
    )

# Session State Initialization
if "token" not in st.session_state:
    st.session_state.token = None
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def login(username, password):
    try:
        response = requests.post(
            f"{API_URL}/api/auth/login",
            json={"username": username, "password": password}
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.token = data["access_token"]
            st.session_state.authenticated = True
            return True
        return False
    except Exception as e:
        st.error(f"Connection error: {e}")
        return False

def get_prediction(ticker, lookback):
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    payload = {"ticker": ticker, "lookback": lookback}
    
    with st.spinner(f"Training LSTM with Attention and generating 30-day forecast for {ticker}..."):
        try:
            response = requests.post(f"{API_URL}/api/predict", json=payload, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                return None
        except Exception as e:
            st.error(f"Failed to reach API: {e}")
            return None

# --- UI LOGIC ---
if not st.session_state.authenticated:
    st.title("🔐 Login")
    with st.form("login_form"):
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if login(user, pwd):
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials")
else:
    # Sidebar
    st.sidebar.title("📈 Controls")
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
    lookback = st.sidebar.slider("Lookback Window (Days)", 30, 150, 60)
    
    if st.sidebar.button("Run Prediction & Forecast"):
        result = get_prediction(ticker, lookback)
        
        if result:
            st.title(f"Analysis for {result['ticker']}")
            
            # 1. Metrics Row
            if 'metrics' in result:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("RMSE", f"₹{result['metrics']['rmse']:.2f}")
                m2.metric("MAE", f"₹{result['metrics']['mae']:.2f}")
                m3.metric("MAPE", f"{result['metrics']['mape']:.2f}%")
                m4.metric("Directional Accuracy", f"{result['metrics']['directional_accuracy']:.2f}%")

            # 2. Main Forecast Plot
            img_data = base64.b64decode(result['plot'])
            st.image(Image.open(BytesIO(img_data)), use_column_width=True, caption="Historical Performance + 30 Day Forecast")

            # 3. Forecast Data
            st.divider()
            col_a, col_b = st.columns([1, 1])

            with col_a:
                st.subheader("🚀 30-Day Future Forecast")
                # Creating a DataFrame for the future predictions
                forecast_df = pd.DataFrame({
                    "Date": result['forecast_dates'],
                    "Predicted Price": result['forecast_30_days']
                })
                st.dataframe(forecast_df.style.format({
                    "Predicted Price": "₹{:.2f}"
                }).highlight_max(subset=["Predicted Price"], color='#2e7d32')
                  .highlight_min(subset=["Predicted Price"], color='#c62828'), 
                  use_container_width=True, height=400)

            with col_b:
                st.subheader("📊 Model Summary")
                st.info(f"The model analyzed the last {lookback} days of data to predict a trend. "
                        "Recursive forecasting uses the model's own predicted output to estimate the next day. "
                        "The predictions may flat-line after the first 10 days.")
                
                last_price = result.get('last_actual_price', result.get('last_predicted_price', 0))
                final_forecast = result['forecast_30_days'][-1]
                
                if last_price and last_price > 0:
                    total_change = ((final_forecast - last_price) / last_price) * 100
                else:
                    total_change = 0
                
                st.metric("Current Price", f"₹{last_price:.2f}")
                st.metric("30-Day Target", f"₹{final_forecast:.2f}", f"{total_change:+.2f}%")

    if st.sidebar.button("Logout"):
        st.session_state.token = None
        st.session_state.authenticated = False
        st.rerun()
    
    # Footer
    render_footer()