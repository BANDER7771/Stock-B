import streamlit as st
import pandas_ta as ta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from data_loader import download_stock_data

# عنوان التطبيق
st.title("\u062a\u062d\u0644\u064a\u0644 \u0627\u0644\u0623\u0633\u0647\u0645 \u0628\u0627\u0633\u062a\u062e\u062f\u0627\u0645 \u0627\u0644\u0645\u0624\u0634\u0631\u0627\u062a \u0627\u0644\u0641\u0646\u064a\u0629 \u0648\u0627\u0644\u062a\u0646\u0628\u0624\u0627\u062a \u0627\u0644\u0645\u0633\u062a\u0642\u0628\u0644\u064a\u0629")

# إد\u062eال ال\u0645\u0633\u062aخد\u0645
ticker = st.text_input("\u0623\u062f\u062e\u0644 \u0631\u0645\u0632 \u0627\u0644\u0633\u0647\u0645 (\u0645\u062b\u0644: AAPL):", value="AAPL")
start_date = st.date_input("\u062a\u0627\u0631\u064a\u062e \u0627\u0644\u0628\u062f\u0627\u064a\u0629:", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("\u062a\u0627\u0631\u064a\u062e \u0627\u0644\u0646\u0647\u0627\u064a\u0629:", value=pd.to_datetime("2023-12-31"))
future_days = st.slider("\u0639\u062f\u062f \u0627\u0644\u0623\u064a\u0627\u0645 \u0644\u0644\u062a\u0646\u0628\u0624:", min_value=1, max_value=30, value=5)

def add_technical_indicators(data):
    """
    إض\u0627ف\u0629 ا\u0644م\u0624ش\u0631\u0627\u062a \u0627\u0644\u0641\u0646\u064a\u0629 ب\u0627س\u062a\u062e\u062f\u0627\u0645 pandas-ta
    """
    try:
        # التأ\u0643\u062f من وج\u0648د عمود 'Close' وعدم وجود ق\u064a\u0645 م\u0641ق\u0648دة
        if 'Close' not in data.columns or data['Close'].isnull().all():
            raise ValueError("\u0627\u0644\u0628\u064a\u0627\u0646\u0627\u062a \u0644\u0627 \u062a\u062d\u062a\u0648\u064a \u0639\u0644\u0649 \u0639\u0645\u0648\u062f 'Close' \u0635\u0627\u0644\u062d.")

        # م\u0644\u0621 ال\u0642\u064a\u0645 ال\u0645\u0641\u0642\u0648دة ف\u064a عم\u0648\u062f 'Close'
        data['Close'] = data['Close'].ffill().bfill()

        # م\u0624شر القو\u0629 النس\u0628\u064a\u0629 (RSI)
        data['RSI_14'] = ta.rsi(data['Close'], length=14)

        # م\u0624شر ال\u0645\u0627\u0643\u062f (MACD)
        macd = ta.macd(data['Close'], fast=12, slow=26, signal=9)
        data['MACD'] = macd['MACD_12_26_9']
        data['Signal'] = macd['MACDs_12_26_9']

        # م\u0624شر متوسط الح\u0631\u0643ة البس\u064a\u0637 (SMA)
        data['SMA_20'] = ta.sma(data['Close'], length=20)

        # م\u0624شر متوسط الح\u0631\u0643ة ال\u0623\u0633\u064a (EMA)
        data['EMA_20'] = ta.ema(data['Close'], length=20)

        # Bollinger Bands
        bb = ta.bbands(data['Close'], length=20)
        data['BB_upper'] = bb['BBU_20_2.0']
        data['BB_middle'] = bb['BBM_20_2.0']
        data['BB_lower'] = bb['BBL_20_2.0']

        return data
    except Exception as e:
        st.error(f"\u062d\u062f\u062b \u062e\u0637\u0623 \u0623\u062b\u0646\u0627\u0621 \u062d\u0633\u0627\u0628 \u0627\u0644\u0645\u0624\u0634\u0631\u0627\u062a \u0627\u0644\u0641\u0646\u064a\u0629: {e}")
        return None

if st.button("\u062a\u062d\u0645\u064a\u0644 \u0627\u0644\u0628\u064a\u0627\u0646\u0627\u062a \u0648\u0627\u0644\u062a\u0646\u0628\u0624"):
    with st.spinner("\u062c\u0627\u0631\u064a \u062a\u062d\u0645\u064a\u0644 \u0627\u0644\u0628\u064a\u0627\u0646\u0627\u062a..."):
        # ت\u062d\u0645\u064a\u0644 ب\u064a\u0627\u0646\u0627\u062a ال\u0633\u0647\u0645
        stock_data = download_stock_data(ticker, start_date, end_date)
        
        if stock_data is not None:
            # إعادة تس\u0645ية الأعمدة
            stock_data.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

            # إض\u0627فة ا\u0644م\u0624\u0634\u0631\u0627\u062a ا\u0644\u0641\u0646\u064a\u0629 ب\u0627\u0633\u062a\u062e\u062f\u0627\u0645 pandas-ta
            stock_data = add_technical_indicators(stock_data)

            if stock_data is not None:
                st.success("\u062a\u0645 \u062a\u062d\u0645\u064a\u0644 \u0627\u0644\u0628\u064a\u0627\u0646\u0627\u062a \u0648\u0625\u0636\u0627\u0641\u0629 \u0627\u0644\u0645\u0624\u0634\u0631\u0627\u062a \u0627\u0644\u0641\u0646\u064a\u0629 \u0628\u0646\u062c\u0627\u062d!")
                
                # ع\u0631\u0636 ال\u0628\u064a\u0627\u0646\u0627\u062a
                st.subheader("\u0627\u0644\u0628\u064a\u0627\u0646\u0627\u062a:")
                st.dataframe(stock_data.tail())
                
                # رسم ال\u0645\u0624\u0634\u0631\u0627\u062a
                st.subheader("\u0627\u0644\u0631\u0633\u0648\u0645\u0627\u062a \u0627\u0644\u0628\u064a\u0627\u0646\u064a\u0629:")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(stock_data['Date'], stock_data['Close'], label="Close Price")
                ax.plot(stock_data['Date'], stock_data['SMA_20'], label="SMA 20")
                ax.plot(stock_data['Date'], stock_data['EMA_20'], label="EMA 20")
                ax.fill_between(stock_data['Date'], stock_data['BB_upper'], stock_data['BB_lower'], color='gray', alpha=0.3, label="Bollinger Bands")
                ax.legend()
                ax.set_title("Close Price and Indicators")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price")
                st.pyplot(fig)
                
                # ال\u062a\u0646\u0628\u0624 \u0628\u0627\u0644\u0623\u0633\u0639\u0627\u0631 \u0627\u0644\u0645\u0633\u062a\u0642\u0628\u0644\u064a\u0629
                st.subheader("\u0627\u0644\u062a\u0646\u0628\u0624\u0627\u062a \u0627\u0644\u0645\u0633\u062a\u0642\u0628\u0644\u064a\u0629:")
                stock_data['Prediction'] = stock_data['Close'].shift(-future_days)
                X = stock_data[['Close']].values[:-future_days]
                y = stock_data['Prediction'].values[:-future_days]
                
                # ت\u0642\u0633\u064a\u0645 ا\u0644\u0628\u064a\u0627\u0646\u0627\u062a إ\u0644\u0649 ت\u062f\u0631\u064a\u0628 و\u0627خ\u062a\u0628ا\u0631
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # ت\u062f\u0631\u064a\u0628 النموذج
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # ال\u062a\u0646\u0628\u0624
                future_X = stock_data[['Close']].values[-future_days:]
                predictions = model.predict(future_X)
                
                # ع\u0631\u0636 ا\u0644\u062a\u0646\u0628\u0624\u0627\u062a
                future_dates = pd.date_range(stock_data['Date'].iloc[-1], periods=future_days+1)[1:]
                prediction_df = pd.DataFrame({"Date": future_dates, "Predicted Close": predictions})
                st.dataframe(prediction_df)
                
                # رس\u0645 ا\u0644\u062a\u0646\u0628\u0624\u0627\u062a
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(stock_data['Date'], stock_data['Close'], label="Actual Prices")
                ax.plot(prediction_df['Date'], prediction_df['Predicted Close'], label="Predicted Prices", linestyle="--")
                ax.legend()
                ax.set_title("Actual vs Predicted Prices")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price")
                st.pyplot(fig)
            else:
                st.error("\u0641\u0634\u0644 \u0641\u064a \u0625\u0636\u0627\u0641\u0629 \u0627\u0644\u0645\u0624\u0634\u0631\u0627\u062a \u0627\u0644\u0641\u0646\u064a\u0629.")
        else:
            st.error(f"\u0644\u0627 \u062a\u0648\u062c\u062f \u0628\u064a\u0627\u0646\u0627\u062a \u0644\u0644\u0633\u0647\u0645 {ticker} \u0641\u064a \u0627\u0644\u0641\u062a\u0631\u0629 \u0627\u0644\u0645\u062d\u062f\u062f\u0629.")
