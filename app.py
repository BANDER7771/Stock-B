import streamlit as st
import pandas_ta as ta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from data_loader import download_stock_data

# عنوان التطبيق
st.title("تحليل الأسهم باستخدام المؤشرات الفنية والتنبؤات المستقبلية")

# إدخال المستخدم
ticker = st.text_input("أدخل رمز السهم (مثل: AAPL):", value="AAPL")
start_date = st.date_input("تاريخ البداية:", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("تاريخ النهاية:", value=pd.to_datetime("2023-12-31"))
future_days = st.slider("عدد الأيام للتنبؤ:", min_value=1, max_value=30, value=5)

def add_technical_indicators(data):
    """
    إضافة المؤشرات الفنية باستخدام مكتبة pandas-ta
    """
    try:
        # التأكد من وجود عمود 'Close' وعدم وجود قيم مفقودة
        if 'Close' not in data.columns or data['Close'].isnull().all():
            raise ValueError("البيانات لا تحتوي على عمود 'Close' صالح.")

        # ملء القيم المفقودة في عمود 'Close'
        data['Close'] = data['Close'].ffill().bfill()

        # مؤشر القوة النسبية (RSI)
        data['RSI_14'] = ta.rsi(data['Close'], length=14)

        # مؤشر الماكد (MACD)
        macd = ta.macd(data['Close'])
        if macd is not None:
            data['MACD'] = macd['MACD_12_26_9']
            data['Signal'] = macd['MACDs_12_26_9']

        # مؤشر متوسط الحركة البسيط (SMA)
        data['SMA_20'] = ta.sma(data['Close'], length=20)

        # مؤشر متوسط الحركة الأسي (EMA)
        data['EMA_20'] = ta.ema(data['Close'], length=20)

        # Bollinger Bands
        bb = ta.bbands(data['Close'], length=20)
        if bb is not None:
            data['BB_upper'] = bb['BBU_20_2.0']
            data['BB_middle'] = bb['BBM_20_2.0']
            data['BB_lower'] = bb['BBL_20_2.0']

        return data
    except Exception as e:
        st.error(f"حدث خطأ أثناء حساب المؤشرات الفنية: {e}")
        return None

if st.button("تحميل البيانات والتنبؤ"):
    with st.spinner("جاري تحميل البيانات..."):
        # تحميل بيانات السهم
        stock_data = download_stock_data(ticker, start_date, end_date)
        
        if stock_data is not None:
            # إعادة تسمية الأعمدة
            stock_data.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

            # إضافة المؤشرات الفنية باستخدام pandas-ta
            stock_data = add_technical_indicators(stock_data)

            if stock_data is not None:
                st.success("تم تحميل البيانات وإضافة المؤشرات الفنية بنجاح!")
                
                # عرض البيانات
                st.subheader("البيانات:")
                st.dataframe(stock_data.tail())
                
                # رسم المؤشرات
                st.subheader("الرسومات البيانية:")
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
                
                # التنبؤ بالأسعار المستقبلية
                st.subheader("التنبؤات المستقبلية:")
                stock_data['Prediction'] = stock_data['Close'].shift(-future_days)
                X = stock_data[['Close']].values[:-future_days]
                y = stock_data['Prediction'].values[:-future_days]
                
                # تقسيم البيانات إلى تدريب واختبار
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # تدريب النموذج
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # التنبؤ
                future_X = stock_data[['Close']].values[-future_days:]
                predictions = model.predict(future_X)
                
                # عرض التنبؤات
                future_dates = pd.date_range(stock_data['Date'].iloc[-1], periods=future_days+1)[1:]
                prediction_df = pd.DataFrame({"Date": future_dates, "Predicted Close": predictions})
                st.dataframe(prediction_df)
                
                # رسم التنبؤات
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(stock_data['Date'], stock_data['Close'], label="Actual Prices")
                ax.plot(prediction_df['Date'], prediction_df['Predicted Close'], label="Predicted Prices", linestyle="--")
                ax.legend()
                ax.set_title("Actual vs Predicted Prices")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price")
                st.pyplot(fig)
            else:
                st.error("فشل في إضافة المؤشرات الفنية.")
        else:
            st.error(f"لا توجد بيانات للسهم {ticker} في الفترة المحددة.")
