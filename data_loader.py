import yfinance as yf
import pandas as pd
import talib

def download_stock_data(ticker, start_date, end_date):
    """
    تحميل بيانات الأسهم التاريخية باستخدام yfinance
    """
    try:
        # تحميل البيانات
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # التحقق من البيانات
        if data.empty:
            print(f"لا توجد بيانات للسهم {ticker} في الفترة المحددة.")
            return None
        
        # إعادة البيانات بعد تنظيف الأعمدة
        data.reset_index(inplace=True)
        print(f"تم تحميل بيانات السهم {ticker} بنجاح.")
        return data
    except Exception as e:
        print(f"حدث خطأ أثناء تحميل بيانات السهم {ticker}: {e}")
        return None

def add_technical_indicators(data):
    """
    إضافة المؤشرات الفنية باستخدام TA-Lib
    """
    try:
        # التأكد من وجود عمود 'Close' وعدم وجود قيم مفقودة
        if 'Close' not in data.columns or data['Close'].isnull().all():
            raise ValueError("البيانات لا تحتوي على عمود 'Close' صالح.")

        # ملء القيم المفقودة في عمود 'Close'
        data['Close'] = data['Close'].ffill().bfill()

        # مؤشر القوة النسبية (RSI)
        data['RSI_14'] = talib.RSI(data['Close'], timeperiod=14)

        # مؤشر الماكد (MACD)
        macd, macd_signal, macd_hist = talib.MACD(
            data['Close'], 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        data['MACD'] = macd
        data['Signal'] = macd_signal

        # مؤشر متوسط الحركة البسيط (SMA)
        data['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)

        # مؤشر متوسط الحركة الأسي (EMA)
        data['EMA_20'] = talib.EMA(data['Close'], timeperiod=20)

        # Bollinger Bands
        upperband, middleband, lowerband = talib.BBANDS(
            data['Close'], 
            timeperiod=20, 
            nbdevup=2, 
            nbdevdn=2, 
            matype=0
        )
        data['BB_upper'] = upperband
        data['BB_middle'] = middleband
        data['BB_lower'] = lowerband

        return data
    except Exception as e:
        print(f"حدث خطأ أثناء حساب المؤشرات الفنية: {e}")
        return None

# اختبار الكود
if __name__ == "__main__":
    ticker = "AAPL"  # رمز السهم (يمكن تغييره)
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    
    stock_data = download_stock_data(ticker, start_date, end_date)
    if stock_data is not None:
        print(stock_data.head())  # طباعة أول 5 صفوف من البيانات

        # إضافة المؤشرات الفنية
        stock_data = add_technical_indicators(stock_data)
        if stock_data is not None:
            print(stock_data.tail())  # طباعة آخر 5 صفوف من البيانات مع المؤشرات
