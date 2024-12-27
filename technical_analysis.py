import pandas as pd
import pandas_ta as ta

def add_technical_indicators(data):
    """
    إضافة المؤشرات الفنية إلى بيانات الأسهم باستخدام pandas-ta.
    """
    try:
        # التأكد من وجود عمود 'Close' وعدم وجود قيم مفقودة
        if 'Close' not in data.columns or data['Close'].isnull().all():
            raise ValueError("البيانات لا تحتوي على عمود 'Close' صالح.")

        # ملء القيم المفقودة في عمود 'Close'
        data['Close'] = data['Close'].ffill().bfill()

        # مؤشر القوة النسبية (RSI)
        data['RSI_14'] = ta.rsi(data['Close'], length=14)
        print("RSI_14 تم حسابه بنجاح.")

        # مؤشر الماكد (MACD)
        macd = ta.macd(data['Close'], fast=12, slow=26, signal=9)
        data['MACD'] = macd['MACD_12_26_9']
        data['Signal'] = macd['MACDs_12_26_9']
        print("MACD تم حسابه بنجاح.")

        # مؤشر متوسط الحركة البسيط (SMA)
        data['SMA_20'] = ta.sma(data['Close'], length=20)
        print("SMA_20 تم حسابه بنجاح.")

        # مؤشر متوسط الحركة الأسي (EMA)
        data['EMA_20'] = ta.ema(data['Close'], length=20)
        print("EMA_20 تم حسابه بنجاح.")

        # Bollinger Bands
        bb = ta.bbands(data['Close'], length=20)
        data['BB_upper'] = bb['BBU_20_2.0']
        data['BB_middle'] = bb['BBM_20_2.0']
        data['BB_lower'] = bb['BBL_20_2.0']
        print("Bollinger Bands تم حسابه بنجاح.")

        print("تم إضافة المؤشرات الفنية بنجاح.")
        return data
    except Exception as e:
        print(f"حدث خطأ أثناء حساب المؤشرات الفنية: {e}")
        return None

if __name__ == "__main__":
    from data_loader import download_stock_data

    # تحميل بيانات السهم
    ticker = "AAPL"  # رمز السهم (يمكن تغييره)
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    stock_data = download_stock_data(ticker, start_date, end_date)

    if stock_data is not None:
        print("معلومات البيانات قبل إضافة المؤشرات:")
        print(stock_data.info())
        print(stock_data.head())

        # إعادة تسمية الأعمدة يدويًا
        stock_data.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

        # التحقق من وجود عمود 'Close'
        if 'Close' not in stock_data.columns:
            print("لا يوجد عمود Close بعد المعالجة!")
        else:
            # إضافة المؤشرات الفنية
            stock_data = add_technical_indicators(stock_data)

            if stock_data is not None:
                print("تمت إضافة المؤشرات بنجاح. عرض البيانات:")
                print(stock_data.tail())
            else:
                print("فشل في إضافة المؤشرات الفنية.")
    else:
        print("فشل في تحميل بيانات السهم.")
