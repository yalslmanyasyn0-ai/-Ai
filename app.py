
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="قطرة AI")

# تحميل نموذج الذكاء الاصطناعي والمحوّل
BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / "qatra_model (1).pkl")
encoder = joblib.load(BASE_DIR / "qatra_encoder.pkl")


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html dir="rtl" lang="ar">
    <head>
        <meta charset="UTF-8">
        <title>قطرة AI 💧</title>
    </head>
    <body style="text-align:center;font-family:Arial;padding:25px">
        <h1 style="color:#0787A5">💧 قطرة AI</h1>
        <h3>نظام التنبؤ باستهلاك المياه</h3>

        <form action="/تحليل" method="post">
            <p>نوع المكان:</p>
            <select name="نوع_المكان">
                <option value="منزل">منزل</option>
                <option value="مدرسة">مدرسة</option>
            </select>

            <p>عدد الأشخاص:</p>
            <input type="number" name="عدد_الأشخاص"
                   min="1" required>

            <br><br>
            <button type="submit">احسب الاستهلاك 💧</button>
        </form>
    </body>
    </html>
    """


@app.post("/تحليل", response_class=HTMLResponse)
def predict(
    نوع_المكان: str = Form(...),
    عدد_الأشخاص: int = Form(...)
):
    المكان_المحوّل = encoder.transform([نوع_المكان])[0]

    البيانات = pd.DataFrame([{
        "عدد الأشخاص": عدد_الأشخاص,
        "نوع المكان": المكان_المحوّل
    }])

    التوقع = model.predict(البيانات)[0]

    return f"""
    <html dir="rtl" lang="ar">
    <meta charset="UTF-8">
    <body style="text-align:center;font-family:Arial;padding:30px">
        <h1 style="color:#0787A5">💧 نتائج قطرة</h1>
        <h3>المكان: {نوع_المكان}</h3>
        <h3>عدد الأشخاص: {عدد_الأشخاص}</h3>
        <h2>الاستهلاك المتوقع: {التوقع:.1f} لتر يومياً</h2>
        <a href="/">العودة للرئيسية</a>
    </body>
    </html>
    """
  
