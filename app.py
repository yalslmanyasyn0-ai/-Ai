
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="قطرة AI")

BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / "qatra_model (1).pkl")
encoder = joblib.load(BASE_DIR / "qatra_encoder.pkl")


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html dir="rtl" lang="ar">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport"
              content="width=device-width, initial-scale=1.0">
        <title>قطرة AI 💧</title>

        <style>
            body {
                font-family: Arial;
                background: #eaf8fc;
                margin: 0;
                padding: 20px;
            }

            .container {
                max-width: 450px;
                margin: 40px auto;
                padding: 25px;
                background: white;
                border-radius: 20px;
                text-align: center;
                box-shadow: 0 4px 15px #0001;
            }

            h1 { color: #0787A5; }

            select, input, button {
                width: 100%;
                box-sizing: border-box;
                padding: 14px;
                margin: 10px 0;
                font-size: 17px;
                border-radius: 10px;
            }

            select, input {
                border: 1px solid #b5dce7;
            }

            button {
                background: #0787A5;
                color: white;
                border: none;
                font-weight: bold;
            }
        </style>
    </head>

    <body>
        <div class="container">
            <h1>💧 قطرة AI</h1>
            <h3>نظام التنبؤ باستهلاك المياه</h3>

            <form action="/تحليل" method="post">

                <p>نوع المكان:</p>
                <select name="نوع_المكان" required>
                    <option value="منزل">منزل</option>
                    <option value="مدرسة">مدرسة</option>
                </select>

                <p>عدد الأشخاص:</p>
                <input type="number"
                       name="عدد_الأشخاص"
                       min="1"
                       required>

                <button type="submit">
                    احسب الاستهلاك 💧
                </button>

            </form>
        </div>
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

    # مطابقة ترتيب الأعمدة مع ترتيب تدريب النموذج
    if hasattr(model, "feature_names_in_"):
        الأعمدة = list(model.feature_names_in_)

        if set(الأعمدة) != set(البيانات.columns):
            raise ValueError(
                f"أسماء الأعمدة غير متطابقة. "
                f"المطلوب: {الأعمدة}"
            )

        البيانات = البيانات[الأعمدة]

    التوقع = float(model.predict(البيانات)[0])

    return f"""
    <!DOCTYPE html>
    <html dir="rtl" lang="ar">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport"
              content="width=device-width, initial-scale=1.0">
        <title>نتائج قطرة AI</title>
        <style>
            body {{
                font-family: Arial;
                background: #eaf8fc;
                text-align: center;
                padding: 20px;
            }}
            .result {{
                background: white;
                max-width: 450px;
                margin: 50px auto;
                padding: 25px;
                border-radius: 20px;
            }}
            h1, h2 {{ color: #0787A5; }}
            a {{ color: #0787A5; }}
        </style>
    </head>
    <body>
        <div class="result">
            <h1>💧 نتائج قطرة AI</h1>
            <h3>المكان: {نوع_المكان}</h3>
            <h3>عدد الأشخاص: {عدد_الأشخاص}</h3>
            <h2>الاستهلاك المتوقع</h2>
            <h1>{التوقع:.1f} لتر يومياً</h1>
            <a href="/">العودة للرئيسية</a>
        </div>
    </body>
    </html>
    """
    
