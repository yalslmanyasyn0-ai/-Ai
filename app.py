
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
        <meta name="viewport" content="width=device-width, initial-scale=1">
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
                margin: 35px auto;
                padding: 25px;
                background: white;
                border-radius: 20px;
                text-align: center;
                box-shadow: 0 4px 15px #0001;
            }
            h1 { color: #0787A5; }
            input, select, button {
                width: 100%;
                box-sizing: border-box;
                padding: 14px;
                margin: 10px 0;
                font-size: 17px;
                border-radius: 10px;
            }
            input, select {
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
            <h3>نظام التنبؤ ومراقبة استهلاك المياه</h3>

            <form action="/تحليل" method="post">
                <p>نوع المكان:</p>
                <select name="نوع_المكان" required>
                    <option value="منزل">منزل</option>
                    <option value="مدرسة">مدرسة</option>
                </select>

                <p>عدد الأشخاص:</p>
                <input type="number" name="عدد_الأشخاص"
                       min="1" required>

                <p>الاستهلاك الفعلي باللتر يومياً:</p>
                <input type="number" name="الاستهلاك_الفعلي"
                       min="0" step="0.1" required
                       placeholder="مثلاً 850">

                <button type="submit">
                    افحص استهلاك المياه 💧
                </button>
            </form>
        </div>
    </body>
    </html>
    """


@app.post("/تحليل", response_class=HTMLResponse)
def predict(
    نوع_المكان: str = Form(...),
    عدد_الأشخاص: int = Form(...),
    الاستهلاك_الفعلي: float = Form(...)
):

    if نوع_المكان not in ["منزل", "مدرسة"]:
        return "نوع المكان غير صحيح"

    if عدد_الأشخاص < 1 or الاستهلاك_الفعلي < 0:
        return "يرجى إدخال بيانات صحيحة"

    المكان_المحوّل = encoder.transform([نوع_المكان])[0]

    البيانات = pd.DataFrame([{
        "عدد الأشخاص": عدد_الأشخاص,
        "نوع المكان": المكان_المحوّل
    }])

    if hasattr(model, "feature_names_in_"):
        الأعمدة = list(model.feature_names_in_)

        if set(الأعمدة) != set(البيانات.columns):
            raise ValueError(
                f"أسماء الأعمدة غير متطابقة: {الأعمدة}"
            )

        البيانات = البيانات[الأعمدة]

    التوقع = float(model.predict(البيانات)[0])

    الفرق = الاستهلاك_الفعلي - التوقع

    if الفرق > 0:
        الحالة = "تنبيه! الاستهلاك الفعلي أعلى من المتوقع 🚨"
        اللون = "#d9534f"
        التفاصيل = f"مقدار التجاوز: {الفرق:.1f} لتر"
    elif الفرق < 0:
        الحالة = "ممتاز! الاستهلاك أقل من المتوقع 💚"
        اللون = "#16834a"
        التفاصيل = f"أقل من المتوقع بمقدار: {abs(الفرق):.1f} لتر"
    else:
        الحالة = "الاستهلاك مطابق للتوقع 💧"
        اللون = "#0787A5"
        التفاصيل = "لا يوجد فرق بين الاستهلاكين"

    return f"""
    <!DOCTYPE html>
    <html dir="rtl" lang="ar">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>نتائج قطرة AI</title>
        <style>
            body {{
                font-family: Arial;
                background: #eaf8fc;
                padding: 20px;
                text-align: center;
            }}
            .result {{
                background: white;
                max-width: 450px;
                margin: 35px auto;
                padding: 25px;
                border-radius: 20px;
                box-shadow: 0 4px 15px #0001;
            }}
            h1, h2 {{ color: #0787A5; }}
            .status {{
                color: {اللون};
                font-weight: bold;
                font-size: 19px;
            }}
            a {{ color: #0787A5; }}
        </style>
    </head>
    <body>
        <div class="result">
            <h1>💧 نتائج قطرة AI</h1>
            <h3>المكان: {نوع_المكان}</h3>
            <h3>عدد الأشخاص: {عدد_الأشخاص}</h3>

            <p>الاستهلاك المتوقع:</p>
            <h2>{التوقع:.1f} لتر يومياً</h2>

            <p>الاستهلاك الفعلي:</p>
            <h2>{الاستهلاك_الفعلي:.1f} لتر يومياً</h2>

            <hr>

            <p class="status">{الحالة}</p>
            <h3>{التفاصيل}</h3>

            <br>
            <a href="/">العودة للرئيسية</a>
        </div>
    </body>
    </html>
    """
        
