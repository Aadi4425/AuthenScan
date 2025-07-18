from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance
import pickle
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load email config
with open('config.json') as config_file:
    config = json.load(config_file)

EMAIL_ADDRESS = config['gmail_user']
EMAIL_PASSWORD = config['gmail_password']
EMAIL_SUBJECT = config['subject']
EMAIL_BODY_TEMPLATE = config['body_template']

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load models
cnn_model = load_model('model/cnn_forgery_model.h5')
with open('model/fraud_model.pkl', 'rb') as f:
    ml_model = pickle.load(f)

# Load encoders
with open('model/account_encoder.pkl', 'rb') as f:
    account_encoder = pickle.load(f)
with open('model/account1_encoder.pkl', 'rb') as f:
    account1_encoder = pickle.load(f)
with open('model/payment_encoder.pkl', 'rb') as f:
    payment_encoder = pickle.load(f)

# 🔧 ELA preprocessing function
def convert_to_ela_image(path, quality=90):
    temp_filename = 'temp_ela_source.jpg'

    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    max_diff = max_diff if max_diff != 0 else 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/invoice', methods=['GET', 'POST'])
def invoice():
    if request.method == 'POST':
        email = request.form['email']
        file = request.files.get('invoice')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # 🔄 Replace with ELA preprocessing
                ela_image = convert_to_ela_image(filepath).resize((128, 128))
                img_array = np.expand_dims(np.array(ela_image) / 255.0, axis=0)

                prediction = cnn_model.predict(img_array)
                is_fraud = np.argmax(prediction) == 0  # 0 = Forged, 1 = Real

                if is_fraud:
                    result = '🚩 Fraudulent Invoice Detected'
                    details = "🚩 Our system detected a *fraudulent invoice*. Immediate action is recommended."
                else:
                    result = '✅ Invoice is Authentic'
                    details = "✅ Your uploaded invoice appears *authentic*. No suspicious activity was detected."

                send_email(email, EMAIL_SUBJECT, details)
                return render_template('invoice_result.html', result=result, image=filename)

            except Exception as e:
                print(f"[ERROR] Invoice analysis failed: {e}")
                flash(f"Invoice analysis failed: {e}")
                return redirect(url_for('invoice'))
        else:
            flash("No valid image file uploaded. Only JPG, JPEG, and PNG are allowed.")
            return redirect(url_for('invoice'))

    return render_template('invoice.html')

@app.route('/transaction', methods=['GET', 'POST'])
def transaction():
    if request.method == 'POST':
        try:
            email = request.form['email']
            from_bank = int(request.form['from_bank'])
            account = request.form['account']
            to_bank = int(request.form['to_bank'])
            receiver_account_raw = request.form['receiver_account']
            amount_received = float(request.form['amount_received'])
            amount_paid = float(request.form['amount_paid'])
            payment_format = request.form['payment_format']

            sender_account = account_encoder.transform([account])[0]
            receiver_account = account1_encoder.transform([receiver_account_raw])[0]
            payment_format_encoded = payment_encoder.transform([payment_format])[0]

            features = np.array([[from_bank, sender_account, to_bank, receiver_account,
                                  amount_received, amount_paid, payment_format_encoded]])

            prediction = ml_model.predict(features)[0]

            if prediction == 1:
                result = "⚠️ Possible Laundering Detected"
                details = "⚠️ Our system detected *possible laundering* in your recent transaction. Please verify the transaction details."
            else:
                result = "✅ Transaction Looks Legitimate"
                details = "✅ Your transaction appears *legitimate*. No suspicious patterns were detected."

            send_email(email, EMAIL_SUBJECT, details)
            return render_template('transaction_result.html', result=result)

        except Exception as e:
            print(f"[ERROR] Transaction check failed: {e}")
            flash(f"Transaction check failed: {e}")
            return redirect(url_for('transaction'))

    return render_template('transaction.html')

def send_email(to_email, subject, details):
    body = EMAIL_BODY_TEMPLATE.replace("{details}", details)
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print(f"[ERROR] Email sending failed: {e}")

if __name__ == '__main__':
    app.run(debug=True)
