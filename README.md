# AuthenScan – Invoice and Transaction Fraud Detection

**AuthenScan** is a fraud detection web application that combines image forgery detection and transaction anomaly detection. It uses machine learning and deep learning models to help detect:
- Forged or tampered invoice images
- Fraudulent banking transactions

---

## Features

### Invoice Forgery Detection
Detects visually manipulated or fake invoice scans using a Convolutional Neural Network (CNN) model trained on ELA (Error Level Analysis) images.

- Upload an invoice image in `.jpg` or `.png` format
- The app uses ELA to preprocess the image
- The CNN model classifies the image as:
  - Real
  - Forged

Use Case: Verifying scanned invoices before processing in financial or enterprise systems.

---

### Bank Transaction Fraud Detection
Detects suspicious or fake bank transactions using a trained machine learning model on real financial transaction datasets.

Input fields:
- Sender Account ID
- Receiver Account ID
- Transaction Amount
- Transaction Type (e.g., NEFT, IMPS)
- Transaction Status (Success/Failed)
- Device Used (Mobile/Desktop)
- PIN Code

Model output:
- Legitimate transaction
- Fraudulent transaction

---

### Email Alert System
- If a transaction is flagged as fraudulent, the system sends an alert email to the registered user.

---

## Machine Learning Models

1. `cnn_forgery_model.h5`  
   CNN trained on ELA-processed invoice images to detect image forgery

2. `fraud_model.pkl`  
   Random Forest or XGBoost model trained on structured bank transaction data

3. Label Encoders:
   - `account_encoder.pkl`
   - `account1_encoder.pkl`
   - `payment_encoder.pkl`

---

## Download Pre-trained Models

Due to GitHub file size limits, the trained models are hosted on Google Drive.

Google Drive Folder:  
[Download Models and Encoders](https://drive.google.com/drive/folders/1HYovIAYPzecIcnNeqfQBJCOjcPlDLIG_?usp=drive_link)

After downloading, place the files in the following structure:

