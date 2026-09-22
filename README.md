# Project TNS — Unified AI Platform

Unified runner for two ML apps:
1. **Customer Segmenter (Unsupervised ML)** — KMeans clustering with Streamlit + FastAPI
2. **Loan Approval Predictor (Supervised ML)** — classification with Streamlit + FastAPI

Root `app.py` launches both FastAPI backends as subprocesses and presents a multi-page Streamlit hub. Ready for 1-click Render deployment.

## Team / Contributors

- **Dinesh** — Contributor
- **Dhavaseelan** — Contributor
- **Hazura** — Contributor
- **Aasish Dairel** — Maintainer (`aasishdairel`)

> Work was developed across multiple devices. This README records all contributors. Future commits should use `Co-authored-by:` trailers and each member's own GitHub identity so history stays accurate.

## Project Structure

```
.
├── app.py                      # Unified Streamlit hub + backend launcher
├── requirements.txt            # Root deps for Render
├── customer-segmenter-app/     # KMeans app (backend :8000)
│   ├── app_unsupervised.py
│   ├── main_unsupervised.py
│   ├── train_kmeans.py
│   └── customers.csv
├── loan-approval-app/          # Loan approval app (backend :8001)
│   ├── app.py
│   ├── main.py
│   ├── train_model.py
│   └── loans.csv
└── docs/plan/
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

This starts:
- Customer Segmenter API on `http://127.0.0.1:8000`
- Loan Approval API on `http://127.0.0.1:8001`
- Streamlit hub (port 8501 by default)

Run apps individually:

```bash
# Customer Segmenter
cd customer-segmenter-app
uvicorn main_unsupervised:app --port 8000 &
streamlit run app_unsupervised.py

# Loan Approval
cd loan-approval-app
uvicorn main:app --port 8001 &
streamlit run app.py
```

## Deploy on Render

- Build: `pip install -r requirements.txt`
- Start: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
