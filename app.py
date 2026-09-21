"""Combined runner script to launch both FastAPI backends in background threads and present a multi-page unified Streamlit hub on Render."""
import multiprocessing
import os
import sys
import time
from pathlib import Path

import uvicorn

ROOT_DIR = Path(__file__).resolve().parent
CUSTOMER_DIR = ROOT_DIR / "customer-segmenter-app"
LOAN_DIR = ROOT_DIR / "loan-approval-app"

# Ensure both directories are on python path
sys.path.insert(0, str(CUSTOMER_DIR))
sys.path.insert(0, str(LOAN_DIR))


def run_customer_backend():
    os.chdir(CUSTOMER_DIR)
    uvicorn.run("main_unsupervised:app", host="127.0.0.1", port=8000, log_level="error")


def run_loan_backend():
    os.chdir(LOAN_DIR)
    uvicorn.run("main:app", host="127.0.0.1", port=8001, log_level="error")


def start_backends():
    p1 = multiprocessing.Process(target=run_customer_backend, daemon=True)
    p2 = multiprocessing.Process(target=run_loan_backend, daemon=True)
    p1.start()
    p2.start()
    time.sleep(2)  # Give backends time to start up


# Start backends when script is loaded by Streamlit
if "backends_started" not in os.environ:
    os.environ["backends_started"] = "1"
    start_backends()

import streamlit as st

st.set_page_config(
    page_title="Unified AI Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("📌 Select Project")
app_mode = st.sidebar.radio(
    "Choose Service:",
    ["Customer Segmenter (Unsupervised ML)", "Loan Approval Predictor (Supervised ML)"],
)

if app_mode == "Customer Segmenter (Unsupervised ML)":
    # Set default backend URL to local port 8000
    os.environ["BACKEND_URL"] = "http://127.0.0.1:8000"
    os.chdir(CUSTOMER_DIR)
    
    # Execute Customer Segmenter App
    with open(CUSTOMER_DIR / "app_unsupervised.py") as f:
        code = compile(f.read(), CUSTOMER_DIR / "app_unsupervised.py", "exec")
        exec(code, {"__name__": "__main__"})

else:
    # Set default backend URL to local port 8001
    os.environ["BACKEND_URL"] = "http://127.0.0.1:8001"
    os.chdir(LOAN_DIR)
    
    # Execute Loan Approval App
    with open(LOAN_DIR / "app.py") as f:
        code = compile(f.read(), LOAN_DIR / "app.py", "exec")
        exec(code, {"__name__": "__main__"})
