"""Combined runner script to launch both FastAPI backends in background subprocesses and present a multi-page unified Streamlit hub on Render."""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
CUSTOMER_DIR = ROOT_DIR / "customer-segmenter-app"
LOAN_DIR = ROOT_DIR / "loan-approval-app"

# Ensure both directories are on python path
sys.path.insert(0, str(CUSTOMER_DIR))
sys.path.insert(0, str(LOAN_DIR))


def start_backends():
    # Start Customer Segmenter backend on port 8000
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main_unsupervised:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(CUSTOMER_DIR),
    )
    # Start Loan Approval backend on port 8001
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8001"],
        cwd=str(LOAN_DIR),
    )
    time.sleep(3)  # Give backends time to start up and bind ports


# Start backends once per container start
if "BACKENDS_STARTED" not in os.environ:
    os.environ["BACKENDS_STARTED"] = "1"
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
    os.environ["BACKEND_URL"] = "http://127.0.0.1:8000"
    os.chdir(CUSTOMER_DIR)
    
    script_path = CUSTOMER_DIR / "app_unsupervised.py"
    with open(script_path) as f:
        code = compile(f.read(), str(script_path), "exec")
        exec(code, {"__name__": "__main__", "__file__": str(script_path)})

else:
    os.environ["BACKEND_URL"] = "http://127.0.0.1:8001"
    os.chdir(LOAN_DIR)
    
    script_path = LOAN_DIR / "app.py"
    with open(script_path) as f:
        code = compile(f.read(), str(script_path), "exec")
        exec(code, {"__name__": "__main__", "__file__": str(script_path)})
