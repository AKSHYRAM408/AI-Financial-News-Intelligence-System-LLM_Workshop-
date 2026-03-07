#!/bin/bash
set -e

# ──────────────────────────────────────────────────────────────
# EC2 User Data — Install Python, clone repo, run Streamlit
# ──────────────────────────────────────────────────────────────

# Update system packages
dnf update -y

# Install Python 3.11 and Git
dnf install -y python3.11 python3.11-pip git

# Create app directory
APP_DIR="/opt/stock-ai"
mkdir -p $APP_DIR

# Clone the repo from GitHub
git clone https://github.com/AKSHYRAM408/AI-Financial-News-Intelligence-System-LLM_Workshop-.git $APP_DIR/app

cd $APP_DIR/app

# Create .env file with API keys (injected by Terraform)
cat > .env << 'ENVEOF'
# Mistral API configuration
MISTRAL_API_KEY=${mistral_api_key}
MISTRAL_MODEL=mistral-small-latest

# NEWSDATA.io API key
NEWSDATA_API_KEY=${newsdata_api_key}

# HuggingFace API key (for BERT embeddings)
HF_API_KEY=${hf_api_key}
HF_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENVEOF

# Install Python dependencies
python3.11 -m pip install --upgrade pip
python3.11 -m pip install -r requirements.txt

# Create a systemd service for Streamlit
cat > /etc/systemd/system/streamlit.service << 'SVCEOF'
[Unit]
Description=Streamlit Stock AI App
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/stock-ai/app
ExecStart=/usr/local/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
Restart=on-failure
RestartSec=5
Environment=PATH=/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=multi-user.target
SVCEOF

# Start the service
systemctl daemon-reload
systemctl enable streamlit
systemctl start streamlit
