#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app_sqlite:app --reload --host 0.0.0.0 --port 8000
