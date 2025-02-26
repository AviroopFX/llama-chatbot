#!/bin/bash

# Create necessary directories
mkdir -p logs data models uploads

# Start the FastAPI server
uvicorn app_sqlite:app --host 0.0.0.0 --port $PORT
