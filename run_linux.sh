##!/bin/bash

# --- Color Definitions ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}[*] Starting Car-Manual-RAG System...${NC}"

# 1. Start Backend (FastAPI)
echo -e "${GREEN}[*] Launching FastAPI backend service in the background...${NC}"
# Redirecting logs to backend.log for debugging
python backend/main.py > backend.log 2>&1 &
BACKEND_PID=$!

# Ensure backend process is killed when script exits
trap "echo -e '${BLUE}\n[*] Shutting down backend and exiting...${NC}'; kill -9 $BACKEND_PID; exit" SIGINT SIGTERM EXIT

# Wait for model initialization
echo -e "${BLUE}[*] Waiting for models to load (this may take a few seconds)...${NC}"
sleep 5

# 2. Start Frontend (Streamlit)
echo -e "${GREEN}[*] Launching Streamlit frontend interface...${NC}"
streamlit run frontend/app.py