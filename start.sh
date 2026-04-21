#!/bin/bash

# Activate venv
source .venv/bin/activate

# Check API key
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "❌ GOOGLE_API_KEY not set"
    echo "Run: export GOOGLE_API_KEY=your-key-here"
    exit 1
fi

# Start API server
echo "🚀 Starting Memvid Knowledge Base API..."
echo "📖 Chatbot UI: open chatbot.html in your browser"
echo "📡 API docs:   http://localhost:8000/docs"
echo ""
python app.py
