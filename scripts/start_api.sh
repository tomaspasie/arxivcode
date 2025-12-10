#!/bin/bash
# Start the ArxivCode Explanation API server

echo "Starting ArxivCode Explanation API..."
echo "=========================================="

# Load from .env if it exists and OPENAI_API_KEY is not set
if [ -z "$OPENAI_API_KEY" ] && [ -f .env ]; then
    echo "Loading environment from .env..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY environment variable not set"
    echo ""
    echo "Option 1: Add to .env file (recommended)"
    echo "  1. Copy .env.example to .env"
    echo "  2. Edit .env and add: OPENAI_API_KEY=your-key-here"
    echo ""
    echo "Option 2: Export directly"
    echo "  export OPENAI_API_KEY='your-key-here'"
    echo ""
    exit 1
fi

echo "✅ OPENAI_API_KEY is set"
echo ""
echo "Starting server on http://localhost:8000"
echo "API docs available at http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Start the server
python3 -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
