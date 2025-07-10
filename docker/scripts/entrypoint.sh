#!/bin/bash
set -e

if [ "$DOWNLOAD_MODEL" = "true" ]; then
    python download_model.py --output api/src/models/v1_0
fi

# Start Kokoro API in the background
echo "Starting Kokoro API on port 8880..."
uv run --extra $DEVICE --no-sync python -m uvicorn api.src.main:app --host 0.0.0.0 --port 8880 --log-level debug &

# Wait a moment for Kokoro to start
sleep 5

# Start the API handler
echo "Starting API Handler on port 8881..."
exec python api_handler.py