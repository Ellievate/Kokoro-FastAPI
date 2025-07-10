#!/bin/bash
set -e

if [ "$DOWNLOAD_MODEL" = "true" ]; then
    python download_model.py --output api/src/models/v1_0
fi

# Start RunPod serverless handler
echo "Starting RunPod serverless handler..."
exec python runpod_handler.py