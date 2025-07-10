# Running Kokoro with Unified API Handler

## Build and Run Instructions

### 1. Build the Docker Image
```bash
cd /Users/ellieschweiger/Kokoro-FastAPI
docker build -t kokoro-unified -f docker/cpu/Dockerfile .
```

### 2. Run the Container
```bash
docker run -p 8881:8881 -p 8880:8880 kokoro-unified
```

### 3. Wait for Startup
The container will:
- Start Kokoro API on port 8880 (internal)
- Start API Handler on port 8881 (your main endpoint)
- You'll see logs showing both services starting

### 4. Test with curl

#### Health Check
```bash
curl -X GET http://localhost:8881/health
```

#### Test the /run endpoint
```bash
curl -X POST http://localhost:8881/run \
  -H "Content-Type: application/json" \
  -d '{
    "path": "v1/dev/captioned-speech",
    "model": "kokoro",
    "input": "The quick brown fox jumped over the lazy dog.",
    "voice": "af_heart",
    "response_format": "mp3",
    "speed": 1,
    "stream": false,
    "return_timestamps": true,
    "return_download_link": false,
    "lang_code": "a",
    "volume_multiplier": 1,
    "normalization_options": {
      "normalize": true,
      "unit_normalization": false,
      "url_normalization": true,
      "email_normalization": true,
      "optional_pluralization_normalization": true,
      "phone_normalization": true,
      "replace_remaining_symbols": true
    }
  }'
```

## Notes
- Main endpoint: `http://localhost:8881/run`
- Health check: `http://localhost:8881/health`
- Direct Kokoro API (if needed): `http://localhost:8880/v1/`
- The handler waits for Kokoro to be ready before starting
- All requests go through the unified `/run` endpoint which translates to the appropriate Kokoro API calls