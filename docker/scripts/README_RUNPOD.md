# RunPod Serverless Setup for Kokoro TTS

## Overview
This Docker setup is configured for RunPod Serverless deployment. It uses the `runpod.serverless.start()` handler to process requests at the `/run` endpoint.

## Build Instructions

### 1. Build the Docker Image
```bash
cd /Users/ellieschweiger/Kokoro-FastAPI
docker build -t kokoro-runpod -f docker/cpu/Dockerfile .
```

### 2. Deploy to RunPod
- Upload the image to a registry (Docker Hub, etc.)
- Create a new RunPod Serverless endpoint
- Configure the endpoint to use your image

## Request Format

Send POST requests to your RunPod endpoint with the following JSON structure:

```json
{
  "input": {
    "path": "dev/captioned_speech",
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
  }
}
```

## Supported Endpoints

### 1. `dev/captioned_speech`
- Returns audio with word-level timestamps
- Supports both streaming and non-streaming modes
- Returns base64-encoded audio data

### 2. `v1/audio/speech`
- OpenAI-compatible endpoint
- Returns base64-encoded audio data
- Standard TTS without timestamps

## Response Format

### Success Response
```json
{
  "status": "success",
  "audio": "base64_encoded_audio_data",
  "timestamps": [
    {
      "word": "The",
      "start": 0.250,
      "end": 0.325
    },
    ...
  ],
  "format": "mp3"
}
```

### Error Response
```json
{
  "error": "Error message description"
}
```

## Local Testing

You can test the handler locally by running:

```bash
docker run kokoro-runpod
```

Then send requests to test the functionality before deploying to RunPod.

## Notes

- The handler automatically initializes Kokoro TTS on first request
- Audio data is returned as hex-encoded strings for JSON compatibility
- The handler supports all the same features as the original Kokoro API
- No ports are exposed since RunPod handles the HTTP routing