#!/usr/bin/env python3
"""
API Translation Layer for Kokoro FastAPI
Provides a unified /run endpoint that accepts JSON requests and translates them to the Kokoro API
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic model for request validation
class RunRequest(BaseModel):
    path: str = Field(..., description="API endpoint path (e.g., '/v1/captioned-speech')")
    model: str = Field(default="kokoro", description="Model to use")
    input: str = Field(..., description="Text input for TTS")
    voice: str = Field(..., description="Voice to use")
    response_format: str = Field(default="mp3", description="Audio format")
    speed: float = Field(default=1.0, description="Speech speed")
    stream: bool = Field(default=False, description="Enable streaming")
    return_timestamps: bool = Field(default=True, description="Return timestamps")
    return_download_link: bool = Field(default=False, description="Return download link")
    lang_code: str = Field(default="a", description="Language code")
    volume_multiplier: float = Field(default=1.0, description="Volume multiplier")
    normalization_options: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Text normalization options"
    )

class APIHandler:
    def __init__(self, kokoro_host: str = "localhost", kokoro_port: int = 8880):
        self.kokoro_base_url = f"http://{kokoro_host}:{kokoro_port}"
        self.app = FastAPI(
            title="Kokoro API Handler",
            description="Unified API endpoint for Kokoro TTS",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.post("/run")
        async def run_endpoint(request: RunRequest):
            """
            Unified endpoint that accepts JSON requests and translates them to Kokoro API calls
            """
            try:
                # Extract the path and build the full URL
                path = request.path.lstrip('/')
                kokoro_url = f"{self.kokoro_base_url}/{path}"
                
                # Prepare the request payload for Kokoro API
                payload = {
                    "model": request.model,
                    "input": request.input,
                    "voice": request.voice,
                    "response_format": request.response_format,
                    "speed": request.speed,
                    "stream": request.stream,
                    "return_timestamps": request.return_timestamps,
                    "return_download_link": request.return_download_link,
                    "lang_code": request.lang_code,
                    "volume_multiplier": request.volume_multiplier,
                }
                
                # Add normalization options if provided
                if request.normalization_options:
                    payload.update(request.normalization_options)
                
                # Make request to Kokoro API
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        kokoro_url,
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        if response.status == 200:
                            # Handle different response types
                            content_type = response.headers.get('content-type', '')
                            
                            if 'application/json' in content_type:
                                result = await response.json()
                                return result
                            else:
                                # Handle binary audio response
                                content = await response.read()
                                return {
                                    "status": "success",
                                    "content_type": content_type,
                                    "data": content.hex()  # Return as hex string for JSON compatibility
                                }
                        else:
                            error_text = await response.text()
                            logger.error(f"Kokoro API error: {response.status} - {error_text}")
                            raise HTTPException(
                                status_code=response.status,
                                detail=f"Kokoro API error: {error_text}"
                            )
                            
            except aiohttp.ClientError as e:
                logger.error(f"Connection error to Kokoro API: {e}")
                raise HTTPException(
                    status_code=503,
                    detail="Unable to connect to Kokoro API"
                )
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="Internal server error"
                )
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            try:
                # Check if Kokoro API is accessible
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.kokoro_base_url}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        kokoro_healthy = response.status == 200
                        
                return {
                    "status": "healthy",
                    "kokoro_api": "connected" if kokoro_healthy else "disconnected"
                }
            except Exception:
                return {
                    "status": "healthy",
                    "kokoro_api": "disconnected"
                }
        
        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "message": "Kokoro API Handler",
                "endpoints": {
                    "run": "POST /run - Unified API endpoint",
                    "health": "GET /health - Health check"
                }
            }

def main():
    """Main entry point"""
    handler = APIHandler()
    
    # Wait for Kokoro API to be ready
    async def wait_for_kokoro():
        logger.info("Waiting for Kokoro API to be ready...")
        max_retries = 30
        for i in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{handler.kokoro_base_url}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            logger.info("Kokoro API is ready!")
                            return
            except Exception as e:
                if i == max_retries - 1:
                    logger.error(f"Kokoro API not ready after {max_retries} attempts: {e}")
                else:
                    logger.info(f"Kokoro API not ready, retrying in 2 seconds... ({i+1}/{max_retries})")
                    await asyncio.sleep(2)
    
    # Run the waiting function
    asyncio.run(wait_for_kokoro())
    
    # Start the handler API
    logger.info("Starting API Handler on port 8881...")
    uvicorn.run(
        handler.app,
        host="0.0.0.0",
        port=8881,
        log_level="info"
    )

if __name__ == "__main__":
    main()