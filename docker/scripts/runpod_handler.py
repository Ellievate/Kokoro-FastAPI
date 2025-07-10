#!/usr/bin/env python3
"""
RunPod Serverless Handler for Kokoro TTS
Handles requests at /run endpoint and translates them to Kokoro API
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

import aiohttp
import runpod
from pydantic import BaseModel, Field, ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the API directory to the path so we can import Kokoro modules
sys.path.insert(0, '/app/api')

# Import Kokoro services
from src.services.tts_service import TTSService
from src.inference.model_manager import get_manager
from src.inference.voice_manager import get_manager as get_voice_manager
from src.services.temp_manager import cleanup_temp_files
from src.routers.openai_compatible import process_and_validate_voices
from src.services.streaming_audio_writer import StreamingAudioWriter
from src.services.audio import AudioService
from src.inference.base import AudioChunk
import numpy as np

# Pydantic model for request validation
class RunRequest(BaseModel):
    path: str = Field(..., description="API endpoint path (e.g., 'dev/captioned_speech')")
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

# Global TTS service instance
_tts_service = None
_initialized = False

async def initialize_kokoro():
    """Initialize Kokoro TTS service"""
    global _tts_service, _initialized
    
    if _initialized:
        return _tts_service
    
    try:
        logger.info("Initializing Kokoro TTS service...")
        
        # Clean old temp files
        await cleanup_temp_files()
        
        # Initialize managers
        model_manager = await get_manager()
        voice_manager = await get_voice_manager()
        
        # Initialize model with warmup
        device, model, voicepack_count = await model_manager.initialize_with_warmup(voice_manager)
        
        # Create TTS service
        _tts_service = await TTSService.create()
        
        logger.info(f"Kokoro initialized on {device} with {voicepack_count} voice packs")
        _initialized = True
        
        return _tts_service
        
    except Exception as e:
        logger.error(f"Failed to initialize Kokoro: {e}")
        raise

async def process_kokoro_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a request using Kokoro TTS service"""
    try:
        logger.info(f"Starting to process request: {request_data}")
        
        # Validate request
        try:
            request = RunRequest(**request_data)
            logger.info(f"Request validation successful: {request.path}")
        except ValidationError as e:
            logger.error(f"Request validation failed: {str(e)}")
            return {"error": f"Invalid request: {str(e)}"}
        
        # Get TTS service
        logger.info("Initializing Kokoro TTS service...")
        tts_service = await initialize_kokoro()
        logger.info("Kokoro TTS service initialized successfully")
        
        # Process and validate voice
        logger.info(f"Processing voice: {request.voice}")
        voice_name = await process_and_validate_voices(request.voice, tts_service)
        logger.info(f"Voice validation successful: {voice_name}")
        
        # Create audio writer
        logger.info(f"Creating audio writer for format: {request.response_format}")
        writer = StreamingAudioWriter(request.response_format, sample_rate=24000)
        
        # Handle different endpoint paths
        if request.path == "dev/captioned_speech":
            # Generate audio with timestamps
            if request.stream:
                # For streaming, we'll collect all chunks and return them
                chunks = []
                async for chunk_data in tts_service.generate_audio_stream(
                    text=request.input,
                    voice=voice_name,
                    writer=writer,
                    speed=request.speed,
                    output_format=request.response_format,
                    lang_code=request.lang_code,
                    volume_multiplier=request.volume_multiplier,
                    normalization_options=request.normalization_options,
                    return_timestamps=request.return_timestamps,
                ):
                    if chunk_data.output:
                        chunks.append({
                            "audio": chunk_data.output.hex(),
                            "timestamps": chunk_data.word_timestamps or []
                        })
                
                return {
                    "status": "success",
                    "chunks": chunks,
                    "format": request.response_format
                }
            else:
                # Non-streaming mode
                audio_data = await tts_service.generate_audio(
                    text=request.input,
                    voice=voice_name,
                    writer=writer,
                    speed=request.speed,
                    return_timestamps=request.return_timestamps,
                    volume_multiplier=request.volume_multiplier,
                    normalization_options=request.normalization_options,
                    lang_code=request.lang_code,
                )
                
                # Convert to requested format
                converted_audio = await AudioService.convert_audio(
                    audio_data,
                    request.response_format,
                    writer,
                    is_last_chunk=False,
                    trim_audio=False,
                )
                
                # Finalize
                final = await AudioService.convert_audio(
                    AudioChunk(np.array([], dtype=np.int16)),
                    request.response_format,
                    writer,
                    is_last_chunk=True,
                )
                
                output = converted_audio.output + final.output
                writer.close()
                
                return {
                    "status": "success",
                    "audio": output.hex(),
                    "timestamps": audio_data.word_timestamps or [],
                    "format": request.response_format
                }
                
        elif request.path == "v1/audio/speech":
            # OpenAI compatible endpoint
            audio_data = await tts_service.generate_audio(
                text=request.input,
                voice=voice_name,
                writer=writer,
                speed=request.speed,
                volume_multiplier=request.volume_multiplier,
                normalization_options=request.normalization_options,
                lang_code=request.lang_code,
            )
            
            # Convert to requested format
            converted_audio = await AudioService.convert_audio(
                audio_data,
                request.response_format,
                writer,
                is_last_chunk=False,
                trim_audio=False,
            )
            
            # Finalize
            final = await AudioService.convert_audio(
                AudioChunk(np.array([], dtype=np.int16)),
                request.response_format,
                writer,
                is_last_chunk=True,
            )
            
            output = converted_audio.output + final.output
            writer.close()
            
            return {
                "status": "success",
                "audio": output.hex(),
                "format": request.response_format
            }
        else:
            return {"error": f"Unsupported endpoint: {request.path}"}
            
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return {"error": str(e)}

def handler(event):
    """
    RunPod serverless handler function
    This function is called by RunPod for each request
    """
    try:
        logger.info(f"Received event: {json.dumps(event, indent=2)}")
        
        # Extract the request data - handle both RunPod format and direct format
        if 'input' in event:
            # RunPod format: {"input": {"path": "...", "model": "...", ...}}
            input_data = event['input']
            logger.info("Using RunPod input format")
        else:
            # Direct format: {"path": "...", "model": "...", ...}
            input_data = event
            logger.info("Using direct input format")
        
        logger.info(f"Processing input data: {json.dumps(input_data, indent=2)}")
        
        # Run the async processing with timeout
        try:
            result = asyncio.wait_for(
                process_kokoro_request(input_data),
                timeout=300  # 5 minute timeout
            )
            result = asyncio.run(result)
            logger.info("Processing completed successfully")
            return result
        except asyncio.TimeoutError:
            logger.error("Request timed out after 5 minutes")
            return {"error": "Request timed out"}
        
    except Exception as e:
        logger.error(f"Handler error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    # Check if running in test mode
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode - run a simple test
        logger.info("Running in test mode...")
        test_event = {
            "input": {
                "path": "dev/captioned_speech",
                "model": "kokoro",
                "input": "Hello world",
                "voice": "af_heart",
                "response_format": "mp3",
                "speed": 1,
                "stream": False,
                "return_timestamps": True,
                "return_download_link": False,
                "lang_code": "a",
                "volume_multiplier": 1
            }
        }
        result = handler(test_event)
        logger.info(f"Test result: {result}")
    else:
        # Initialize RunPod serverless
        logger.info("Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": handler})