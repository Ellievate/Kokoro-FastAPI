#!/usr/bin/env python3
"""
Local test script for RunPod handler
"""

import json
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runpod_handler import handler

def test_handler():
    """Test the RunPod handler locally"""
    
    # Test with RunPod format
    test_event = {
        "input": {
            "path": "dev/captioned_speech",
            "model": "kokoro",
            "input": "Hello world test",
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
    
    print("Testing RunPod handler...")
    print(f"Input: {json.dumps(test_event, indent=2)}")
    
    result = handler(test_event)
    
    print(f"Result: {json.dumps(result, indent=2)}")
    
    return result

if __name__ == "__main__":
    test_handler()