#!/usr/bin/env python3
"""
Test script for the unified /run endpoint
"""

import asyncio
import json
import aiohttp

async def test_run_endpoint():
    """Test the unified /run endpoint"""
    
    # Test payload matching your requirements
    test_payload = {
        "path": "dev/captioned-speech",
        "model": "kokoro",
        "input": "The quick brown fox jumped over the lazy dog.",
        "voice": "af_heart",
        "response_format": "mp3",
        "speed": 1,
        "stream": False,
        "return_timestamps": True,
        "return_download_link": False,
        "lang_code": "a",
        "volume_multiplier": 1,
        "normalization_options": {
            "normalize": True,
            "unit_normalization": False,
            "url_normalization": True,
            "email_normalization": True,
            "optional_pluralization_normalization": True,
            "phone_normalization": True,
            "replace_remaining_symbols": True
        }
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            print("Testing /run endpoint...")
            print(f"Payload: {json.dumps(test_payload, indent=2)}")
            
            async with session.post(
                "http://localhost:8881/run",
                json=test_payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                print(f"Status: {response.status}")
                print(f"Headers: {dict(response.headers)}")
                
                if response.status == 200:
                    result = await response.json()
                    print(f"Response: {json.dumps(result, indent=2)}")
                    print("✅ Test passed!")
                else:
                    error_text = await response.text()
                    print(f"❌ Test failed: {error_text}")
                    
    except aiohttp.ClientError as e:
        print(f"❌ Connection error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

async def test_health_endpoint():
    """Test the health endpoint"""
    try:
        async with aiohttp.ClientSession() as session:
            print("\nTesting /health endpoint...")
            
            async with session.get("http://localhost:8881/health") as response:
                print(f"Status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    print(f"Response: {json.dumps(result, indent=2)}")
                    print("✅ Health check passed!")
                else:
                    error_text = await response.text()
                    print(f"❌ Health check failed: {error_text}")
                    
    except aiohttp.ClientError as e:
        print(f"❌ Connection error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

async def main():
    """Main test function"""
    await test_health_endpoint()
    await test_run_endpoint()

if __name__ == "__main__":
    asyncio.run(main())