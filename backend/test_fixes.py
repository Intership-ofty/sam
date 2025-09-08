#!/usr/bin/env python3
"""
Test script to verify error fixes
"""

import asyncio
import aiohttp
import json

async def test_health_endpoint():
    """Test the health endpoint"""
    try:
        async with aiohttp.ClientSession() as session:
            # Test direct health endpoint
            async with session.get('http://localhost:8000/health') as response:
                print(f"Health endpoint status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Health check passed: {data.get('status', 'unknown')}")
                else:
                    print(f"❌ Health check failed: {response.status}")
                    
            # Test health with trailing slash
            async with session.get('http://localhost:8000/health/') as response:
                print(f"Health/ endpoint status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Health/ check passed: {data.get('status', 'unknown')}")
                else:
                    print(f"❌ Health/ check failed: {response.status}")
                    
    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")

async def test_auth_config():
    """Test the auth config endpoint"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/api/v1/auth/config') as response:
                print(f"Auth config status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Auth config passed: {data}")
                else:
                    text = await response.text()
                    print(f"❌ Auth config failed: {response.status} - {text}")
                    
    except Exception as e:
        print(f"❌ Auth config test failed: {e}")

async def test_metrics_endpoint():
    """Test the metrics endpoint"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/metrics') as response:
                print(f"Metrics endpoint status: {response.status}")
                if response.status == 200:
                    print("✅ Metrics endpoint working")
                else:
                    print(f"❌ Metrics endpoint failed: {response.status}")
                    
    except Exception as e:
        print(f"❌ Metrics endpoint test failed: {e}")

async def main():
    """Run all tests"""
    print("🔍 Testing error fixes...")
    print("=" * 50)
    
    await test_health_endpoint()
    print()
    
    await test_auth_config()
    print()
    
    await test_metrics_endpoint()
    print()
    
    print("=" * 50)
    print("✅ All tests completed")

if __name__ == "__main__":
    asyncio.run(main())
