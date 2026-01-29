"""
Quick test to verify Gemini setup
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Test API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY not found in .env file!")
    exit(1)

print(f"✓ API Key found: {api_key[:20]}...")

# Test API call
try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content("Say 'Hello World'")
    print(f"✓ API Response: {response.text}")
    print("\n🎉 Setup successful! You're ready to use PageIndex with Gemini.")
except Exception as e:
    print(f"❌ API Error: {e}")
