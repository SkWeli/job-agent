import os
import sys

from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY", "")

if not api_key.startswith("gsk_"):
    sys.exit("GROQ_API_KEY is missing or invalid. Check the .env file in the project root.")

print(f"GROQ_API_KEY loaded (length {len(api_key)}).")