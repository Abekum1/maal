# config.py

# Telegram Config
##TELEGRAM_TOKEN = "5150870610:AAE_yms2A7okSsAQUNfCIwn00A6fSJnulYU" # Your actual token

# This is your correct Telegram Chat ID for a private chat with your bot
##TELEGRAM_CHAT_ID = '1808088885'

# config.py
import os

# Telegram Config
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # Read from environment
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # Read from environment

# Optional: Debug to confirm values are loaded (remove in production)
if not TELEGRAM_TOKEN:
    print("❌ TELEGRAM_TOKEN is not set!")
else:
    print("✅ TELEGRAM_TOKEN loaded.")

if not TELEGRAM_CHAT_ID:
    print("❌ TELEGRAM_CHAT_ID is not set!")
else:
    print(f"✅ TELEGRAM_CHAT_ID loaded: {TELEGRAM_CHAT_ID}")
