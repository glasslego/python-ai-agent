"""
https://aistudio.google.com/apikey

google-genai
"""

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
)


def basic_chat_example():
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents="파이썬으로 간단한 계산기 만들어줘"
    )
    print(response.text)


if __name__ == "__main__":
    basic_chat_example()
