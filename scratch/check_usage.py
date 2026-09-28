import asyncio
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    client = genai.Client()
    response = await client.aio.models.generate_content(
        model="gemini-3.5-flash",
        contents="Think carefully and tell me what is 2+2.",
        config=types.GenerateContentConfig(
            temperature=0.7,
        )
    )
    print("Candidates:", response.candidates)
    print("Usage Metadata:", dir(response.usage_metadata))
    print(response.usage_metadata)

if __name__ == "__main__":
    asyncio.run(main())
