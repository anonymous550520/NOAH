import os
import asyncio
import importlib

from openai import AsyncOpenAI
from google import genai
from google.genai import types
# OpenAI client (async)



async def _call_openai(prompt: str, model: str, temperature: float) -> str:
    """OpenAI chat completion (async)."""
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = await openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


async def _call_gemini(prompt: str, model: str, temperature: float) -> str:
    """Gemini text generation via google-generativeai (sync wrapped for async)."""
    def _sync_call() -> str:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature)
        )
        text = resp.text
        if isinstance(text, str) and text.strip():
            return text.strip()
        # Fallback: string representation
        return str(resp).strip()

    return await asyncio.to_thread(_sync_call)


async def call_judge_model(prompt, model="gpt-4o", temperature=0.0):
    """Unified entry: routes to OpenAI or Gemini based on model name. Returns plain text."""
    try:
        if str(model).lower().startswith("gemini"):
            return await _call_gemini(prompt=prompt, model=model, temperature=temperature)
        return await _call_openai(prompt=prompt, model=model, temperature=temperature)
    except Exception as e:
        print(f"[call_judge_model] Error: {e}")
        raise