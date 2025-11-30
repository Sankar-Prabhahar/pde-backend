# server.py
import os
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# import everything you defined in A.py
from A import commandcore_runner, run_session, APP_NAME, USER_ID, session_service

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

async def run_single_turn(message: str, session_id: str) -> str:
    # minimal version of run_session that returns text instead of printing
    from google.genai import types

    try:
        try:
            session = await session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id,
            )
        except Exception:
            session = await session_service.get_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id,
            )

        content = types.Content(
            role="user",
            parts=[types.Part(text=message)],
        )

        full_text = ""
        async for event in commandcore_runner.run_async(
            user_id=USER_ID,
            session_id=session.id,
            new_message=content,
        ):
            if event.content and event.content.parts and event.content.parts[0].text:
                full_text += event.content.parts[0].text

        return full_text.strip() or "[No response text]"
    except Exception as e:
        return f"[Server error: {e}]"

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    return ChatResponse(reply="backend ok")
