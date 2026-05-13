"""
AssemblyAI Voice Agent API provider for end-to-end voice processing.

Wire protocol summary:
  - WebSocket: wss://agents.assemblyai.com/v1/ws
  - Bearer auth via Authorization header
  - Audio: PCM16, 24kHz, base64-encoded in {"type": "input.audio", "audio": ...}
  - Session config via {"type": "session.update", "session": {...}}
  - Tools registered as function-style in session.update
  - Tool responses sent via {"type": "tool.result", "call_id": ..., "result": ...}
"""

import asyncio
import base64
import json
import os
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional

import websockets
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from tau2.environment.tool import Tool
from tau2.utils.retry import websocket_retry
from tau2.voice.audio_native.assemblyai.events import (
    BaseAssemblyAIEvent,
    TimeoutEvent,
    UnknownEvent,
    parse_assemblyai_event,
)

load_dotenv()


# ---------------------------------------------------------------------------
# Provider-level constants (kept here rather than tau2/config.py to limit
# blast radius for the new provider).
# ---------------------------------------------------------------------------
DEFAULT_ASSEMBLYAI_WS_URL = "wss://agents.assemblyai.com/v1/ws"
DEFAULT_ASSEMBLYAI_MODEL = "assemblyai-voice-agent"
DEFAULT_ASSEMBLYAI_VOICE = "emma"
DEFAULT_ASSEMBLYAI_INPUT_SAMPLE_RATE = 24000
DEFAULT_ASSEMBLYAI_OUTPUT_SAMPLE_RATE = 24000
DEFAULT_ASSEMBLYAI_VAD_THRESHOLD = 0.5


class AssemblyAIVADMode(str, Enum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"


class AssemblyAIVADConfig(BaseModel):
    mode: AssemblyAIVADMode = AssemblyAIVADMode.AUTOMATIC
    vad_threshold: float = DEFAULT_ASSEMBLYAI_VAD_THRESHOLD


class AssemblyAIVoiceAgentProvider:
    """AssemblyAI Voice Agent API provider.

    Manages a single WebSocket session to AssemblyAI's full-duplex Voice
    Agent endpoint.
    """

    BASE_URL = DEFAULT_ASSEMBLYAI_WS_URL
    DEFAULT_MODEL = DEFAULT_ASSEMBLYAI_MODEL

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        greeting: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "AssemblyAI API key not provided. Set ASSEMBLYAI_API_KEY env var."
            )

        self.model = model or self.DEFAULT_MODEL
        self.voice = voice or DEFAULT_ASSEMBLYAI_VOICE
        self.greeting = greeting
        self.reasoning_effort = reasoning_effort
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.session_id: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        if self.ws is None:
            return False
        from websockets.protocol import State

        return self.ws.state == State.OPEN

    @websocket_retry
    async def connect(self) -> None:
        if self.is_connected:
            return

        headers = {"Authorization": f"Bearer {self.api_key}"}
        self.ws = await websockets.connect(self.BASE_URL, additional_headers=headers)
        logger.info("AssemblyAI Voice Agent: WebSocket opened")

    async def disconnect(self) -> None:
        if self.ws:
            logger.info("AssemblyAI Voice Agent: closing WebSocket")
            await self.ws.close()
            self.ws = None

    def _format_tools_for_api(self, tools: List[Tool]) -> List[Dict]:
        formatted = []
        for tool in tools:
            schema = tool.openai_schema
            fn = schema["function"]
            formatted.append(
                {
                    "type": "function",
                    "name": fn["name"],
                    "description": fn["description"],
                    "parameters": fn["parameters"],
                }
            )
        return formatted

    async def configure_session(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: AssemblyAIVADConfig,
        modality: str = "audio",
    ) -> None:
        if not self.is_connected:
            raise RuntimeError("Not connected to AssemblyAI Voice Agent")

        session: Dict = {
            "system_prompt": system_prompt,
            "input": {
                "format": {"encoding": "audio/pcm"},
                "turn_detection": {"vad_threshold": vad_config.vad_threshold},
            },
            "output": {
                "voice": self.voice,
                "format": {"encoding": "audio/pcm"},
            },
            "tools": self._format_tools_for_api(tools),
        }
        if self.greeting:
            session["greeting"] = self.greeting

        await self.ws.send(json.dumps({"type": "session.update", "session": session}))

        # AssemblyAI emits session.ready (with session_id) and/or session.updated
        # after a successful session.update. Treat either as success.
        while True:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=15.0)
            data = json.loads(raw)
            t = data.get("type")
            if t == "session.ready":
                self.session_id = data.get("session_id")
                logger.info(
                    f"AssemblyAI Voice Agent: session ready "
                    f"(session_id={self.session_id})"
                )
                return
            if t == "session.updated":
                logger.info("AssemblyAI Voice Agent: session updated")
                return
            if t == "session.error":
                raise RuntimeError(
                    f"AssemblyAI session.update failed: "
                    f"{data.get('message', 'unknown')}"
                )

    async def send_audio(self, audio_data: bytes) -> None:
        if not self.is_connected:
            raise RuntimeError("Not connected to AssemblyAI Voice Agent")
        if not audio_data:
            return
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
        await self.ws.send(json.dumps({"type": "input.audio", "audio": audio_b64}))

    async def send_tool_result(
        self,
        call_id: str,
        result: str,
        request_response: bool = True,  # unused; AssemblyAI continues on its own
    ) -> None:
        if not self.is_connected:
            raise RuntimeError("Not connected to AssemblyAI Voice Agent")
        message = {"type": "tool.result", "call_id": call_id, "result": result}
        await self.ws.send(json.dumps(message))

    async def receive_events(
        self,
    ) -> AsyncGenerator[BaseAssemblyAIEvent, None]:
        if not self.is_connected:
            raise RuntimeError("Not connected to AssemblyAI Voice Agent")

        while self.is_connected:
            try:
                raw_message = await asyncio.wait_for(self.ws.recv(), timeout=0.01)
                data = json.loads(raw_message)
                yield parse_assemblyai_event(data)
            except asyncio.TimeoutError:
                yield TimeoutEvent(type="timeout")
            except websockets.ConnectionClosed as e:
                logger.error(
                    f"AssemblyAI Voice Agent: WS closed (code={e.code}, "
                    f"reason='{e.reason or 'no reason provided'}')"
                )
                raise RuntimeError(
                    f"AssemblyAI WS closed (code={e.code}, "
                    f"reason='{e.reason or 'no reason provided'}')"
                ) from e
            except Exception as e:
                logger.error(
                    f"AssemblyAI Voice Agent: receive error "
                    f"{type(e).__name__}: {e}"
                )
                yield UnknownEvent(type="error", raw={"error": str(e)})

    async def receive_events_for_duration(
        self, duration_seconds: float
    ) -> List[BaseAssemblyAIEvent]:
        events: List[BaseAssemblyAIEvent] = []
        end_time = asyncio.get_event_loop().time() + duration_seconds
        async for event in self.receive_events():
            if not isinstance(event, TimeoutEvent):
                events.append(event)
            if asyncio.get_event_loop().time() >= end_time:
                break
        return events
