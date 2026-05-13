"""Pydantic models for AssemblyAI Voice Agent API events."""

from typing import Any, Literal, Optional

from pydantic import BaseModel


class BaseAssemblyAIEvent(BaseModel):
    type: str


class SessionReadyEvent(BaseAssemblyAIEvent):
    type: Literal["session.ready"]
    session_id: Optional[str] = None


class SessionUpdatedEvent(BaseAssemblyAIEvent):
    type: Literal["session.updated"]


class SessionErrorEvent(BaseAssemblyAIEvent):
    type: Literal["session.error"]
    code: Optional[str] = None
    message: Optional[str] = None
    timestamp: Optional[str] = None
    param: Optional[str] = None


class SpeechStartedEvent(BaseAssemblyAIEvent):
    type: Literal["input.speech.started"]


class SpeechStoppedEvent(BaseAssemblyAIEvent):
    type: Literal["input.speech.stopped"]


class UserTranscriptDeltaEvent(BaseAssemblyAIEvent):
    type: Literal["transcript.user.delta"]
    text: str = ""


class UserTranscriptEvent(BaseAssemblyAIEvent):
    type: Literal["transcript.user"]
    text: str = ""
    item_id: Optional[str] = None


class ReplyStartedEvent(BaseAssemblyAIEvent):
    type: Literal["reply.started"]
    reply_id: Optional[str] = None


class ReplyAudioEvent(BaseAssemblyAIEvent):
    type: Literal["reply.audio"]
    data: str = ""  # base64 PCM16


class AgentTranscriptEvent(BaseAssemblyAIEvent):
    type: Literal["transcript.agent"]
    text: str = ""
    reply_id: Optional[str] = None
    item_id: Optional[str] = None
    interrupted: bool = False


class ReplyDoneEvent(BaseAssemblyAIEvent):
    type: Literal["reply.done"]
    status: Optional[str] = None  # "interrupted" if interrupted
    reply_id: Optional[str] = None


class ToolCallEvent(BaseAssemblyAIEvent):
    type: Literal["tool.call"]
    call_id: str
    name: str
    arguments: Any = None  # already-parsed dict in most cases


class TimeoutEvent(BaseAssemblyAIEvent):
    type: Literal["timeout"]


class UnknownEvent(BaseAssemblyAIEvent):
    type: str
    raw: Optional[dict] = None


_EVENT_TYPE_MAP: dict[str, type[BaseAssemblyAIEvent]] = {
    "session.ready": SessionReadyEvent,
    "session.updated": SessionUpdatedEvent,
    "session.error": SessionErrorEvent,
    "input.speech.started": SpeechStartedEvent,
    "input.speech.stopped": SpeechStoppedEvent,
    "transcript.user.delta": UserTranscriptDeltaEvent,
    "transcript.user": UserTranscriptEvent,
    "reply.started": ReplyStartedEvent,
    "reply.audio": ReplyAudioEvent,
    "transcript.agent": AgentTranscriptEvent,
    "reply.done": ReplyDoneEvent,
    "tool.call": ToolCallEvent,
    "timeout": TimeoutEvent,
}


def parse_assemblyai_event(raw: dict) -> BaseAssemblyAIEvent:
    event_type = raw.get("type", "unknown")
    cls = _EVENT_TYPE_MAP.get(event_type)
    if cls is None:
        return UnknownEvent(type=event_type, raw=raw)
    try:
        return cls.model_validate(raw)
    except Exception:
        return UnknownEvent(type=event_type, raw=raw)
