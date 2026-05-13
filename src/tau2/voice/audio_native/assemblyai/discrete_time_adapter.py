"""Discrete-time adapter for AssemblyAI's Voice Agent API."""

import asyncio
import base64
import json
import uuid
from typing import Any, List, Optional, Tuple

from loguru import logger

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
)
from tau2.data_model.message import ToolCall
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.assemblyai.events import (
    AgentTranscriptEvent,
    ReplyAudioEvent,
    ReplyDoneEvent,
    ReplyStartedEvent,
    SessionErrorEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    ToolCallEvent,
    UserTranscriptDeltaEvent,
    UserTranscriptEvent,
)
from tau2.voice.audio_native.assemblyai.provider import (
    DEFAULT_ASSEMBLYAI_INPUT_SAMPLE_RATE,
    DEFAULT_ASSEMBLYAI_OUTPUT_SAMPLE_RATE,
    AssemblyAIVADConfig,
    AssemblyAIVoiceAgentProvider,
)
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.audio_converter import StreamingTelephonyConverter
from tau2.voice.audio_native.tick_result import TickResult, UtteranceTranscript


ASSEMBLYAI_OUTPUT_BYTES_PER_SECOND = DEFAULT_ASSEMBLYAI_OUTPUT_SAMPLE_RATE * 2


class DiscreteTimeAssemblyAIAdapter(DiscreteTimeAdapter):
    """Tick-based adapter for AssemblyAI Voice Agent.

    External format is telephony (8kHz μ-law). Internally:
      - Convert user audio 8kHz μ-law → 24kHz PCM16 before sending.
      - Convert agent audio 24kHz PCM16 → 8kHz μ-law before storing in tick.
    """

    def __init__(
        self,
        tick_duration_ms: int,
        send_audio_instant: bool = False,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        provider: Optional[AssemblyAIVoiceAgentProvider] = None,
    ):
        super().__init__(tick_duration_ms, send_audio_instant=send_audio_instant)

        # 20ms-of-PCM16-at-24kHz chunk size for VoIP-style sends
        self._chunk_size = int(
            DEFAULT_ASSEMBLYAI_INPUT_SAMPLE_RATE * 2 * self._voip_interval_ms / 1000
        )

        if model is not None and provider is not None:
            raise ValueError("model and provider cannot be provided together")

        self.model = model
        self.reasoning_effort = reasoning_effort
        self._provider = provider
        self._owns_provider = provider is None

        self._audio_converter = StreamingTelephonyConverter(
            input_sample_rate=DEFAULT_ASSEMBLYAI_INPUT_SAMPLE_RATE,
            output_sample_rate=DEFAULT_ASSEMBLYAI_OUTPUT_SAMPLE_RATE,
        )

        self._bg_loop = BackgroundAsyncLoop()
        self._connected = False

        # AssemblyAI sends a single transcript.agent event with the full text
        # for an item — but audio arrives via reply.audio events that don't
        # carry an item_id. We synthesize an item_id per reply on reply.started
        # and use it for both audio chunks and the agent transcript.
        self._current_reply_item_id: Optional[str] = None
        # Tool call argument shape: AssemblyAI sends already-parsed dicts
        # but we keep the call_id → name mapping for symmetry.
        self._tool_call_names: dict[str, str] = {}
        # Per AssemblyAI docs, tool.result must wait for reply.done. We set
        # this flag when reply.done arrives (non-interrupted), and the next
        # tick's _flush_pending_tool_results actually sends. This handles the
        # race where the orchestrator queues the result AFTER our run_tick
        # returns (so we can't send inline in _process_event).
        self._reply_done_pending_flush: bool = False

    @property
    def provider(self) -> AssemblyAIVoiceAgentProvider:
        if self._provider is None:
            self._provider = AssemblyAIVoiceAgentProvider(
                model=self.model,
                reasoning_effort=self.reasoning_effort,
            )
        return self._provider

    @property
    def is_connected(self) -> bool:
        return self._connected and self._bg_loop.is_running

    def connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: Any = None,
        modality: str = "audio",
    ) -> None:
        if self._connected:
            logger.warning("Already connected, disconnecting first")
            self.disconnect()

        if vad_config is None:
            vad_config = AssemblyAIVADConfig()

        self._bg_loop.start()
        try:
            self._bg_loop.run_coroutine(
                self._async_connect(system_prompt, tools, vad_config, modality),
                timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
            )
            self._connected = True
            logger.info(
                f"DiscreteTimeAssemblyAIAdapter connected "
                f"(tick={self.tick_duration_ms}ms, bytes_per_tick={self.bytes_per_tick})"
            )
        except Exception as e:
            logger.error(f"Failed to connect to AssemblyAI Voice Agent: {e}")
            self._bg_loop.stop()
            raise RuntimeError(
                f"Failed to connect to AssemblyAI Voice Agent: {e}"
            ) from e

    async def _async_connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: AssemblyAIVADConfig,
        modality: str,
    ) -> None:
        await self.provider.connect()
        await self.provider.configure_session(
            system_prompt=system_prompt,
            tools=tools,
            vad_config=vad_config,
            modality=modality,
        )

    def disconnect(self) -> None:
        if not self._connected:
            return

        if self._bg_loop.is_running:
            try:
                self._bg_loop.run_coroutine(
                    self._async_disconnect(),
                    timeout=DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
                )
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")

        self._bg_loop.stop()
        self._connected = False
        self._tick_count = 0
        self._cumulative_user_audio_ms = 0
        self.clear_buffers()
        self._audio_converter.reset()
        self._current_reply_item_id = None
        self._tool_call_names.clear()
        self._reply_done_pending_flush = False
        logger.info("DiscreteTimeAssemblyAIAdapter disconnected")

    async def _async_disconnect(self) -> None:
        if self._owns_provider and self._provider is not None:
            await self.provider.disconnect()

    def run_tick(
        self, user_audio: bytes, tick_number: Optional[int] = None
    ) -> TickResult:
        if not self.is_connected:
            raise RuntimeError(
                "Not connected to AssemblyAI Voice Agent. Call connect() first."
            )

        if tick_number is None:
            tick_number = self._tick_count
        self._tick_count = tick_number + 1

        try:
            return self._bg_loop.run_coroutine(
                self._async_run_tick(user_audio, tick_number),
                timeout=self.tick_duration_ms / 1000
                + DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
            )
        except Exception as e:
            logger.error(f"Error in run_tick (tick={tick_number}): {e}")
            raise

    async def _flush_pending_tool_results(self) -> None:
        """Send tool results only after a normal reply.done has been observed.

        Per AssemblyAI's Voice Agent docs, `tool.result` must NOT be sent
        immediately after `tool.call`. The agent speaks a transition phrase
        and signals it's ready for results via `reply.done`. We flag that
        event and flush on the next tick start — this handles the case where
        the orchestrator queues the tool result AFTER `run_tick` returns
        (the queue is empty during the tick that received reply.done).
        Results are discarded when `reply.done.status == "interrupted"`,
        which is handled in `_process_event`.
        """
        if not self._reply_done_pending_flush:
            return
        if not self._pending_tool_results:
            # reply.done fired but no result was queued — leave the flag set
            # so we flush as soon as the orchestrator delivers the result on
            # a subsequent tick.
            return

        for (
            call_id,
            result_str,
            _request_response,
            _is_error,
        ) in self._pending_tool_results:
            await self.provider.send_tool_result(call_id, result_str)
        logger.debug(
            f"Sent {len(self._pending_tool_results)} tool result(s) to AssemblyAI "
            f"after reply.done"
        )
        self._pending_tool_results.clear()
        self._reply_done_pending_flush = False

    async def _execute_tick(
        self,
        user_audio: bytes,
        tick_number: int,
        result: TickResult,
        tick_start: float,
    ) -> None:
        # Convert user audio: 8kHz μ-law → 24kHz PCM16
        provider_audio = self._audio_converter.convert_input(user_audio)
        provider_audio_received: List[Tuple[bytes, Optional[str]]] = []

        async def receive_events():
            elapsed_so_far = asyncio.get_running_loop().time() - tick_start
            remaining = max(0.01, (self.tick_duration_ms / 1000) - elapsed_so_far)
            return await self.provider.receive_events_for_duration(remaining)

        _, events = await asyncio.gather(
            self._send_audio_chunked(
                provider_audio, self.provider.send_audio, self._chunk_size
            ),
            receive_events(),
        )

        for event in events:
            await self._process_event(result, event, provider_audio_received)

        # Convert agent audio: 24kHz PCM16 → 8kHz μ-law and store in tick
        for pcm_bytes, item_id in provider_audio_received:
            telephony_bytes = self._audio_converter.convert_output(pcm_bytes)
            result.agent_audio_chunks.append((telephony_bytes, item_id))

    async def _process_event(
        self,
        result: TickResult,
        event: Any,
        provider_audio_received: List[Tuple[bytes, Optional[str]]],
    ) -> None:
        result.events.append(event)

        if isinstance(event, ReplyStartedEvent):
            # Use the reply_id as item_id (or synthesize one) and remember it
            # for matching reply.audio + transcript.agent that follow.
            self._current_reply_item_id = event.reply_id or f"reply_{uuid.uuid4().hex}"
            self._utterance_transcripts.setdefault(
                self._current_reply_item_id,
                UtteranceTranscript(item_id=self._current_reply_item_id),
            )

        elif isinstance(event, ReplyAudioEvent):
            if not event.data:
                return
            item_id = self._current_reply_item_id

            # Skip audio from a truncated utterance
            if result.skip_item_id is not None and item_id == result.skip_item_id:
                pcm_bytes = base64.b64decode(event.data)
                # Account for telephony equivalence after format conversion
                estimated_telephony_bytes = int(
                    len(pcm_bytes)
                    * self.audio_format.bytes_per_second
                    / ASSEMBLYAI_OUTPUT_BYTES_PER_SECOND
                )
                result.truncated_audio_bytes += estimated_telephony_bytes
                return

            pcm_bytes = base64.b64decode(event.data)
            provider_audio_received.append((pcm_bytes, item_id))

            if item_id is not None:
                if item_id not in self._utterance_transcripts:
                    self._utterance_transcripts[item_id] = UtteranceTranscript(
                        item_id=item_id
                    )
                # We track received audio in *telephony* bytes (the format the
                # tick result uses) so proportional transcript stays aligned.
                telephony_bytes_est = int(
                    len(pcm_bytes)
                    * self.audio_format.bytes_per_second
                    / ASSEMBLYAI_OUTPUT_BYTES_PER_SECOND
                )
                self._utterance_transcripts[item_id].add_audio(telephony_bytes_est)

        elif isinstance(event, AgentTranscriptEvent):
            # AssemblyAI sends one transcript.agent per reply with full text
            item_id = self._current_reply_item_id or event.item_id
            if item_id and event.text:
                if item_id not in self._utterance_transcripts:
                    self._utterance_transcripts[item_id] = UtteranceTranscript(
                        item_id=item_id
                    )
                self._utterance_transcripts[item_id].add_transcript(event.text)

        elif isinstance(event, ReplyDoneEvent):
            if event.status == "interrupted":
                # User barged in. Per AssemblyAI docs, discard pending tool
                # results — stale results confuse the agent's state.
                result.was_truncated = True
                if self._pending_tool_results:
                    logger.debug(
                        f"reply.done interrupted: discarding "
                        f"{len(self._pending_tool_results)} pending tool result(s)"
                    )
                    self._pending_tool_results.clear()
                self._reply_done_pending_flush = False
            else:
                # Normal reply completion. Flag the next tick's flush to send
                # any pending tool results. We can't send inline here because
                # the orchestrator hasn't queued them yet (tool.call → run_tick
                # returns → orchestrator executes tool → adapter.send_tool_result
                # → next tick).
                self._reply_done_pending_flush = True
            self._current_reply_item_id = None

        elif isinstance(event, SpeechStartedEvent):
            logger.debug("AssemblyAI: input.speech.started")
            result.vad_events.append("speech_started")

            has_agent_audio = (
                result.agent_audio_chunks or self._buffered_agent_audio
            )
            if has_agent_audio:
                last_item_id = None
                if result.agent_audio_chunks:
                    last_item_id = result.agent_audio_chunks[-1][1]
                elif self._buffered_agent_audio:
                    last_item_id = self._buffered_agent_audio[-1][1]

                if self._buffered_agent_audio:
                    buffered_bytes = sum(
                        len(c[0]) for c in self._buffered_agent_audio
                    )
                    result.truncated_audio_bytes += buffered_bytes
                    self._buffered_agent_audio.clear()

                result.truncate_agent_audio(
                    item_id=last_item_id,
                    audio_start_ms=self._cumulative_user_audio_ms,
                    cumulative_user_audio_at_tick_start_ms=(
                        result.cumulative_user_audio_at_tick_start_ms
                    ),
                    bytes_per_tick=result.bytes_per_tick,
                )

        elif isinstance(event, SpeechStoppedEvent):
            logger.debug("AssemblyAI: input.speech.stopped")
            result.vad_events.append("speech_stopped")

        elif isinstance(event, UserTranscriptDeltaEvent):
            logger.debug(f"AssemblyAI user delta: {event.text!r}")

        elif isinstance(event, UserTranscriptEvent):
            logger.debug(f"AssemblyAI user final: {event.text!r}")

        elif isinstance(event, ToolCallEvent):
            if event.arguments is None:
                arguments = {}
            elif isinstance(event.arguments, str):
                try:
                    arguments = json.loads(event.arguments)
                except json.JSONDecodeError:
                    arguments = {}
            elif isinstance(event.arguments, dict):
                arguments = event.arguments
            else:
                arguments = {}

            self._tool_call_names[event.call_id] = event.name
            tool_call = ToolCall(
                id=event.call_id,
                name=event.name,
                arguments=arguments,
            )
            result.tool_calls.append(tool_call)
            logger.debug(f"AssemblyAI tool call: {event.name}({event.call_id})")

        elif isinstance(event, SessionErrorEvent):
            logger.error(
                f"AssemblyAI session.error: code={event.code} msg={event.message}"
            )

        else:
            logger.debug(
                f"AssemblyAI event {getattr(event, 'type', 'unknown')} received"
            )
