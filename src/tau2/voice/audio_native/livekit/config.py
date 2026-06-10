"""Configuration types for LiveKit cascaded voice agent.

Defines typed configurations for each component of the cascaded pipeline:
- STT: Speech-to-Text (Deepgram)
- LLM: Language Model (OpenAI, Anthropic)
- TTS: Text-to-Speech (Deepgram, ElevenLabs)

Also provides preset configurations for common use cases.
"""

from typing import Dict, Literal, Optional, Union

from pydantic import BaseModel, Field

# =============================================================================
# STT Configurations
# =============================================================================


class DeepgramSTTConfig(BaseModel):
    """Configuration for Deepgram STT with integrated VAD.

    Deepgram's streaming API includes built-in Voice Activity Detection,
    making it ideal for real-time voice applications.

    Attributes:
        provider: Provider identifier (always "deepgram").
        model: Deepgram model to use. "nova-3" is the latest and most accurate.
        language: Language code (e.g., "en-US", "es", "fr").
        interim_results: Whether to return interim (partial) transcripts.
        vad_events: Whether to emit VAD events (speech start/end).
        endpointing_ms: Silence duration (ms) before considering speech ended.
            Lower values = faster response, higher values = fewer false endpoints.
        utterance_end_ms: Fallback turn-end detection (ms). If a FINAL_TRANSCRIPT
            arrives but no END_OF_SPEECH follows (common with background noise),
            trigger the LLM after this duration of no new transcript activity.
        smart_format: Apply formatting (numbers, dates, etc.).
        punctuate: Add punctuation to transcripts.
    """

    provider: Literal["deepgram"] = "deepgram"
    model: str = "nova-3"
    language: str = "en-US"
    interim_results: bool = True
    vad_events: bool = True
    endpointing_ms: int = 350
    utterance_end_ms: int = 2000
    smart_format: bool = False
    punctuate: bool = True


class DeepgramFluxSTTConfig(BaseModel):
    """Configuration for Deepgram Flux STT (v2 streaming API).

    Flux uses Deepgram's newer "v2" streaming endpoint with a different
    turn-detection model (eager-EOT confidence + EOT threshold + EOT
    timeout) rather than the simple silence threshold that Nova uses.
    It maps to a separate plugin class (`deepgram.STTv2`) on a different
    base URL (`wss://api.deepgram.com/v2/listen`).

    Attributes:
        provider: Provider identifier (always "deepgram-flux").
        model: Flux model id (default "flux-general-en").
        eager_eot_threshold: Confidence threshold for the eager end-of-turn
            check. When None, the plugin's own default is used.
        eot_threshold: Confidence threshold for the final end-of-turn
            decision. When None, the plugin's own default is used.
        eot_timeout_ms: Maximum silence (ms) before forcing the turn to
            end regardless of confidence. When None, plugin default.
        sample_rate: PCM sample rate sent to Deepgram.
        keyterm / keyterms: Domain term boosting.
    """

    provider: Literal["deepgram-flux"] = "deepgram-flux"
    model: str = "flux-general-en"
    sample_rate: int = 16000
    eager_eot_threshold: Optional[float] = None
    eot_threshold: Optional[float] = None
    eot_timeout_ms: Optional[int] = None
    keyterm: Optional[list[str]] = None
    keyterms: Optional[list[str]] = None


class AssemblyAISTTConfig(BaseModel):
    """Configuration for AssemblyAI Universal-3 Pro Streaming STT.

    Streaming variant of AssemblyAI's Universal-3 Pro speech recognition
    model. Uses punctuation-based turn detection — `min_turn_silence` and
    `max_turn_silence` control endpointing directly, not LiveKit's
    `endpointing.min_delay`/`max_delay`.

    Defaults follow AssemblyAI's recommended starting parameters for
    STT-based turn detection in LiveKit:
        - https://www.assemblyai.com/docs/voice-agents/livekit-u3-rt-pro

    Attributes:
        provider: Provider identifier (always "assemblyai").
        model: AssemblyAI streaming model id. "u3-rt-pro" is Universal-3
            Pro Streaming. Other options include "universal-streaming-english"
            and "universal-streaming-multilingual".
        api_key: API key override. If None, the plugin reads
            `ASSEMBLYAI_API_KEY` from the environment.
        min_turn_silence: Silence (ms) before a speculative end-of-turn
            check fires. Default 100 ms (matches both LiveKit plugin and
            AssemblyAI API defaults).
        max_turn_silence: Maximum silence (ms) before forcing a turn to
            end regardless of punctuation. Default 1000 ms — matches the
            AssemblyAI API default and is recommended for STT-driven turn
            detection in LiveKit. (Plugin's own default is 100 ms, which
            is optimized for third-party turn detection models and would
            split entities like phone numbers across turns.)
        vad_threshold: AssemblyAI's internal VAD onset threshold (0.0-1.0).
            Default 0.3 per the LiveKit best-practices doc; pair with a
            matching Silero `activation_threshold` if running an external
            VAD to avoid the dead-zone where one detects speech and the
            other doesn't.
        format_turns: Apply formatting (numbers, dates, etc.) to final
            transcripts. When None, uses the plugin's default.
        keyterms_prompt: List of domain terms to boost in recognition
            (up to 100 terms, each ≤50 chars). When None, none provided.
        end_of_turn_confidence_threshold: Confidence threshold for the
            end-of-turn detector. When None, uses the plugin's default.
        min_end_of_turn_silence_when_confident: Minimum silence (ms)
            before end-of-turn when the model is confident. When None,
            uses the plugin's default.
    """

    provider: Literal["assemblyai"] = "assemblyai"
    model: str = "u3-rt-pro"
    api_key: Optional[str] = None
    min_turn_silence: int = 100
    max_turn_silence: int = 1000
    vad_threshold: float = 0.3
    format_turns: Optional[bool] = None
    keyterms_prompt: Optional[list[str]] = None
    end_of_turn_confidence_threshold: Optional[float] = None
    min_end_of_turn_silence_when_confident: Optional[int] = None
    voice_focus: Optional[str] = None
    voice_focus_threshold: Optional[float] = None
    turn_left_pad_ms: Optional[int] = None


# Type alias for STT configs
STTConfig = Union[DeepgramSTTConfig, DeepgramFluxSTTConfig, AssemblyAISTTConfig]


# =============================================================================
# LLM Configurations
# =============================================================================


class OpenAILLMConfig(BaseModel):
    """Configuration for OpenAI LLM.

    Provides full control over OpenAI model parameters, including:
    - Thinking models (o1, o3) via reasoning_effort
    - Standard models (gpt-4.1, gpt-4.1-mini) via temperature

    Attributes:
        provider: Provider identifier (always "openai").
        model: Model name (e.g., "gpt-4.1", "o3-mini", "gpt-4.1-mini").
        temperature: Sampling temperature (0.0-2.0). Not used for thinking models.
        top_p: Nucleus sampling parameter.
        reasoning_effort: For thinking models (o1, o3): "minimal", "low", "medium", "high".
            Controls how much "thinking" the model does before responding.
        max_completion_tokens: Maximum tokens in the response.
        timeout_seconds: Request timeout in seconds.
        parallel_tool_calls: Whether to allow parallel tool calls.
    """

    provider: Literal["openai"] = "openai"
    model: str = "gpt-4.1"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    reasoning_effort: Optional[Literal["minimal", "low", "medium", "high"]] = None
    max_completion_tokens: Optional[int] = None
    timeout_seconds: Optional[float] = None
    parallel_tool_calls: Optional[bool] = None


class AnthropicLLMConfig(BaseModel):
    """Configuration for Anthropic LLM.

    Supports Claude models including extended thinking capabilities.

    Attributes:
        provider: Provider identifier (always "anthropic").
        model: Model name (e.g., "claude-sonnet-4-20250514").
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.
        thinking_budget_tokens: For extended thinking, max thinking tokens.
            Set to enable Claude's internal reasoning before responding.
    """

    provider: Literal["anthropic"] = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: Optional[float] = None
    thinking_budget_tokens: Optional[int] = None


# Type alias for LLM configs
LLMConfig = Union[OpenAILLMConfig, AnthropicLLMConfig]


# =============================================================================
# TTS Configurations
# =============================================================================


class DeepgramTTSConfig(BaseModel):
    """Configuration for Deepgram TTS.

    Deepgram's Aura voices provide fast, high-quality speech synthesis.

    Attributes:
        provider: Provider identifier (always "deepgram").
        model: Voice model (e.g., "aura-asteria-en", "aura-luna-en").
        sample_rate: Output sample rate in Hz.
    """

    provider: Literal["deepgram"] = "deepgram"
    model: str = "aura-asteria-en"
    sample_rate: int = 24000


class ElevenLabsTTSConfig(BaseModel):
    """Configuration for ElevenLabs TTS.

    ElevenLabs provides highly expressive, human-like voices.

    Attributes:
        provider: Provider identifier (always "elevenlabs").
        voice_id: Voice ID or name (e.g., "aria", "roger").
        model: Model to use (e.g., "eleven_turbo_v2_5", "eleven_multilingual_v2").
        stability: Voice stability (0.0-1.0). Lower = more expressive.
        similarity_boost: How closely to match the original voice (0.0-1.0).
    """

    provider: Literal["elevenlabs"] = "elevenlabs"
    voice_id: str = "aria"
    model: str = "eleven_turbo_v2_5"
    stability: float = 0.5
    similarity_boost: float = 0.75


# Type alias for TTS configs
TTSConfig = Union[DeepgramTTSConfig, ElevenLabsTTSConfig]


# =============================================================================
# Master Cascaded Configuration
# =============================================================================


class CascadedConfig(BaseModel):
    """Configuration for the complete cascaded STT → LLM → TTS pipeline.

    Combines configurations for all three components of the voice pipeline.
    Each component can be configured independently for easy experimentation.

    Attributes:
        stt: Speech-to-Text configuration.
        llm: Language Model configuration.
        tts: Text-to-Speech configuration.
        log_prompts: If True, log the full prompt sent to LLM for debugging.
    """

    stt: STTConfig = Field(default_factory=DeepgramSTTConfig)
    llm: LLMConfig = Field(default_factory=OpenAILLMConfig)
    tts: TTSConfig = Field(default_factory=DeepgramTTSConfig)
    preamble: bool = False
    preamble_text: str = "One moment please."
    log_prompts: bool = False


# =============================================================================
# Preset Configurations
# =============================================================================

CASCADED_CONFIGS: Dict[str, CascadedConfig] = {
    # Default: Balanced speed and quality (Deepgram nova-3 STT)
    "default": CascadedConfig(
        stt=DeepgramSTTConfig(model="nova-3"),
        llm=OpenAILLMConfig(model="gpt-4.1"),
        tts=DeepgramTTSConfig(model="aura-asteria-en"),
    ),
    # Deepgram Flux STT (v2 streaming) — same LLM + TTS as "default".
    # Flux uses Deepgram's confidence-based EOT model on the v2 endpoint.
    "deepgram-flux": CascadedConfig(
        stt=DeepgramFluxSTTConfig(model="flux-general-en"),
        llm=OpenAILLMConfig(model="gpt-4.1"),
        tts=DeepgramTTSConfig(model="aura-asteria-en"),
    ),
    # OpenAI thinking: Uses OpenAI's thinking models with high reasoning effort
    "openai-thinking": CascadedConfig(
        stt=DeepgramSTTConfig(model="nova-3"),
        llm=OpenAILLMConfig(model="gpt-5.2", reasoning_effort="high"),
        tts=DeepgramTTSConfig(model="aura-asteria-en"),
        # preamble=True,
    ),
    # AssemblyAI U3-RT-Pro STT, identical LLM + TTS as "default" — used for
    # head-to-head STT comparison against the "default" Deepgram preset.
    # `min_turn_silence=350` is critical: with the default 100ms, the
    # speculative end-of-turn check fires during the natural pauses between
    # letters when a user spells a name or reads a phone number, and the
    # model inserts terminal punctuation after each fragment. Verified
    # empirically — at 100/1000, "K, O, V, A, C, S" arrived as 3 separate
    # turns and the agent looked up garbage names like "Kobac"/"Kove"/"Ko".
    "assemblyai": CascadedConfig(
        stt=AssemblyAISTTConfig(
            model="u3-rt-pro",
            min_turn_silence=350,
            max_turn_silence=1000,
            vad_threshold=0.3,
        ),
        llm=OpenAILLMConfig(model="gpt-4.1"),
        tts=DeepgramTTSConfig(model="aura-asteria-en"),
    ),
}

# ---------------------------------------------------------------------------
# Local-only experiment fixtures (NOT included in upstream PR).
# Restored after the PR-clean commit so we can keep running parameter
# sweeps without re-editing this file each time. Do not commit.
# ---------------------------------------------------------------------------

# Legacy/experimental: original LiveKit-docs default (min=100, max=1000).
# Reproduces the entity-splitting failure mode.
CASCADED_CONFIGS["assemblyai-fast-min"] = CascadedConfig(
    stt=AssemblyAISTTConfig(
        model="u3-rt-pro",
        min_turn_silence=100,
        max_turn_silence=1000,
        vad_threshold=0.3,
    ),
    llm=OpenAILLMConfig(model="gpt-4.1"),
    tts=DeepgramTTSConfig(model="aura-asteria-en"),
)

# Tighter forced-end (max=350). Experiment fixture, do not use.
CASCADED_CONFIGS["assemblyai-tight"] = CascadedConfig(
    stt=AssemblyAISTTConfig(
        model="u3-rt-pro",
        min_turn_silence=100,
        max_turn_silence=350,
        vad_threshold=0.3,
    ),
    llm=OpenAILLMConfig(model="gpt-4.1"),
    tts=DeepgramTTSConfig(model="aura-asteria-en"),
)

# Sweep over min_turn_silence at 50ms intervals (250-750ms), max held at 1000ms.
for _min_ms in range(250, 800, 50):
    CASCADED_CONFIGS[f"aai-sweep-{_min_ms}"] = CascadedConfig(
        stt=AssemblyAISTTConfig(
            model="u3-rt-pro",
            min_turn_silence=_min_ms,
            max_turn_silence=1000,
            vad_threshold=0.3,
        ),
        llm=OpenAILLMConfig(model="gpt-4.1"),
        tts=DeepgramTTSConfig(model="aura-asteria-en"),
    )
del _min_ms

# Sweep over vad_threshold (0.1, 0.2, 0.3) — sensitivity of AAI's internal VAD
# onset detector. Endpointing held at 350/1000 (the corrected canonical config).
for _vad in [0.1, 0.2, 0.3]:
    _label = f"vad{int(_vad*10):02d}"  # "vad01", "vad02", "vad03"
    CASCADED_CONFIGS[f"aai-{_label}"] = CascadedConfig(
        stt=AssemblyAISTTConfig(
            model="u3-rt-pro",
            min_turn_silence=350,
            max_turn_silence=1000,
            vad_threshold=_vad,
        ),
        llm=OpenAILLMConfig(model="gpt-4.1"),
        tts=DeepgramTTSConfig(model="aura-asteria-en"),
    )
del _vad, _label

# vad=0.1, no voice_focus, min_turn_silence=500ms — isolates the silence
# bump independently of voice_focus. Cell 2 in the n=100 sweep matrix.
CASCADED_CONFIGS["aai-vad01-500"] = CascadedConfig(
    stt=AssemblyAISTTConfig(
        model="u3-rt-pro",
        min_turn_silence=500,
        max_turn_silence=1000,
        vad_threshold=0.1,
    ),
    llm=OpenAILLMConfig(model="gpt-4.1"),
    tts=DeepgramTTSConfig(model="aura-asteria-en"),
)

# vad=0.1 + voice_focus=near-field (threshold=1.0) — best file-level config
# observed on noisy/accented audio. Recovers 9 finals vs 4 baseline on the
# Yusuf Rossi sample. voice_focus is an alpha-only parameter not in the public
# LiveKit plugin — forwarded via local patch.
CASCADED_CONFIGS["aai-vad01-nf"] = CascadedConfig(
    stt=AssemblyAISTTConfig(
        model="u3-rt-pro",
        min_turn_silence=350,
        max_turn_silence=1000,
        vad_threshold=0.1,
        voice_focus="near-field",
        voice_focus_threshold=1.0,
    ),
    llm=OpenAILLMConfig(model="gpt-4.1"),
    tts=DeepgramTTSConfig(model="aura-asteria-en"),
)

# Same as aai-vad01-nf but with min_turn_silence bumped to test whether
# longer EOT silence avoids splitting digit-bursts (e.g. "555-123-2002"
# spoken with 200ms pauses between digit clusters).
for _ms in (500, 750):
    CASCADED_CONFIGS[f"aai-vad01-nf-{_ms}"] = CascadedConfig(
        stt=AssemblyAISTTConfig(
            model="u3-rt-pro",
            min_turn_silence=_ms,
            max_turn_silence=1000,
            vad_threshold=0.1,
            voice_focus="near-field",
            voice_focus_threshold=1.0,
        ),
        llm=OpenAILLMConfig(model="gpt-4.1"),
        tts=DeepgramTTSConfig(model="aura-asteria-en"),
    )
del _ms
