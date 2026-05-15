"""LiveKit-based cascaded voice pipeline (STT → LLM → TTS).

This module provides a cascaded voice provider that orchestrates:
- STT: Deepgram streaming transcription with integrated VAD
- LLM: OpenAI/Anthropic with full parameter control (reasoning_effort, etc.)
- TTS: Deepgram/ElevenLabs streaming synthesis

Architecture:
- provider.py: Core pipeline logic (CascadedVoiceProvider)
    - Turn-taking decisions
    - Interruption handling
    - Context management
    - Streaming orchestration

- discrete_time_adapter.py: Thin wrapper (LiveKitCascadedAdapter)
    - Tick-based timing interface
    - Audio buffering for tick alignment
    - Event mapping to TickResult

- config.py: Configuration models
    - STT configs (DeepgramSTTConfig)
    - LLM configs (OpenAILLMConfig, AnthropicLLMConfig)
    - TTS configs (DeepgramTTSConfig, ElevenLabsTTSConfig)
    - Named presets (CASCADED_CONFIGS)

Usage:
    from tau2.voice.audio_native.livekit import (
        CASCADED_CONFIGS,
        LiveKitCascadedAdapter,
    )

    adapter = LiveKitCascadedAdapter(
        tick_duration_ms=1000,
        cascaded_config=CASCADED_CONFIGS["openai-thinking"],
    )
"""

from tau2.voice.audio_native.livekit.config import (
    CASCADED_CONFIGS,
    AnthropicLLMConfig,
    AssemblyAISTTConfig,
    CascadedConfig,
    DeepgramFluxSTTConfig,
    DeepgramSTTConfig,
    DeepgramTTSConfig,
    ElevenLabsTTSConfig,
    LLMConfig,
    OpenAILLMConfig,
    STTConfig,
    TTSConfig,
)
from tau2.voice.audio_native.livekit.discrete_time_adapter import (
    LiveKitCascadedAdapter,
    LiveKitVADConfig,
)
from tau2.voice.audio_native.livekit.provider import (
    CascadedEvent,
    CascadedEventType,
    CascadedVoiceProvider,
    ConversationContext,
    ProviderState,
    TurnTakingConfig,
)


def preregister_livekit_plugins() -> None:
    """Pre-register LiveKit plugins on the main thread.

    LiveKit requires plugins to be registered on the main thread. This function
    imports the plugins before ThreadPoolExecutor workers are spawned, ensuring
    they are available in worker threads without thread registration errors.

    Must be called from the main thread before any workers are created.
    """
    from loguru import logger

    # Each plugin gets its own try/except so a missing optional plugin
    # doesn't prevent the others from registering. Importing the module
    # triggers its `Plugin.register_plugin(...)` call at module-import time,
    # which requires the main thread.
    registered: list[str] = []
    for plugin_name in ("deepgram", "openai", "anthropic", "elevenlabs", "assemblyai"):
        try:
            __import__(f"livekit.plugins.{plugin_name}")
            registered.append(plugin_name)
        except ImportError:
            logger.debug(f"livekit-plugins-{plugin_name} not installed, skipping")
    if registered:
        logger.debug(f"LiveKit plugins pre-registered on main thread: {registered}")
    else:
        logger.warning(
            "No LiveKit plugins could be pre-registered. Install with: "
            "pip install livekit-plugins-openai livekit-plugins-deepgram "
            "livekit-plugins-assemblyai"
        )


__all__ = [
    # Main adapter for discrete-time simulation
    "LiveKitCascadedAdapter",
    "LiveKitVADConfig",
    # Core provider
    "CascadedVoiceProvider",
    "CascadedEvent",
    "CascadedEventType",
    "ProviderState",
    "ConversationContext",
    "TurnTakingConfig",
    # Utilities
    "preregister_livekit_plugins",
    # Configuration
    "CascadedConfig",
    "STTConfig",
    "LLMConfig",
    "TTSConfig",
    "DeepgramSTTConfig",
    "DeepgramFluxSTTConfig",
    "AssemblyAISTTConfig",
    "OpenAILLMConfig",
    "AnthropicLLMConfig",
    "DeepgramTTSConfig",
    "ElevenLabsTTSConfig",
    "CASCADED_CONFIGS",
]
