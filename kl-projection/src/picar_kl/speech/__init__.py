"""Speech input and output support for phase runtimes."""

from picar_kl.speech.config import AudioIOConfig, STTConfig, SpeechConfig, SpeechStreamConfig, TTSConfig, choose_runtime_device
from picar_kl.speech.service import SpeechRoundTrip, SpeechService
from picar_kl.speech.stream import ContinuousSpeechStream, QueuedSpeechEvent, UtteranceSegmenter

__all__ = [
    "AudioIOConfig",
    "STTConfig",
    "SpeechConfig",
    "SpeechRoundTrip",
    "SpeechService",
    "SpeechStreamConfig",
    "ContinuousSpeechStream",
    "QueuedSpeechEvent",
    "UtteranceSegmenter",
    "TTSConfig",
    "choose_runtime_device",
]
