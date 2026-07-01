from types import SimpleNamespace

import numpy as np

from picar_kl.speech.config import SpeechStreamConfig
from picar_kl.speech.stream import ContinuousSpeechStream, UtteranceSegmenter


class FakeTranscriber:
    def transcribe_array(self, samples, sample_rate):
        del sample_rate
        energy = float(np.abs(samples).mean())
        text = "heard speech" if energy > 0.01 else ""
        return SimpleNamespace(text=text, segments=[{"text": text}])


def test_utterance_segmenter_emits_after_silence_gap():
    segmenter = UtteranceSegmenter(
        sample_rate=100,
        amplitude_threshold=0.1,
        silence_seconds=0.2,
        min_speech_seconds=0.1,
    )

    emitted = []
    emitted.extend(segmenter.ingest(np.ones(20, dtype=np.float32) * 0.5))
    emitted.extend(segmenter.ingest(np.zeros(25, dtype=np.float32)))

    assert len(emitted) == 1
    assert emitted[0].shape[0] == 45


def test_continuous_speech_stream_push_audio_enqueues_transcription():
    stream = ContinuousSpeechStream(
        SpeechStreamConfig(
            sample_rate=100,
            amplitude_threshold=0.1,
            silence_seconds=0.2,
            min_speech_seconds=0.1,
        ),
        transcriber=FakeTranscriber(),
    )

    stream.push_audio(np.ones(20, dtype=np.float32) * 0.5)
    stream.push_audio(np.zeros(25, dtype=np.float32))
    events = stream.drain()

    assert [event.text for event in events] == ["heard speech"]
