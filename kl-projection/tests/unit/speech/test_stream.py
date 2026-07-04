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



def test_continuous_speech_stream_drains_audio_while_transcribing():
    import threading
    import time

    class BlockingTranscriber:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def transcribe_array(self, samples, sample_rate):
            del samples, sample_rate
            self.started.set()
            self.release.wait(timeout=1.0)
            return SimpleNamespace(text="heard speech", segments=[])

    transcriber = BlockingTranscriber()
    stream = ContinuousSpeechStream(
        SpeechStreamConfig(
            sample_rate=100,
            amplitude_threshold=0.1,
            silence_seconds=0.2,
            min_speech_seconds=0.1,
            callback_queue_size=4,
            utterance_queue_size=4,
            poll_interval_seconds=0.01,
        ),
        transcriber=transcriber,
    )

    worker = threading.Thread(target=stream._worker_loop, daemon=True)
    transcription_worker = threading.Thread(target=stream._transcription_loop, daemon=True)
    worker.start()
    transcription_worker.start()

    speech = np.ones(20, dtype=np.float32) * 0.5
    silence = np.zeros(25, dtype=np.float32)
    stream._audio_queue.put(speech, timeout=0.2)
    stream._audio_queue.put(silence, timeout=0.2)
    assert transcriber.started.wait(timeout=1.0)

    stream._audio_queue.put(speech, timeout=0.2)
    stream._audio_queue.put(silence, timeout=0.2)
    deadline = time.time() + 1.0
    while time.time() < deadline and not stream._audio_queue.empty():
        time.sleep(0.01)
    assert stream._audio_queue.empty()

    transcriber.release.set()
    stream._stop_event.set()
    worker.join(timeout=1.0)
    transcription_worker.join(timeout=1.0)

    assert [event.text for event in stream.drain()]
