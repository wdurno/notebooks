from types import SimpleNamespace

from picar_kl.speech.adapters import EmptySpeechSource, QueueSpeechSource, SpeechServiceSpeaker, SpeechServiceSource, StreamingSpeechSource


class FakeTranscription:
    text = "find the red ball"


class FakeSpeechService:
    def __init__(self):
        self.spoken = []

    def listen_once(self):
        return FakeTranscription()

    def speak(self, text):
        self.spoken.append(text)


def test_empty_speech_source_returns_no_texts():
    assert EmptySpeechSource().drain_texts() == []


def test_queue_speech_source_drains_once():
    source = QueueSpeechSource(["hello"])
    source.append("world")

    assert source.drain_texts() == ["hello", "world"]
    assert source.drain_texts() == []


def test_speech_service_source_adapts_service_result():
    source = SpeechServiceSource(FakeSpeechService())

    assert source.drain_texts() == ["find the red ball"]


def test_speech_service_speaker_ignores_blank_text():
    service = FakeSpeechService()
    speaker = SpeechServiceSpeaker(service)

    speaker.speak("")
    speaker.speak("Checking.")

    assert service.spoken == ["Checking."]



class FakeStream:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def drain(self):
        return [SimpleNamespace(text=" turn left "), SimpleNamespace(text="")]


def test_streaming_speech_source_drains_without_blocking():
    stream = FakeStream()
    source = StreamingSpeechSource(stream)

    source.start()
    assert source.drain_texts() == [" turn left "]
    source.stop()

    assert stream.started is True
    assert stream.stopped is True
