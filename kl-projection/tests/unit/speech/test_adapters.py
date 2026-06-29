from picar_kl.speech.adapters import EmptySpeechSource, LegacySpeaker, LegacySpeechSource, QueueSpeechSource


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


def test_legacy_speech_source_adapts_service_result():
    source = LegacySpeechSource(FakeSpeechService())

    assert source.drain_texts() == ["find the red ball"]


def test_legacy_speaker_ignores_blank_text():
    service = FakeSpeechService()
    speaker = LegacySpeaker(service)

    speaker.speak("")
    speaker.speak("Checking.")

    assert service.spoken == ["Checking."]
