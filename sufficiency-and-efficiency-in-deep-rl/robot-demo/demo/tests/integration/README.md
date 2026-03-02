# Integration Tests

These tests exercise microphone, speaker, STT, and TTS behavior that cannot be
validated in the fast unit test suite.

## Manual round-trip speech test

Run:

```bash
pytest demo/tests/integration/test_speech_roundtrip_manual.py -s
```

The test will:

1. wait for you to press Enter;
2. record microphone audio until you press Enter again;
3. print the recognized STT text in the terminal; and
4. synthesize the recognized text with Piper and play it back.

## Expected model locations

If the assets are not already present, the speech code will attempt to download:

- STT into `demo/model/stt/whisper-large-v3-turbo-int8/`
- TTS into `demo/model/tts/piper/en_US-lessac-medium/`

## Python dependencies

Install these before running the manual test:

```bash
pip install faster-whisper huggingface_hub numpy sounddevice piper-tts pytest
```

## System dependencies

### macOS

You may need PortAudio for `sounddevice`:

```bash
brew install portaudio
```

### Ubuntu

You may need audio and speech runtime libraries:

```bash
sudo apt-get install portaudio19-dev ffmpeg espeak-ng
```

If you want GPU-backed STT on the Ubuntu tower, ensure the CUDA runtime used by
your `faster-whisper` installation is available as well.

## Notes

- Speech inference prefers CUDA when available and falls back to CPU otherwise.
- Piper playback and microphone access depend on the local audio device setup.
- This test is intended for manual invocation, not the default fast pytest run.
