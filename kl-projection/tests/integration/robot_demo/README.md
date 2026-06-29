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

## Manual PiCar env smoke test

Run:

```bash
PICAR_V_HOST=<host:port> pytest demo/tests/integration/test_picar_env_manual.py -s
```

Optional environment variables:

- `PICAR_TEST_STEPS=2` controls how many env iterations run after the operator says `start`
- `PICAR_USE_REAL_ACTION_MODEL=1` switches from the fast fake action model to the real Qwen-backed policy path

The test will:

1. print setup instructions in the terminal;
2. speak the same setup instructions through TTS;
3. ask you to:
   set up the PiCar;
   put it on blocks or pick it up;
   put a red ball in view; and
   say `start` when ready;
4. wait for the continuous speech stream to recognize `start`;
5. run a short real env loop against the PiCar API;
6. verify replay/checkpoint/log artifacts were written before exit.

This smoke test is designed to validate the new env orchestration quickly. By
default it uses the real speech stack, the real frozen reward scorer, and a
fast fake action-policy backbone so the robot loop starts promptly. If you want
to exercise the full Qwen action path as well, set `PICAR_USE_REAL_ACTION_MODEL=1`.

## Manual vision stream test

Run (default 30 seconds):

```bash
PICAR_V_HOST=<host:port> pytest demo/tests/integration/vision_test.py -s
```

Change duration with a command-line flag:

```bash
PICAR_V_HOST=<host:port> pytest demo/tests/integration/vision_test.py -s --vision-seconds=45
```

Run until you stop it with `Ctrl-C`:

```bash
PICAR_V_HOST=<host:port> pytest demo/tests/integration/vision_test.py -s --vision-seconds=-1
```

This test:

1. prints instructions and reminds you that `Ctrl-C` ends early;
2. runs the `PiCarGymEnv` loop with per-step prompts asking the robot to
   describe what it sees;
3. streams frames to a desktop OpenCV window; and
4. plays generated robot text through TTS on each step.
