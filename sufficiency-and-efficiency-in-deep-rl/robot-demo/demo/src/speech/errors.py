class SpeechError(RuntimeError):
    pass


class MissingDependencyError(SpeechError):
    pass


class ModelAssetError(SpeechError):
    pass


class AudioDeviceError(SpeechError):
    pass
