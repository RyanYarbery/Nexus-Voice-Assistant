"""Wake word with openWakeWord (https://github.com/dscripka/openWakeWord).

You supply the model: an .onnx file trained for your phrase. The model is
loaded once per microphone channel, because each copy remembers the last
second or so of the one audio stream it is fed.
"""

from pathlib import Path

from .. import config
from ..parts import PartError, WakeEngine

env_str, env_float, env_int, env_flag = config.group("wake: openwakeword")

WAKE_MODEL_PATH = env_str(
    "NEXUS_WAKE_MODEL_PATH",
    str(config.BASE_DIR / "models" / "hey_nexus.onnx"),
    "openWakeWord .onnx model for your wake phrase.",
)


class OpenWakeWordEngine(WakeEngine):
    name = "openwakeword"

    def __init__(self):
        self.models = {}

    def check(self):
        problems = []

        if not Path(WAKE_MODEL_PATH).is_file():
            problems.append(
                f"Wake-word model not found at {WAKE_MODEL_PATH}. Train or download an openWakeWord "
                ".onnx model for your phrase and point NEXUS_WAKE_MODEL_PATH at it."
            )

        try:
            import openwakeword  # noqa: F401
        except ImportError as exc:
            problems.append(f"The openwakeword package is not installed ({exc}). pip install openwakeword")

        return problems

    def _new_model(self):
        try:
            from openwakeword.model import Model
        except ImportError as exc:
            raise PartError(
                f"The openwakeword package is not installed ({exc}). pip install openwakeword"
            ) from exc

        if not Path(WAKE_MODEL_PATH).is_file():
            raise PartError(
                f"Wake-word model not found at {WAKE_MODEL_PATH}. Set NEXUS_WAKE_MODEL_PATH to your .onnx model."
            )

        return Model(wakeword_model_paths=[WAKE_MODEL_PATH])

    def load(self):
        print(f"Loading wake-word model: {WAKE_MODEL_PATH}")

        # Two channels at most; a one-channel microphone only ever uses the first.
        self.models = {0: self._new_model(), 1: self._new_model()}

    def score(self, channel, block):
        predictions = self.models[channel].predict(block)

        if not predictions:
            return 0.0

        return max(float(value) for value in predictions.values())

    def reset(self):
        # openWakeWord versions have used different reset method names,
        # and reset() clears only the score history, not the remembered
        # audio. The engine therefore feeds every block in every state
        # and does not depend on this.
        for model in self.models.values():
            for method_name in ("reset", "reset_state"):
                method = getattr(model, method_name, None)

                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass

                    break
