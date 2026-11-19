"""The swappable parts of Nexus, and how one is chosen.

Nexus is an engine plus four parts. Each part is a small class; which class
is used comes from a setting:

    NEXUS_AUDIO        microphone and speaker        (AudioIO)
    NEXUS_WAKE         wake-word detector            (WakeEngine)
    NEXUS_SPEAKER_ID   "is this the owner?" check    (SpeakerVerifier)
    NEXUS_BACKEND      the voice service that answers (VoiceBackend)

The value is either a built-in name (see BUILT_IN below) or
`package.module:ClassName` for your own class. docs/ADDING-HARDWARE.md walks
through writing one, and `nexus.testing` holds checks you can run against it.

Features (timers, weather, your own) are not parts; they are plugins. See
nexus/tools/__init__.py and docs/ADDING-FEATURES.md.
"""

import importlib
from abc import ABC, abstractmethod

from . import config

env_str, env_float, env_int, env_flag = config.group("core")

AUDIO_NAME = env_str(
    "NEXUS_AUDIO",
    "generic",
    "Microphone and speaker: generic (any sound card), xvf3800, fake, or module:Class.",
)
WAKE_NAME = env_str(
    "NEXUS_WAKE",
    "openwakeword",
    "Wake-word detector: openwakeword, fake, or module:Class.",
)
SPEAKER_ID_NAME = env_str(
    "NEXUS_SPEAKER_ID",
    "none",
    "Voice check: none (anyone may speak), ecapa, fake, or module:Class.",
)
BACKEND_NAME = env_str(
    "NEXUS_BACKEND",
    "openai",
    "Voice service: openai, fake, or module:Class.",
)


class PartError(Exception):
    """A part could not be set up. The message says what to do about it."""


# ---------------------------------------------------------------------------
# Microphone and speaker
# ---------------------------------------------------------------------------


class ReplyStream(ABC):
    """The speaker, held open for one spoken reply."""

    @abstractmethod
    def start(self):
        """Begin playback. Called once, just before the first write()."""

    @abstractmethod
    def write(self, pcm):
        """Play one chunk and return when it has been handed to the device.

        `pcm` is mono int16 numpy audio at the rate given to open_reply().
        Chunks are about 50 ms. Block for roughly the chunk's length: the
        engine relies on that to pace playback and to stop within one chunk
        when interrupted.
        """

    @abstractmethod
    def close(self, abort=False):
        """Stop and release the speaker. abort=True drops queued audio at once."""


class AudioIO(ABC):
    """A microphone and a speaker.

    channels: how many microphone channels you deliver, 1 or 2. Two lets the
    engine pick the better channel per request (for example an echo-cancelled
    beam and a second mix from a microphone array).
    """

    name = "unnamed"
    channels = 1

    @abstractmethod
    def start(self, on_block):
        """Open the devices and start capturing.

        Call on_block(ch0, ch1) for every 80 ms of microphone audio: two
        int16 numpy arrays of config.INPUT_BLOCK_FRAMES samples at
        config.INPUT_RATE. With one channel pass the same array twice. It
        may be called from any thread and returns at once.

        Raise PartError with a plain-words message if a device is missing.
        """

    @abstractmethod
    def stop(self):
        """Stop capturing and release every device. Must be safe to call twice."""

    @abstractmethod
    def open_reply(self, rate):
        """Return a ReplyStream for one reply of mono int16 audio at `rate` Hz."""

    @abstractmethod
    def play_clip(self, pcm, rate):
        """Play a short mono int16 clip (a beep, a notice) and return when it is done."""

    def describe(self):
        """Lines printed at start and by `doctor`: which devices are in use."""
        return [f"Audio: {self.name}"]

    def check(self):
        """For `doctor`: return a list of problems in plain words (empty = fine).

        Must not leave any device open.
        """
        return []


# ---------------------------------------------------------------------------
# Wake word
# ---------------------------------------------------------------------------


class WakeEngine(ABC):
    """Scores microphone audio for the wake phrase."""

    name = "unnamed"

    @abstractmethod
    def load(self):
        """Load models. Called once at start, never at import."""

    @abstractmethod
    def score(self, channel, block):
        """Return 0.0 to 1.0: how sure you are the phrase just ended.

        Called for every block of every channel, in order, in every state
        (including while Nexus is speaking), so keep one stream of memory
        per channel. `block` is int16, config.INPUT_BLOCK_FRAMES samples at
        config.INPUT_RATE. Must return within a few milliseconds.
        """

    def reset(self):
        """Forget recent audio, if the detector can."""

    def check(self):
        return []


# ---------------------------------------------------------------------------
# Speaker check
# ---------------------------------------------------------------------------


class SpeakerVerifier(ABC):
    """Decides whether the voice that spoke belongs to an enrolled person.

    enabled = False means "no check": anyone who says the wake phrase is
    answered, and score() is never called.
    """

    name = "unnamed"
    enabled = True

    def load(self):
        """Load models and voice profiles. Called once at start."""

    @abstractmethod
    def score(self, audio_ch0, audio_ch1):
        """Return (score_ch0, score_ch1), each comparable with NEXUS_SPEAKER_THRESHOLD.

        Each argument is one or more seconds of int16 audio at
        config.INPUT_RATE from one microphone channel. Called in a worker
        thread, so it may take a second or two.
        """

    def check(self):
        return []


# ---------------------------------------------------------------------------
# Voice service
# ---------------------------------------------------------------------------


class VoiceSession(ABC):
    """One live conversation with the voice service.

    events() yields dicts. The engine understands these, and ignores others:

        {"type": "ready"}                      settings accepted; start listening
        {"type": "speech_started"}             the service hears someone talking
        {"type": "speech_stopped"}             they stopped
        {"type": "reply_started", "id": str}
        {"type": "audio", "id": str, "pcm": bytes}   mono int16 at output_rate
        {"type": "transcript", "text": str}    words of the reply, for the log
        {"type": "audio_done"}
        {"type": "reply_done", "id": str, "status": "completed" | "cancelled" | ...,
         "tool_calls": [{"name": str, "call_id": str, "arguments": dict}],
         "usage": dict}
        {"type": "error", "message": str, "code": str}

    The service must detect speech itself (the two speech events) and must
    not answer until request_reply() is called.
    """

    output_rate = 24000

    @abstractmethod
    async def configure(self, instructions, tools):
        """Send the system prompt and the tool list (see nexus.tools.schemas())."""

    @abstractmethod
    async def send_audio(self, pcm):
        """Send microphone audio: int16 at config.INPUT_RATE."""

    @abstractmethod
    async def clear_input(self):
        """Forget audio sent so far that has not been answered."""

    @abstractmethod
    async def request_reply(self):
        """Ask for a reply to what was said."""

    @abstractmethod
    async def cancel_reply(self):
        """Stop the reply being generated."""

    @abstractmethod
    async def send_tool_result(self, call_id, result):
        """Return a tool's result (a JSON-able dict)."""

    @abstractmethod
    async def add_system_note(self, text):
        """Add an instruction the model should see before its next reply."""

    @abstractmethod
    def events(self):
        """An async iterator of event dicts. Ends when the service closes the connection."""


class VoiceBackend(ABC):
    name = "unnamed"

    @abstractmethod
    def connect(self):
        """An async context manager that yields a VoiceSession.

        Raise an exception whose text says why if connecting fails;
        nexus.alerts.classify_failure() sorts it into credit, key, model or
        network from that text.
        """

    def check(self):
        return []


# ---------------------------------------------------------------------------
# Choosing a part
# ---------------------------------------------------------------------------

BUILT_IN = {
    "audio": {
        "generic": "nexus.audio.sounddevice_io:SoundDeviceAudio",
        "xvf3800": "nexus.audio.xvf3800:XVF3800Audio",
        "fake": "nexus.audio.fake:FakeAudio",
    },
    "wake": {
        "openwakeword": "nexus.wake.openwakeword_engine:OpenWakeWordEngine",
        "fake": "nexus.wake.fake:FakeWakeEngine",
    },
    "speaker_id": {
        "none": "nexus.speaker_id.none:NoVerifier",
        "ecapa": "nexus.speaker_id.ecapa:EcapaVerifier",
        "fake": "nexus.speaker_id.fake:FakeVerifier",
    },
    "backend": {
        "openai": "nexus.backends.openai_realtime:OpenAIRealtimeBackend",
        "fake": "nexus.backends.fake:FakeBackend",
    },
}

INTERFACES = {
    "audio": AudioIO,
    "wake": WakeEngine,
    "speaker_id": SpeakerVerifier,
    "backend": VoiceBackend,
}

SETTING_FOR = {
    "audio": "NEXUS_AUDIO",
    "wake": "NEXUS_WAKE",
    "speaker_id": "NEXUS_SPEAKER_ID",
    "backend": "NEXUS_BACKEND",
}


def register(kind, name, target):
    """Make your own part selectable by a short name.

    `target` is a class or a "package.module:ClassName" string. Plugins may
    call this from register(); the short name then works in the setting.
    """
    if kind not in BUILT_IN:
        raise PartError(f"Unknown kind of part {kind!r}. Kinds: {', '.join(BUILT_IN)}.")

    BUILT_IN[kind][name] = target


def chosen(kind):
    return {
        "audio": AUDIO_NAME,
        "wake": WAKE_NAME,
        "speaker_id": SPEAKER_ID_NAME,
        "backend": BACKEND_NAME,
    }[kind].strip()


def load_class(kind, name=None):
    name = (name or chosen(kind)).strip()
    target = BUILT_IN[kind].get(name.lower(), name)

    if isinstance(target, str):
        if ":" not in target:
            known = ", ".join(sorted(BUILT_IN[kind]))
            raise PartError(
                f"{SETTING_FOR[kind]}={name!r} is not a known {kind} part. "
                f"Use one of: {known}; or module:ClassName for your own."
            )

        module_name, _, class_name = target.partition(":")

        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise PartError(
                f"{SETTING_FOR[kind]}={name!r} needs the Python module {module_name!r}, "
                f"which could not be imported: {exc}"
            ) from exc

        try:
            target = getattr(module, class_name)
        except AttributeError as exc:
            raise PartError(f"{module_name} has no class named {class_name!r}.") from exc

    if not (isinstance(target, type) and issubclass(target, INTERFACES[kind])):
        raise PartError(
            f"{SETTING_FOR[kind]}={name!r} is not a {INTERFACES[kind].__name__}. "
            f"Subclass nexus.parts.{INTERFACES[kind].__name__}."
        )

    return target


def create(kind, name=None):
    """Build the chosen part. Nothing is opened or loaded yet."""
    return load_class(kind, name)()
