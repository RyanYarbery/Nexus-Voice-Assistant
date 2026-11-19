"""The OpenAI Realtime API as the voice service (NEXUS_BACKEND=openai).

Speech goes up as audio and the reply comes back as audio; the model can
call tools in between. Usage is billed by OpenAI to the account that owns
OPENAI_API_KEY.

This file is the only place that knows OpenAI's message names. It turns
them into the small event vocabulary in nexus.parts.VoiceSession.
"""

import base64
import contextlib
import json
import os

from .. import config
from ..dsp import StreamResampler, resample_int16
from ..parts import VoiceBackend, VoiceSession

env_str, env_float, env_int, env_flag = config.group("backend: openai")

MODEL_RATE = 24000

# OpenAI lists gpt-realtime for shutdown on 2027-01-20 with
# gpt-realtime-2.1 as its replacement at the same price
# (developers.openai.com/api/docs/deprecations, read 2026-11-18).
OPENAI_MODEL = env_str(
    "OPENAI_REALTIME_MODEL",
    "gpt-realtime-2.1",
    "OpenAI Realtime model name.",
)
OPENAI_VOICE = env_str("NEXUS_VOICE", "marin", "OpenAI voice used for replies.")

VAD_THRESHOLD = env_float("NEXUS_VAD_THRESHOLD", 0.5, "OpenAI speech detector sensitivity, 0 to 1.")
VAD_SILENCE_MS = env_int("NEXUS_VAD_SILENCE_MS", 700, "Silence that ends your turn, in milliseconds.")
VAD_PREFIX_MS = env_int("NEXUS_VAD_PREFIX_MS", 300, "Audio kept from before speech was detected.")
TURN_DETECTION = env_str(
    "NEXUS_TURN_DETECTION",
    "server_vad",
    "How OpenAI decides you have finished: server_vad (fixed silence) or semantic_vad.",
).lower()
NOISE_REDUCTION = env_str(
    "NEXUS_NOISE_REDUCTION",
    "off",
    "OpenAI input noise filter: off, near_field or far_field.",
).lower()

PING_INTERVAL_SECONDS = env_float("NEXUS_PING_INTERVAL_SECONDS", 5, "Seconds between keep-alive pings.")
PING_TIMEOUT_SECONDS = env_float(
    "NEXUS_PING_TIMEOUT_SECONDS", 20, "Drop the connection after this long without a ping reply."
)
UPLINK_STREAM_RESAMPLE = env_flag(
    "NEXUS_UPLINK_STREAM_RESAMPLE",
    True,
    "Resample microphone audio as one stream (0 = block by block, which clicks at block edges).",
)


def turn_detection_settings():
    if TURN_DETECTION == "semantic_vad":
        return {
            "type": "semantic_vad",
            "create_response": False,
            "interrupt_response": False,
        }

    return {
        "type": "server_vad",
        "threshold": VAD_THRESHOLD,
        "prefix_padding_ms": VAD_PREFIX_MS,
        "silence_duration_ms": VAD_SILENCE_MS,
        "create_response": False,
        "interrupt_response": False,
    }


def noise_reduction_settings():
    if NOISE_REDUCTION in ("near_field", "far_field"):
        return {"noise_reduction": {"type": NOISE_REDUCTION}}

    return {}


class OpenAIRealtimeSession(VoiceSession):
    output_rate = MODEL_RATE

    def __init__(self, ws):
        self.ws = ws
        self._uplink = None

    async def _send(self, payload):
        await self.ws.send(json.dumps(payload))

    async def configure(self, instructions, tools):
        await self._send(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "instructions": instructions,
                    "tools": tools,
                    "tool_choice": "auto",
                    "output_modalities": ["audio"],
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": MODEL_RATE},
                            # Nexus decides when a reply is wanted (after the
                            # speaker check), so the server only reports speech.
                            "turn_detection": turn_detection_settings(),
                            **noise_reduction_settings(),
                        },
                        "output": {
                            "format": {"type": "audio/pcm", "rate": MODEL_RATE},
                            "voice": OPENAI_VOICE,
                        },
                    },
                },
            }
        )

    async def send_audio(self, pcm):
        if len(pcm) == 0:
            return

        if UPLINK_STREAM_RESAMPLE:
            # One resampler for the whole upload. Resampling each 80 ms
            # block on its own leaves a click at every block edge.
            if self._uplink is None:
                self._uplink = StreamResampler(config.INPUT_RATE, MODEL_RATE)

            pcm_24k = self._uplink.process(pcm)
        else:
            pcm_24k = resample_int16(pcm, config.INPUT_RATE, MODEL_RATE)

        await self._send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm_24k.tobytes()).decode("ascii"),
            }
        )

    async def clear_input(self):
        # A new utterance starts after a clear; start its resampling clean.
        self._uplink = None

        await self._send({"type": "input_audio_buffer.clear"})

    async def request_reply(self):
        await self._send({"type": "response.create"})

    async def cancel_reply(self):
        await self._send({"type": "response.cancel"})

    async def send_tool_result(self, call_id, result):
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result),
                },
            }
        )

    async def add_system_note(self, text):
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )

    async def events(self):
        async for raw in self.ws:
            message = json.loads(raw)
            kind = message.get("type")

            if kind == "session.created":
                print("Realtime session created.")

            elif kind == "session.updated":
                yield {"type": "ready"}

            elif kind == "input_audio_buffer.speech_started":
                yield {"type": "speech_started"}

            elif kind == "input_audio_buffer.speech_stopped":
                yield {"type": "speech_stopped"}

            elif kind == "response.created":
                yield {"type": "reply_started", "id": message.get("response", {}).get("id")}

            elif kind == "response.output_audio.delta":
                delta = message.get("delta")

                if delta:
                    yield {"type": "audio", "id": message.get("response_id"), "pcm": base64.b64decode(delta)}

            elif kind == "response.output_audio_transcript.delta":
                yield {"type": "transcript", "text": message.get("delta", "")}

            elif kind == "response.output_audio.done":
                yield {"type": "audio_done"}

            elif kind == "response.done":
                response = message.get("response", {})
                tool_calls = []

                for item in response.get("output", []):
                    if item.get("type") != "function_call":
                        continue

                    try:
                        arguments = json.loads(item.get("arguments", "{}"))
                    except Exception:
                        arguments = {}

                    tool_calls.append(
                        {
                            "name": item.get("name"),
                            "call_id": item.get("call_id"),
                            "arguments": arguments if isinstance(arguments, dict) else {},
                        }
                    )

                yield {
                    "type": "reply_done",
                    "id": response.get("id"),
                    "status": response.get("status"),
                    "tool_calls": tool_calls,
                    "usage": response.get("usage") or {},
                }

            elif kind == "error":
                error = message.get("error", {})
                error_message = str(error.get("message", error))

                if "maximum duration" in error_message.lower():
                    # The hourly session limit: end this connection so a
                    # fresh one is opened.
                    raise ConnectionError(error_message)

                yield {
                    "type": "error",
                    "message": error_message,
                    "code": f"{error.get('code', '')} {error.get('type', '')}".strip(),
                }


class OpenAIRealtimeBackend(VoiceBackend):
    name = "openai"
    label = f"openai:{OPENAI_MODEL}"

    def check(self):
        problems = []

        if not os.environ.get("OPENAI_API_KEY"):
            problems.append(
                "OPENAI_API_KEY is not set. Create a key at platform.openai.com and put it in the env file."
            )

        try:
            import websockets  # noqa: F401
        except ImportError as exc:
            problems.append(f"The websockets package is not installed ({exc}). pip install websockets")

        return problems

    @contextlib.asynccontextmanager
    async def connect(self):
        api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        import websockets

        async with websockets.connect(
            f"wss://api.openai.com/v1/realtime?model={OPENAI_MODEL}",
            additional_headers={"Authorization": f"Bearer {api_key}"},
            max_size=None,
            ping_interval=PING_INTERVAL_SECONDS,
            ping_timeout=PING_TIMEOUT_SECONDS,
            close_timeout=3,
        ) as ws:
            yield OpenAIRealtimeSession(ws)
