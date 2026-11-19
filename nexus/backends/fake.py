"""A voice service that answers every request with one second of silence (NEXUS_BACKEND=fake).

For trying the whole loop with no account and no network, and for tests.
"""

import asyncio
import contextlib

import numpy as np

from ..parts import VoiceBackend, VoiceSession


class FakeSession(VoiceSession):
    output_rate = 24000

    def __init__(self):
        self.queue = asyncio.Queue()
        self.calls = []
        self.replies = 0

    async def configure(self, instructions, tools):
        self.calls.append(("configure", len(tools)))
        self.queue.put_nowait({"type": "ready"})

    async def send_audio(self, pcm):
        self.calls.append(("send_audio", len(pcm)))

    async def clear_input(self):
        self.calls.append(("clear_input",))

    async def request_reply(self):
        self.calls.append(("request_reply",))
        self.replies += 1
        reply_id = f"fake-{self.replies}"
        self.queue.put_nowait({"type": "reply_started", "id": reply_id})
        self.queue.put_nowait(
            {"type": "audio", "id": reply_id, "pcm": np.zeros(self.output_rate, np.int16).tobytes()}
        )
        self.queue.put_nowait(
            {"type": "reply_done", "id": reply_id, "status": "completed", "tool_calls": [], "usage": {}}
        )

    async def cancel_reply(self):
        self.calls.append(("cancel_reply",))

    async def send_tool_result(self, call_id, result):
        self.calls.append(("send_tool_result", call_id))

    async def add_system_note(self, text):
        self.calls.append(("add_system_note", text))

    async def events(self):
        while True:
            event = await self.queue.get()

            if event is None:
                return

            yield event


class FakeBackend(VoiceBackend):
    name = "fake"
    label = "fake"

    @contextlib.asynccontextmanager
    async def connect(self):
        yield FakeSession()
