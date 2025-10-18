#!/usr/bin/env python3
"""Audio-only CLI client for the OpenAI Realtime API with persistence."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

import audioop
import sounddevice as sd

try:  # websockets >= 12
    from websockets.asyncio.client import ClientConnection as WebSocketClient
    from websockets.asyncio.client import connect as ws_connect
except ImportError:  # legacy interface
    from websockets.legacy.client import WebSocketClientProtocol as WebSocketClient  # type: ignore
    from websockets.legacy.client import connect as ws_connect  # type: ignore


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        print(f"Environment variable {name} is required", file=sys.stderr)
        sys.exit(1)
    return value


API_KEY = _env("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")
REALTIME_URL = os.getenv("OPENAI_REALTIME_URL", f"wss://api.openai.com/v1/realtime?model={MODEL}")
HISTORY_PATH = Path(os.getenv("OPENAI_REALTIME_HISTORY", "realtime_history.json")).expanduser()
VOICE = os.getenv("OPENAI_REALTIME_VOICE", "alloy")
AUDIO_FORMAT = os.getenv("OPENAI_REALTIME_AUDIO_FORMAT", "g711_ulaw").lower()
MODALITIES = [m.strip() for m in os.getenv("OPENAI_REALTIME_MODALITIES", "audio").split(",") if m.strip()]
if not MODALITIES:
    MODALITIES = ["audio"]
PROMPT_TEXT = os.getenv("OPENAI_AUDIO_PROMPT", "Respond to the user's latest audio input.")
STARTUP_GREETING = os.getenv("OPENAI_STARTUP_GREETING", "Hello. I am ready to talk.")

AUDIO_SAMPLE_RATE = int(os.getenv("OPENAI_AUDIO_SAMPLE_RATE", "8000"))
RECORD_SECONDS_DEFAULT = float(os.getenv("OPENAI_RECORD_SECONDS", "5.0"))
AUDIO_INPUT_DEVICE_RAW = os.getenv("OPENAI_AUDIO_INPUT_DEVICE", "1")
AUDIO_INPUT_CHANNELS = int(os.getenv("OPENAI_AUDIO_INPUT_CHANNELS", "1"))
AUDIO_OUTPUT_DEVICE = os.getenv("OPENAI_AUDIO_OUTPUT_DEVICE", "hw:1")
AUDIO_PLAYBACK_CHANNELS = int(os.getenv("OPENAI_AUDIO_OUTPUT_CHANNELS", "1"))
AUDIO_PLAYER = os.getenv("OPENAI_AUDIO_PLAYER")

try:
    AUDIO_INPUT_DEVICE: int | str = int(AUDIO_INPUT_DEVICE_RAW)
except ValueError:
    AUDIO_INPUT_DEVICE = AUDIO_INPUT_DEVICE_RAW

history: list[dict[str, Any]] = []
pending_text: dict[str, str] = {}
pending_audio: dict[str, bytearray] = {}
_effective_input_channels: int | None = None
_processed_responses: set[str] = set()


def load_history() -> None:
    if not HISTORY_PATH.exists():
        return
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Warning: could not read history file ({exc}). Starting fresh.", file=sys.stderr)
        return
    if isinstance(data, list):
        history.extend(data)
    else:
        print("Warning: history file format unexpected; ignoring.", file=sys.stderr)


def save_history() -> None:
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)
    except OSError as exc:
        print(f"Warning: unable to write history file ({exc}).", file=sys.stderr)


def determine_input_channels() -> int:
    global _effective_input_channels
    if _effective_input_channels is not None:
        return _effective_input_channels

    try:
        device_info = sd.query_devices(AUDIO_INPUT_DEVICE, kind="input")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not query input device {AUDIO_INPUT_DEVICE!r}: {exc}") from exc

    max_channels = int(device_info.get("max_input_channels", 0)) if device_info else 0
    if max_channels <= 0:
        raise RuntimeError("Selected input device reports no capture channels; check OPENAI_AUDIO_INPUT_DEVICE.")

    channels = min(AUDIO_INPUT_CHANNELS, max_channels) if AUDIO_INPUT_CHANNELS > 0 else max_channels
    if channels <= 0:
        raise RuntimeError("No valid channel count determined for the input device.")

    _effective_input_channels = channels
    return channels


def record_audio(duration: float) -> bytes:
    channels = determine_input_channels()
    try:
        frames = sd.rec(
            int(duration * AUDIO_SAMPLE_RATE),
            samplerate=AUDIO_SAMPLE_RATE,
            channels=channels,
            dtype="int16",
            device=AUDIO_INPUT_DEVICE,
        )
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Audio recording failed: {exc}") from exc

    if frames.size == 0:
        raise RuntimeError("No audio captured.")

    return frames.tobytes()


def encode_for_api(pcm_bytes: bytes) -> bytes:
    fmt = AUDIO_FORMAT
    if fmt in {"g711_ulaw", "g711-ulaw", "mulaw", "ulaw"}:
        return audioop.lin2ulaw(pcm_bytes, 2)
    if fmt in {"g711_alaw", "g711-alaw", "alaw"}:
        return audioop.lin2alaw(pcm_bytes, 2)
    return pcm_bytes


def decode_for_playback(audio_bytes: bytes, fmt: str) -> bytes:
    fmt = fmt.lower()
    if fmt in {"g711_ulaw", "g711-ulaw", "mulaw", "ulaw"}:
        return audioop.ulaw2lin(audio_bytes, 2)
    if fmt in {"g711_alaw", "g711-alaw", "alaw"}:
        return audioop.alaw2lin(audio_bytes, 2)
    return audio_bytes


async def send_audio_request(ws: WebSocketClient, pcm_bytes: bytes) -> None:
    encoded_audio = encode_for_api(pcm_bytes)
    encoded = base64.b64encode(encoded_audio).decode("ascii")
    await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": encoded, "audio_format": AUDIO_FORMAT, "sample_rate": AUDIO_SAMPLE_RATE}))
    await ws.send(json.dumps({"type": "input_audio_buffer.commit", "audio_format": AUDIO_FORMAT, "sample_rate": AUDIO_SAMPLE_RATE}))
    payload: dict[str, Any] = {
        "type": "response.create",
        "response": {
            "modalities": MODALITIES,
            "conversation": "default",
            "audio": {"voice": VOICE, "format": AUDIO_FORMAT},
            "instructions": PROMPT_TEXT,
        },
    }
    await ws.send(json.dumps(payload))
    history.append({"role": "user", "content": "(audio input)", "input": "audio", "encoding": AUDIO_FORMAT})
    save_history()


async def send_startup_greeting(ws: WebSocketClient) -> None:
    if not STARTUP_GREETING:
        return
    payload: dict[str, Any] = {
        "type": "response.create",
        "response": {
            "modalities": MODALITIES,
            "conversation": "default",
            "audio": {"voice": VOICE, "format": AUDIO_FORMAT},
            "instructions": STARTUP_GREETING,
        },
    }
    await ws.send(json.dumps(payload))


async def play_audio_response(audio_bytes: bytes, fmt: str) -> None:
    pcm = decode_for_playback(audio_bytes, fmt)
    await asyncio.to_thread(_play_audio_blocking, pcm, AUDIO_SAMPLE_RATE, AUDIO_PLAYBACK_CHANNELS)


def _play_audio_blocking(pcm_audio: bytes, sample_rate: int, channels: int) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name
    try:
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_audio)
        cmd = build_play_command(tmp_path)
        try:
            subprocess.run(cmd, check=False)
        except FileNotFoundError:
            print(f"Audio player not found: {cmd[0]}", file=sys.stderr)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def build_play_command(path: str) -> list[str]:
    player = AUDIO_PLAYER or "aplay"
    if player == "aplay":
        return [player, "-q", "-D", AUDIO_OUTPUT_DEVICE, path]
    return [player, path]


async def handle_final_response(response_id: str) -> None:
    if response_id in _processed_responses:
        return
    _processed_responses.add(response_id)

    text = pending_text.pop(response_id, "").strip()
    audio_bytes = bytes(pending_audio.pop(response_id, b""))

    if audio_bytes:
        await play_audio_response(audio_bytes, AUDIO_FORMAT)

    if text:
        print(text)

    entry = {
        "role": "assistant",
        "content": text,
        "modalities": (["audio"] if audio_bytes else []) + (["text"] if text else []),
    }
    history.append(entry)
    save_history()


async def receive_loop(ws: WebSocketClient) -> None:
    async for message in ws:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            print("Received non-JSON message from server", file=sys.stderr)
            continue
        event_type = data.get("type")
        response = data.get("response") or {}
        response_id = data.get("response_id") or response.get("id")

        if event_type == "response.output_text.delta" and response_id:
            delta = data.get("delta") or data.get("text") or ""
            pending_text[response_id] = pending_text.get(response_id, "") + delta
        elif event_type == "response.output_audio.delta" and response_id:
            chunk = data.get("audio")
            if chunk:
                pending_audio.setdefault(response_id, bytearray()).extend(base64.b64decode(chunk))
        elif event_type == "response.output_audio.done" and response_id:
            await handle_final_response(response_id)
        elif event_type == "response.completed" and response_id:
            await handle_final_response(response_id)
        elif event_type == "response.error" and response_id:
            pending_text.pop(response_id, None)
            pending_audio.pop(response_id, None)
            print(f"Server error event: {data}", file=sys.stderr)


async def capture_loop(ws: WebSocketClient) -> None:
    while True:
        try:
            audio_bytes = await asyncio.to_thread(record_audio, RECORD_SECONDS_DEFAULT)
        except RuntimeError as exc:
            print(f"Recording error: {exc}", file=sys.stderr)
            await asyncio.sleep(0.5)
            continue
        except asyncio.CancelledError:
            break
        await send_audio_request(ws, audio_bytes)


async def main() -> None:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    load_history()
    if history:
        print(f"Loaded {len(history)} history entries from {HISTORY_PATH}.", file=sys.stderr)

    try:
        channels = determine_input_channels()
    except RuntimeError as exc:
        print(f"Input device setup failed: {exc}", file=sys.stderr)
        return
    print(f"Using input device {AUDIO_INPUT_DEVICE!r} with {channels} channel(s).", file=sys.stderr)

    connect_sig = inspect.signature(ws_connect)
    connect_kwargs: dict[str, Any] = {}
    if "extra_headers" in connect_sig.parameters:
        connect_kwargs["extra_headers"] = headers
    elif "additional_headers" in connect_sig.parameters:
        connect_kwargs["additional_headers"] = headers
    else:
        raise RuntimeError("websockets.connect implementation does not support custom headers")
    if "ping_interval" in connect_sig.parameters:
        connect_kwargs["ping_interval"] = 20
    if "ping_timeout" in connect_sig.parameters:
        connect_kwargs["ping_timeout"] = 20

    print(f"Connecting to {REALTIME_URL} (model={MODEL})", file=sys.stderr)
    async with ws_connect(REALTIME_URL, **connect_kwargs) as ws:
        receive_task = asyncio.create_task(receive_loop(ws))
        if STARTUP_GREETING:
            print("Calling startup greeting...")
            await send_startup_greeting(ws)
        print("Listening... (Ctrl+C to stop)", file=sys.stderr)
        capture_task = asyncio.create_task(capture_loop(ws))
        try:
            await asyncio.gather(receive_task, capture_task)
        finally:
            for task in (receive_task, capture_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(receive_task, capture_task, return_exceptions=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
