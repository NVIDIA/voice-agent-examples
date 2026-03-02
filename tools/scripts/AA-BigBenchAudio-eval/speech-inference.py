#!/usr/bin/env python3
"""Speech inference and preprocess for Big Bench Audio. See README for prerequisites and usage."""

import asyncio
import contextlib
import os
import subprocess
import time
import wave

import numpy as np
import resampy
import soundfile as sf
import websockets
from pipecat.frames.protobufs import frames_pb2

# 16 kHz mono WAV is required by our Nemotron voice agent. Adjust for your voice agent's input format.
SAMPLE_RATE, CHUNK_MS = 16000, 32
# After we finish sending input, allow a short initial delay for the first output frame to arrive.
POST_SEND_INITIAL_WAIT_SEC = 5.0
# Once output has started, tolerate output gaps up to ~2s (measured since last received frame).
POST_SEND_GAP_TOLERANCE_SEC = 2.0
# How long to wait for a single websocket recv() before checking termination conditions.
RECV_TIMEOUT_SEC = 2.0
# Bump websocket timeouts to tolerate slower connections
OPEN_TIMEOUT_SEC = 5.0


class BenchmarkClient:
    """WebSocket client for streaming WAV to benchmark server and writing output.wav per sample."""

    def __init__(self, host="localhost", port=8100):
        """Connect to benchmark server at host:port."""
        self.host = host
        self.port = port
        # Base endpoint; per-sample suffix will be appended when connecting.
        self.base_uri = f"ws://{host}:{port}/ws/benchmark"

    async def process_conversation(self, websocket, input_wav_path, output_dir, output_filename: str = "output.wav"):
        """Stream input_wav_path to websocket, collect audio responses, write output_dir/output_filename.

        Send and receive run in parallel (no blocking); output chunk times use session start_time
        so input and output are on the same timeline. Exit rules (gap tolerance, initial wait) are
        applied only after the full input file has been sent, so we capture the main response.
        """
        output_chunks, chunk_times = [], []
        stop_silence_event = asyncio.Event()
        input_send_complete_event = asyncio.Event()
        send_task = asyncio.create_task(
            self.send_audio_file(websocket, input_wav_path, stop_silence_event, input_send_complete_event)
        )
        sender_done_at = None
        last_frame_at = None
        received_any_frame = False
        start_time = time.time()

        while True:
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=RECV_TIMEOUT_SEC)
                if frames_pb2.Frame.FromString(response).WhichOneof("frame") == "audio":
                    audio = frames_pb2.Frame.FromString(response).audio.audio
                    if not audio:
                        continue
                    raw = audio[44:] if len(audio) > 44 and audio.startswith(b"RIFF") else audio
                    chunk = np.frombuffer(raw, dtype=np.int16)
                    if len(chunk) == 0:
                        continue

                    last_frame_at = time.time()
                    received_any_frame = True
                    output_chunks.append(chunk)
                    curr_time = time.time() - start_time  # same time base as session for input/output sync
                    # Use current time if this is the first chunk or if time gap is significant,
                    # otherwise use sequential timing from previous chunk
                    if not chunk_times or abs(curr_time - chunk_times[-1]) >= 0.05:
                        chunk_times.append(curr_time)
                    else:
                        chunk_times.append(chunk_times[-1] + len(chunk) / SAMPLE_RATE)
            except TimeoutError:
                current_time = time.time()

                # Apply exit rules only after the full input file has been sent (so we don't
                # treat intermediate response as "done" while we're still sending a long file).
                if not input_send_complete_event.is_set():
                    continue

                # Input send complete: apply gap tolerance or initial wait.
                if received_any_frame:
                    if last_frame_at and (current_time - last_frame_at) >= POST_SEND_GAP_TOLERANCE_SEC:
                        stop_silence_event.set()
                        break
                    continue

                # No output yet: wait POST_SEND_INITIAL_WAIT_SEC after input send complete.
                if sender_done_at is None:
                    sender_done_at = current_time
                elif (current_time - sender_done_at) >= POST_SEND_INITIAL_WAIT_SEC:
                    stop_silence_event.set()
                    break
                continue
            except websockets.exceptions.ConnectionClosed:
                stop_silence_event.set()
                break

        # Stop sending silence; send_task will exit when it checks the event or when caller closes websocket.
        stop_silence_event.set()
        if not send_task.done():
            send_task.cancel()

        # Write output file
        try:
            if output_chunks and len(output_chunks) > 0:
                # Calculate duration based on actual received chunks
                # Use the latest timestamp + duration of last chunk, or total audio duration, whichever is larger
                if len(chunk_times) != len(output_chunks):
                    # Safety check: ensure arrays match
                    min_len = min(len(chunk_times), len(output_chunks))
                    chunk_times = chunk_times[:min_len]
                    output_chunks = output_chunks[:min_len]

                duration = max(
                    chunk_times[-1] + len(output_chunks[-1]) / SAMPLE_RATE,
                    sum(len(c) / SAMPLE_RATE for c in output_chunks),
                )
                output = np.zeros(int(duration * SAMPLE_RATE), dtype=np.int16)

                for chunk, t in zip(output_chunks, chunk_times, strict=True):
                    if not len(chunk):
                        continue
                    start = max(0, int(t * SAMPLE_RATE))
                    if start >= len(output):
                        continue
                    end = min(start + len(chunk), len(output))

                    if np.any(output[start:end] != 0):
                        while start < len(output) and np.any(output[start : start + len(chunk)] != 0):
                            start += len(chunk)
                        if start >= len(output):
                            continue
                        end = min(start + len(chunk), len(output))

                    output[start:end] = chunk[: end - start]

                output_path = os.path.join(output_dir, output_filename)
                sf.write(output_path, output, SAMPLE_RATE)
            else:
                output_path = os.path.join(output_dir, output_filename)
                sf.write(output_path, np.array([], dtype=np.int16), SAMPLE_RATE)
        except Exception as e:
            print(f"Error writing output file for {output_filename}: {e}")
            raise

    async def process_directory(self, input_dir, batch_size: int = 1, start=None, end=None, samples=None):
        """Process sample dirs: stream input.wav, write output.wav. Optional start/end/samples filter."""
        sample_ids = []
        for name in os.listdir(input_dir):
            if not os.path.isdir(os.path.join(input_dir, name)):
                continue
            try:
                sample_ids.append(int(name))
            except ValueError:
                print(f"warning: skipping non-numeric subdir: {name!r}")
        sample_ids.sort()

        # Filter by specific samples list if provided
        if samples is not None:
            sample_ids = [i for i in sample_ids if i in samples]
        # Otherwise filter by start/end range if specified
        elif start is not None and end is not None:
            sample_ids = [i for i in sample_ids if start <= i <= end]
        elif start is not None:
            sample_ids = [i for i in sample_ids if i >= start]
        elif end is not None:
            sample_ids = [i for i in sample_ids if i <= end]

        semaphore = asyncio.Semaphore(max(1, batch_size))

        async def process_single(sample_id: str):
            sample_dir = os.path.join(input_dir, sample_id)
            input_wav = os.path.join(sample_dir, "input.wav")
            output_filename = "output.wav"

            if not os.path.exists(input_wav):
                return

            uri = f"{self.base_uri}_{sample_id}"
            print(f"processing idx= {sample_id} -> {uri}")
            websocket = None
            try:
                websocket = await websockets.connect(uri, open_timeout=OPEN_TIMEOUT_SEC)
                await self.process_conversation(websocket, input_wav, sample_dir, output_filename)
                print(f"successfully processed idx: {sample_id}")
            except Exception as e:
                print(f"unsuccessful idx: {sample_id} — {e}")
            finally:
                # Close websocket with very short timeout to avoid blocking
                if websocket:
                    with contextlib.suppress(asyncio.TimeoutError, Exception):
                        await asyncio.wait_for(websocket.close(), timeout=0.1)
                await asyncio.sleep(1)

        async def guarded(sample_id: str):
            async with semaphore:
                await process_single(sample_id)

        tasks = [asyncio.create_task(guarded(str(sample_id))) for sample_id in sample_ids]
        if tasks:
            await asyncio.gather(*tasks)

    async def send_audio_file(self, websocket, file_path, stop_silence_event, input_send_complete_event: asyncio.Event):
        """Stream file_path WAV to websocket; set input_send_complete_event when file is fully sent."""
        if not os.path.exists(file_path):
            print(f"Input audio file not found: {file_path}")
            return

        try:
            with wave.open(file_path, "rb") as wav_file:
                n_channels = wav_file.getnchannels()
                frame_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()

                # Calculate chunk size based on target parameters
                chunk_samples = int(SAMPLE_RATE * CHUNK_MS / 1000)
                chunk_dur = CHUNK_MS / 1000

                # Calculate original chunk size for reading from file
                original_chunk_samples = int(frame_rate * CHUNK_MS / 1000)

                silence = np.zeros(chunk_samples, dtype=np.int16).tobytes()
                next_time = time.time()
                # Only signal "input send over" after we've been sending silence this long
                # (so we don't apply exit rules while server is still receiving/processing the stream).
                silence_phase_min_sec = 2.0
                silence_send_start = None

                # Stream the audio file chunk by chunk
                while True:
                    try:
                        await asyncio.sleep(max(0, next_time - time.time()))
                    except asyncio.CancelledError:
                        break

                    if stop_silence_event.is_set():
                        break

                    chunk_bytes = wav_file.readframes(original_chunk_samples)
                    if not chunk_bytes:
                        if silence_send_start is None:
                            silence_send_start = time.time()
                        if (time.time() - silence_send_start) >= silence_phase_min_sec:
                            input_send_complete_event.set()
                        if stop_silence_event.is_set():
                            break
                        try:
                            await websocket.send(
                                frames_pb2.Frame(
                                    audio=frames_pb2.AudioRawFrame(
                                        audio=silence, sample_rate=SAMPLE_RATE, num_channels=1
                                    )
                                ).SerializeToString()
                            )
                        except (
                            websockets.exceptions.ConnectionClosed,
                            websockets.exceptions.ConnectionClosedOK,
                            Exception,
                        ):
                            break
                    else:
                        # Convert bytes to numpy array
                        if sample_width == 1:
                            chunk = np.frombuffer(chunk_bytes, dtype=np.uint8).astype(np.float32) / 127.5 - 1.0
                        elif sample_width == 2:
                            chunk = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                        elif sample_width == 4:
                            chunk = np.frombuffer(chunk_bytes, dtype=np.int32).astype(np.float32) / 2147483647.0
                        else:
                            print(
                                f"warning: unsupported sample_width={sample_width} in {file_path}, "
                                f"falling back to int16 (chunk_bytes len={len(chunk_bytes)}). Supported: 1, 2, 4 bytes."
                            )
                            chunk = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32767.0

                        # Handle multi-channel by averaging
                        if n_channels > 1:
                            chunk = chunk.reshape(-1, n_channels).mean(axis=1)

                        # Resample if necessary
                        if frame_rate != SAMPLE_RATE:
                            chunk = resampy.resample(chunk, frame_rate, SAMPLE_RATE)

                        # Ensure correct chunk size
                        if len(chunk) < chunk_samples:
                            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
                        elif len(chunk) > chunk_samples:
                            chunk = chunk[:chunk_samples]

                        # Convert back to int16 and send
                        chunk_int16 = (chunk * 32767).astype(np.int16)
                        try:
                            await websocket.send(
                                frames_pb2.Frame(
                                    audio=frames_pb2.AudioRawFrame(
                                        audio=chunk_int16.tobytes(), sample_rate=SAMPLE_RATE, num_channels=1
                                    )
                                ).SerializeToString()
                            )
                        except (
                            websockets.exceptions.ConnectionClosed,
                            websockets.exceptions.ConnectionClosedOK,
                            Exception,
                        ):
                            break

                    next_time += chunk_dur

        except wave.Error as e:
            print(f"Failed to read WAV file {file_path}: {e}")
            return
        except Exception as e:
            print(f"Error in send_audio_file: {e}")
            return


def validate_input_dir(input_dir: str) -> str:
    """Resolve and validate input directory; raise ValueError if missing or not a directory."""
    resolved = os.path.realpath(os.path.expanduser(str(input_dir)))
    if not os.path.exists(resolved):
        raise ValueError(f"Input directory does not exist: {resolved}")
    if not os.path.isdir(resolved):
        raise ValueError(f"Input path is not a directory: {resolved}")
    return resolved


def preprocess_bigbench_audio(input_root: str):
    """Convert input.mp3 → input.wav (16 kHz mono). See README for format notes."""
    sample_ids = []
    for name in os.listdir(input_root):
        if not os.path.isdir(os.path.join(input_root, name)):
            continue
        try:
            sample_ids.append(int(name))
        except ValueError:
            print(f"warning: skipping non-numeric subdir: {name!r}")
    sample_ids.sort()

    for sample_id in map(str, sample_ids):
        sample_dir = os.path.join(input_root, sample_id)
        mp3_path = os.path.join(sample_dir, "input.mp3")
        wav_path = os.path.join(sample_dir, "input.wav")
        if not os.path.exists(mp3_path):
            continue
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    mp3_path,
                    "-ac",
                    "1",
                    "-ar",
                    str(SAMPLE_RATE),
                    wav_path,
                ],
                check=True,
            )
        except Exception as e:
            print(f"Warning: preprocessing failed for id={sample_id}: {e}")
            continue


async def main():
    """Parse args, optionally preprocess MP3→WAV, then stream WAVs to benchmark server when --inference is set."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Speech inference for Big Bench Audio: preprocess MP3→WAV and/or stream WAVs to benchmark server."
    )
    parser.add_argument(
        "--input_dir", required=True, help="Directory containing per-sample folders with input.mp3 or input.wav"
    )
    parser.add_argument("--host", default="localhost", help="Voice agent server host")
    parser.add_argument("--port", type=int, default=8100, help="Inference server port")
    parser.add_argument(
        "--preprocess", action="store_true", help="Convert input.mp3 to input.wav (16 kHz mono) under input_dir"
    )
    parser.add_argument("--start", type=int, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, help="End index (inclusive)")
    parser.add_argument("--retry_samples", type=str, help="Comma-separated list of specific sample IDs to retry")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of samples to stream in parallel during inference (default: 1)",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Stream prepared WAVs to the benchmark server (disabled by default)",
    )
    args = parser.parse_args()
    input_root = validate_input_dir(args.input_dir)

    if args.preprocess:
        print(f"Preprocessing MP3s to WAV under: {input_root}")
        preprocess_bigbench_audio(input_root)

    if not args.inference:
        print("Inference skipped (pass --inference to enable streaming).")
        return

    # Parse retry_samples list if provided
    samples_list = None
    if args.retry_samples:
        try:
            samples_list = [int(s.strip()) for s in args.retry_samples.split(",")]
        except ValueError:
            parser.error("--retry_samples must contain only comma-separated integers; got malformed ID")

    client = BenchmarkClient(host=args.host, port=args.port)
    await client.process_directory(
        input_root,
        batch_size=args.batch_size,
        start=args.start,
        end=args.end,
        samples=samples_list,
    )


if __name__ == "__main__":
    asyncio.run(main())
