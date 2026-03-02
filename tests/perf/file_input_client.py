"""Voice Agent WebSocket client with latency measurement for performance testing.

This module provides a WebSocket client that sends audio files to a voice agent
service and measures the latency between the end of user audio and the beginning of the agent's response.
Additionally, it detects reverse barge-in and audio glitches on the client side.
"""

import argparse
import asyncio
import datetime
import io
import json
import os
import signal
import sys
import time
import uuid
import wave

import websockets
from pipecat.frames.protobufs import frames_pb2
from websockets.exceptions import ConnectionClosed


def log_error(msg):
    """Write error message to stderr with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[ERROR] {timestamp} - {msg}", file=sys.stderr, flush=True)


# Global constants
SILENCE_TIMEOUT = 0.2  # Standard silence timeout in seconds
CHUNK_DURATION_MS = 32  # Standard chunk duration in milliseconds

# List to store latency values
latency_values = []

# List to store valid latency values (>= minimum latency threshold).
# Latencies below the threshold are considered reverse barge-ins and excluded
# from valid-latency averages to avoid counting non-turns.
valid_latency_values = []

# Global variable to track timestamps
timestamps = {"input_audio_file_end": None, "first_response_after_input": None}

# Global glitch detection
glitch_detected = False

# Global flag and event for controlling silence sending
silence_control = {
    "running": False,
    "event": asyncio.Event(),
    "audio_params": None,  # Will store (frame_rate, n_channels, chunk_size)
}

# Global control for continuous operation
continuous_control = {
    "running": True,
    "collecting_metrics": False,
    "start_time": None,
    "test_duration": 100,  # Default 100 seconds
    "threshold": 0.5,  # Default threshold for valid latency
}


# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    continuous_control["running"] = False
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Constants
DEFAULT_OUTPUT_WAV = "bot_response.wav"


def write_audio_to_wav(data, wf, create_new_file=False, output_file=DEFAULT_OUTPUT_WAV):
    """Write audio data to WAV file."""
    try:
        # Parse protobuf frame
        try:
            proto = frames_pb2.Frame.FromString(data)
            which = proto.WhichOneof("frame")
            if which is None:
                return wf, None, None, None
        except Exception as e:
            log_error(f"Failed to parse protobuf frame: {e}")
            return wf, None, None, None

        args = getattr(proto, which)
        sample_rate = getattr(args, "sample_rate", 16000)
        num_channels = getattr(args, "num_channels", 1)
        audio_data = getattr(args, "audio", None)
        if audio_data is None:
            return wf, None, None, None

        # Extract raw audio data from WAV format if needed
        try:
            with io.BytesIO(audio_data) as buffer, wave.open(buffer, "rb") as wav_file:
                audio_data = wav_file.readframes(wav_file.getnframes())
                sample_rate = wav_file.getframerate()
                num_channels = wav_file.getnchannels()
        except Exception:
            # If not WAV format, use audio_data as-is
            pass

        # Create WAV file if needed
        if create_new_file and wf is None:
            try:
                wf = wave.open(output_file, "wb")  # noqa: SIM115
                wf.setnchannels(num_channels)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
            except Exception as e:
                log_error(f"Failed to create WAV file {output_file}: {e}")
                return None, None, None, None

        # Write audio data directly
        if wf is not None:
            try:
                wf.writeframes(audio_data)
            except Exception as e:
                log_error(f"Failed to write audio data: {e}")
                return None, None, None, None

        return wf, sample_rate, num_channels, audio_data
    except Exception as e:
        log_error(f"Unexpected error in write_audio_to_wav: {e}")
        return wf, None, None, None


async def send_audio_file(websocket, file_path):
    """Send audio file content with streaming simulation."""
    # Pause silence sending while we send the real audio
    silence_control["event"].set()

    try:
        if not os.path.exists(file_path):
            log_error(f"Input audio file not found: {file_path}")
            return

        try:
            with wave.open(file_path, "rb") as wav_file:
                n_channels = wav_file.getnchannels()
                frame_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()

                # Store audio parameters for silence generation
                chunk_size = int((frame_rate * n_channels * CHUNK_DURATION_MS) / 1000) * sample_width
                silence_control["audio_params"] = (
                    frame_rate,
                    n_channels,
                    chunk_size,
                )

                # Stream the audio file
                frames_sent = 0
                while True:
                    try:
                        chunk = wav_file.readframes(chunk_size // sample_width)
                        if not chunk:
                            break
                        audio_frame = frames_pb2.AudioRawFrame(
                            audio=chunk, sample_rate=frame_rate, num_channels=n_channels
                        )
                        frame = frames_pb2.Frame(audio=audio_frame)
                        await websocket.send(frame.SerializeToString())
                        frames_sent += 1
                        await asyncio.sleep(CHUNK_DURATION_MS / 1000)
                    except Exception as e:
                        log_error(f"Error sending audio frame {frames_sent}: {e}")
                        raise  # Re-raise to handle in outer try block
        except wave.Error as e:
            log_error(f"Failed to read WAV file {file_path}: {e}")
            return
    except Exception as e:
        log_error(f"Error in send_audio_file: {e}")
        return
    finally:
        # Always record when input audio ends and resume silence sending
        timestamps["input_audio_file_end"] = datetime.datetime.now()
        print(f"User stopped speaking at: {timestamps['input_audio_file_end'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        silence_control["event"].clear()


async def silence_sender_loop(websocket):
    """Background task to continuously send silence when no other audio is being sent."""
    silence_control["running"] = True
    print("Silence sender loop started")
    consecutive_errors = 0
    max_consecutive_errors = 5

    try:
        while silence_control["running"]:
            try:
                # Wait until we're allowed to send silence
                if silence_control["event"].is_set() or silence_control["audio_params"] is None:
                    await asyncio.sleep(0.1)  # Short sleep to avoid CPU spinning
                    continue

                # Extract audio parameters
                frame_rate, n_channels, chunk_size = silence_control["audio_params"]

                # Send a chunk of silence
                silent_chunk = b"\x00" * chunk_size
                audio_frame = frames_pb2.AudioRawFrame(
                    audio=silent_chunk, sample_rate=frame_rate, num_channels=n_channels
                )
                frame = frames_pb2.Frame(audio=audio_frame)
                await websocket.send(frame.SerializeToString())
                await asyncio.sleep(CHUNK_DURATION_MS / 1000)

                # Reset error counter on successful send
                consecutive_errors = 0

            except ConnectionClosed:
                print("WebSocket connection closed in silence sender loop")
                break
            except Exception as e:
                consecutive_errors += 1
                print(f"Error in silence sender loop (attempt {consecutive_errors}/{max_consecutive_errors}): {e}")

                # If too many consecutive errors, stop the loop
                if consecutive_errors >= max_consecutive_errors:
                    print(f"Too many consecutive errors ({consecutive_errors}), stopping silence sender")
                    break

                # Brief pause before retry to avoid overwhelming the system
                await asyncio.sleep(1.0)

    except Exception as e:
        print(f"Fatal error in silence sender loop: {e}")
    finally:
        print("Silence sender loop stopped")
        silence_control["running"] = False


async def receive_audio(
    websocket,
    wf=None,
    create_new_file=True,
    is_after_input=False,
    output_wav=DEFAULT_OUTPUT_WAV,
    is_initial=False,
    timeout=1.0,
):
    """Receive audio data and handle streaming playback simulation."""
    global glitch_detected

    if is_initial:
        print("Waiting up to 5 seconds for initial bot introduction audio if available...")
        try:
            # Wait for first data packet with 5 second timeout
            data = await asyncio.wait_for(websocket.recv(), timeout=5.0)
        except TimeoutError:
            print("No initial bot introduction received after 5 seconds, continuing...")
            return wf
    else:
        # For non-initial audio, receive normally
        data = await websocket.recv()

    try:
        # Record first response timestamp if after input
        if is_after_input:
            timestamps["first_response_after_input"] = datetime.datetime.now()
            formatted_time = timestamps["first_response_after_input"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print(f"Bot started speaking at {formatted_time}")

        # Initialize playback buffer simulation
        playback_buffer_duration = 0.0  # Amount of audio buffered (seconds)
        last_update_time = None  # Last time buffer was updated
        chunk_count = 0
        last_data_time = time.time()

        # Process all chunks uniformly
        while True:
            # Process current chunk
            wf, sample_rate, num_channels, audio_data = write_audio_to_wav(data, wf, create_new_file, output_wav)

            # Update buffer and detect starvation
            if audio_data and sample_rate and num_channels:
                current_time = time.time()
                chunk_count += 1
                bytes_per_sample = 2  # Assuming 16-bit audio
                samples_in_chunk = len(audio_data) // (num_channels * bytes_per_sample)
                chunk_duration_seconds = samples_in_chunk / sample_rate

                # Update buffer: consume played audio, then check for starvation
                starvation_detected = False
                starvation_duration = 0.0

                if last_update_time is not None:
                    time_elapsed = current_time - last_update_time
                    playback_buffer_duration -= time_elapsed

                    # Detect buffer starvation (underrun) - only mark as glitch if >= 20ms gap
                    if playback_buffer_duration < -0.020:  # 20ms threshold
                        starvation_duration = -playback_buffer_duration
                        starvation_detected = True
                        glitch_detected = True
                        playback_buffer_duration = 0

                # Log starvation BEFORE chunk details
                if starvation_detected:
                    print(
                        f"⚠️  STARVATION at chunk #{chunk_count}: buffer ran dry for "
                        f"{starvation_duration * 1000:.1f}ms before this chunk arrived (AUDIO GLITCH)"
                    )

                # Add new chunk to buffer
                playback_buffer_duration += chunk_duration_seconds
                last_update_time = current_time

                print(
                    f"Client chunk #{chunk_count}: size={len(audio_data)}B, "
                    f"duration={chunk_duration_seconds * 1000:.1f}ms, "
                    f"buffer={playback_buffer_duration * 1000:.1f}ms"
                )

            # Receive next chunk
            try:
                data = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                last_data_time = time.time()
            except TimeoutError:
                # Check if silence duration exceeds threshold
                if time.time() - last_data_time >= SILENCE_TIMEOUT:
                    return wf
            except Exception as e:
                log_error(f"Error receiving audio data: {e}")
                if wf is not None and create_new_file:
                    try:
                        wf.close()
                    except Exception as close_error:
                        log_error(f"Error closing WAV file: {close_error}")
                return None
    except Exception as e:
        log_error(f"Fatal error in receive_audio: {e}")
        if wf is not None and create_new_file:
            try:
                wf.close()
            except Exception as close_error:
                log_error(f"Error closing WAV file: {close_error}")
        return None


async def process_conversation_turn(websocket, audio_file_path, wf, turn_index, output_wav="bot_response.wav"):
    """Process a single conversation turn with the given audio file."""
    print(f"\n----- Processing conversation turn {turn_index + 1} -----")

    # Reset timestamps for this turn
    timestamps["input_audio_file_end"] = None
    timestamps["first_response_after_input"] = None

    # Start both sending and receiving in parallel for realistic latency measurement
    print(f"Sending user input audio from {audio_file_path}...")

    # Start sending audio file in background
    send_task = asyncio.create_task(send_audio_file(websocket, audio_file_path))

    # Start receiving bot response immediately (parallel to sending)
    receive_task = asyncio.create_task(
        receive_audio(websocket, wf=wf, create_new_file=(wf is None), is_after_input=True, output_wav=output_wav)
    )

    # Wait for both tasks to complete
    wf = await receive_task
    await send_task  # Ensure sending is also complete

    # Calculate and store latency only if we're collecting metrics
    if continuous_control["collecting_metrics"]:
        latency = None
        if timestamps["input_audio_file_end"] is not None and timestamps["first_response_after_input"] is not None:
            latency = (timestamps["first_response_after_input"] - timestamps["input_audio_file_end"]).total_seconds()
            print(f"Latency for Turn {turn_index + 1}: {latency:.3f} seconds")
            latency_values.append(latency)

            # Classify latency against minimum valid threshold
            if latency >= continuous_control["threshold"]:
                valid_latency_values.append(latency)
            else:
                print(
                    f"Reverse barge-in detected: latency {latency:.3f}s "
                    f"< minimum {continuous_control['threshold']:.3f}s"
                )

    return wf


async def continuous_audio_loop(websocket, audio_files, wf, output_wav):
    """Continuously loop through audio files until stopped."""
    turn_index = 0

    while continuous_control["running"]:
        # Check if we should start collecting metrics
        if (
            continuous_control["start_time"]
            and time.time() >= continuous_control["start_time"]
            and not continuous_control["collecting_metrics"]
        ):
            continuous_control["collecting_metrics"] = True
            print(f"\n=== STARTING METRICS COLLECTION at {datetime.datetime.now().strftime('%H:%M:%S')} ===")

        # Check if we should stop collecting metrics
        if (
            continuous_control["start_time"]
            and continuous_control["collecting_metrics"]
            and time.time() >= continuous_control["start_time"] + continuous_control["test_duration"]
        ):
            print(f"\n=== STOPPING METRICS COLLECTION at {datetime.datetime.now().strftime('%H:%M:%S')} ===")
            continuous_control["collecting_metrics"] = False
            continuous_control["running"] = False
            break

        # Process current audio file
        audio_file = audio_files[turn_index % len(audio_files)]
        wf = await process_conversation_turn(websocket, audio_file, wf, turn_index, output_wav)
        turn_index += 1

        # Small delay between turns to prevent overwhelming the system
        await asyncio.sleep(0.1)

    return wf


async def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Speech-to-speech client with latency measurement")
    parser.add_argument(
        "--stream-id", type=str, default=str(uuid.uuid4()), help="Unique stream ID (default: random UUID)"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="WebSocket server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8100, help="WebSocket server port (default: 8100)")
    parser.add_argument(
        "--output-dir", type=str, default="./results", help="Directory to store output files (default: ./results)"
    )
    parser.add_argument("--start-delay", type=float, default=0, help="Delay in seconds before starting (default: 0)")
    parser.add_argument(
        "--metrics-start-time",
        type=float,
        default=0,
        help="Unix timestamp when to start collecting metrics (default: 0)",
    )
    parser.add_argument(
        "--test-duration", type=float, default=100, help="Duration in seconds to collect metrics (default: 100)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold for valid average latency calculation (default: 0.5)"
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Construct WebSocket URI with unique stream ID
    uri = f"ws://{args.host}:{args.port}/ws/{args.stream_id}"

    # Output file paths
    output_wav = os.path.join(args.output_dir, f"bot_response_{args.stream_id}.wav")
    output_results = os.path.join(args.output_dir, f"latency_results_{args.stream_id}.json")

    print(f"Starting client with stream ID: {args.stream_id}")
    print(f"WebSocket URI: {uri}")
    print(f"Start delay: {args.start_delay} seconds")
    print(f"Metrics start time: {args.metrics_start_time}")
    print(f"Test duration: {args.test_duration} seconds")
    print(f"Latency threshold (minimum valid turn): {args.threshold} seconds")

    # Set up timing controls
    if args.start_delay > 0:
        print(f"Waiting {args.start_delay} seconds before starting...")
        await asyncio.sleep(args.start_delay)

    if args.metrics_start_time > 0:
        continuous_control["start_time"] = args.metrics_start_time
        continuous_control["test_duration"] = args.test_duration
        print(f"Will start collecting metrics at timestamp {args.metrics_start_time}")

    # Set threshold for valid latency calculation
    continuous_control["threshold"] = args.threshold

    # Define the array of input audio files
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_files_dir = os.path.join(script_dir, "audio_files")

    input_audio_files = [
        os.path.join(audio_files_dir, "query_1.wav"),
        os.path.join(audio_files_dir, "query_2.wav"),
        os.path.join(audio_files_dir, "query_3.wav"),
        os.path.join(audio_files_dir, "query_4.wav"),
        os.path.join(audio_files_dir, "query_5.wav"),
        os.path.join(audio_files_dir, "query_6.wav"),
        os.path.join(audio_files_dir, "query_7.wav"),
        os.path.join(audio_files_dir, "query_8.wav"),
        os.path.join(audio_files_dir, "query_9.wav"),
        os.path.join(audio_files_dir, "query_10.wav"),
    ]

    # Clear any previous values
    latency_values.clear()
    valid_latency_values.clear()

    # Initialize silence control
    silence_control["event"] = asyncio.Event()
    silence_control["event"].set()  # Start with silence sending paused

    try:
        async with websockets.connect(uri) as websocket:
            # First, try to receive any initial output audio
            wf = await receive_audio(
                websocket,
                wf=None,
                create_new_file=True,
                is_after_input=False,
                output_wav=output_wav,
                is_initial=True,
            )

            # Start the silence sender task
            asyncio.create_task(silence_sender_loop(websocket))

            # Start continuous audio loop
            wf = await continuous_audio_loop(websocket, input_audio_files, wf, output_wav)

            # Clean up and stop the silence sender
            silence_control["running"] = False
            silence_control["event"].set()  # Make sure it's not waiting
            await asyncio.sleep(0.2)  # Give it time to exit cleanly

            if wf is not None:
                wf.close()
                print(f"All output saved to {output_wav}")

    except ConnectionClosed:
        # Normal WebSocket closure, not an error
        pass
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        # Always save results, regardless of how the connection ended
        if latency_values:
            avg_latency = sum(latency_values) / len(latency_values)

            # Calculate valid (thresholded) average latency
            valid_avg_latency = None
            if valid_latency_values:
                valid_avg_latency = sum(valid_latency_values) / len(valid_latency_values)

            print("\n----- Final Latency Summary -----")
            print(f"Average Latency across {len(latency_values)} turns: {avg_latency:.3f} seconds")

            if valid_avg_latency is not None:
                print(
                    f"Valid Average Latency (>= {args.threshold}s) across {len(valid_latency_values)} turns: "
                    f"{valid_avg_latency:.3f} seconds"
                )
            else:
                print(f"Valid Average Latency: No latencies >= {args.threshold}s threshold")

            # Calculate reverse barge-ins (latencies below threshold)
            reverse_barge_ins_count = len(latency_values) - len(valid_latency_values)
            print(f"Reverse Barge-Ins Detected: {reverse_barge_ins_count} latencies below {args.threshold}s threshold")

            # Report glitch detection results
            if glitch_detected:
                print("⚠️  AUDIO GLITCHES DETECTED: Audio chunks arrived with gaps larger than playback time")
            else:
                print("✅ No audio glitches detected: Audio streaming was smooth")

            print("----------------------------------------")

            # Save results to JSON file
            results = {
                "stream_id": args.stream_id,
                "individual_latencies": latency_values,
                "average_latency": avg_latency,
                "valid_latencies": valid_latency_values,
                "valid_average_latency": valid_avg_latency,
                "threshold": args.threshold,
                "num_turns": len(latency_values),
                "num_valid_turns": len(valid_latency_values),
                "reverse_barge_ins_count": len(latency_values) - len(valid_latency_values),
                "glitch_detected": glitch_detected,
                "timestamp": datetime.datetime.now().isoformat(),
                "metrics_start_time": continuous_control["start_time"],
                "test_duration": continuous_control["test_duration"],
            }

            with open(output_results, "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
