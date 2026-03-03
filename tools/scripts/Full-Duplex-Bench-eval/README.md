## Full‑Duplex‑Bench Eval – Nvidia Voice Agent Inference Client

This folder contains a simple inference client that streams audio to your WebSocket‑based voice agent and saves the returned audio, so you can evaluate it with the upstream benchmark tools from [Full‑Duplex‑Bench](https://github.com/DanielLin94144/Full-Duplex-Bench).

### What this provides
- **inference.py**: Sends input audio to `ws://<host>:<port>/ws/benchmark` and writes the model’s audio response back to disk.
- Handles basic preprocessing (mono, 16 kHz, int16) and postprocessing to align output with the input duration.

## Quickstart
1) Install dependencies:

```bash
pip install numpy resampy soundfile websockets pipecat-ai
```

2) Start the WebSocket voice agent server. For reference, you can use the example app [voice_agent_websocket](../../../examples/voice_agent_websocket/README.md).

   Make sure it’s reachable at the host and port you plan to use (default example runs on port 8100). The client connects to the route `/ws/benchmark`.

3) Prepare your dataset directory structure (see “Dataset layout” below), and then run the client(from currenty dir Full-Duplex-Bench eval/):

```bash
python inference.py \
  --input_dir /path/to/samples \
  --host 127.0.0.1 \
  --port 8100
```

Optional: retry only specific sample IDs later:

```bash
python inference.py \
  --input_dir /path/to/samples \
  --host 127.0.0.1 \
  --port 8100 \
  --retry_samples 1 5 10
```

## Dataset layout
The client expects numeric subdirectories under `--input_dir`. In each sample directory:
- `input.wav` is always processed if present → outputs `output.wav`.
- `clean_input.wav` is also processed if present → outputs `clean_output.wav`.

Example:

```text
/path/to/samples/
  1/
    input.wav
    clean_input.wav
  2/
    input.wav
  3/
    clean_input.wav
```

After running, you’ll find corresponding `output.wav` and/or `clean_output.wav` in each sample directory.

## CLI usage
```text
--input_dir       (str, required)  Root directory containing numeric sample folders
--host            (str, required)  Server hostname or IP (e.g., 127.0.0.1)
--port            (int, required)  Server port (e.g., 8100)
--retry_samples   (ints, optional) Specific sample IDs to process (space‑separated)
```

Notes:
- Audio is streamed at 16 kHz mono, int16. The client will resample and convert as needed.
- The client sends a brief silence tail after the input audio to allow the server to finish responding.

## How it works (brief)
- Preprocess input audio to 16 kHz mono int16.
- Stream in 32 ms chunks with real‑time pacing to `/ws/benchmark`.
- Receive audio frames, place them on a time axis, and write trimmed output aligned to the input duration.


## Credits
- Benchmark datasets and evaluation tooling: [Full‑Duplex‑Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)
- This client is provided here only to interface your Nvidia Voice Agent server with that benchmark.