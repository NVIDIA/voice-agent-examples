# BigBench Audio Eval

Tools to run and evaluate **Big Bench Audio** experiments: download the dataset, run speech or text inference, transcribe outputs, and compute accuracy.

## Key Features

- **Dataset**: [ArtificialAnalysis/big_bench_audio](https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio) — audio questions with official answers
- **Speech-to-speech pipeline**: Stream audio to your voice agent, transcribe with Riva ASR, evaluate with an LLM judge
- **Text-inference pipeline**: Transcribe inputs, run LLM text inference, then evaluate
- **Reproducible setup**: Dependencies managed with [uv](https://docs.astral.sh/uv/) and locked in `uv.lock`

## Directory Layout

One folder per sample ID (e.g. `0/`, `1/`, …). Each folder can contain:

- `input.mp3` / `input.wav` — question audio
- `meta.json` — id, category, official_answer, file_name
- `question.txt` — transcript of question (from transcribe script)
- `output.wav` — model speech output (speech pipeline only)
- `response.txt` — model text response or transcript of `output.wav`
- `result.txt` — CORRECT/INCORRECT (from eval)

## Prerequisites and Setup

- **Python 3.10+**, **ffmpeg** (for MP3→WAV in speech pipeline)
- Dependencies are managed with [uv](https://docs.astral.sh/uv/); install and run from `tools/scripts/AA-BigBenchAudio-eval/`:

1. Install dependencies:

   ```bash
   # Install uv if needed: https://docs.astral.sh/uv/getting-started/installation/
   uv sync
   ```

2. Run scripts with `uv run` (or activate the project venv):

   ```bash
   uv run python download_dataset.py --input_dir ./datasets/bigbench_audio --split train
   ```

## Downloading the Dataset

```bash
uv run python download_dataset.py --input_dir ./datasets/bigbench_audio --split train
```

**Flags**: `--input_dir` (required), `--split` (default: `train`), `--token` (optional). For gated repos or rate limits: `huggingface-cli login`, or set `HF_TOKEN`.

## Experiment 1: Speech-to-speech pipeline

Audio question → voice agent → audio response → transcribe → LLM judge → accuracy.

1. **Prerequisites**

   Start your **voice agent server** (audio in, audio out) and note `host` and `port`. Follow [examples/voice_agent_websocket/README.md](../../../examples/voice_agent_websocket/README.md) to start the server. The benchmark's websocket client will stream `input.wav` and receive `output.wav` per sample.

2. **Preprocess**

   Convert downloaded MP3s to 16 kHz mono WAV for the server. This configuration is required by the **Nemotron voice agent**; you can change the script (e.g. `SAMPLE_RATE` in `speech-inference.py`) to match your voice agent's input format.

   ```bash
   uv run python speech-inference.py --input_dir ./datasets/bigbench_audio --preprocess
   ```

3. **Run speech inference**

   Point at the dataset directory; the script streams each sample to the server and saves `output.wav`:

   ```bash
   uv run python speech-inference.py --input_dir ./datasets/bigbench_audio --inference --host YOUR_HOST --port YOUR_PORT
   ```

4. **Transcribe input and output**

   Transcribe `input.wav` → `question.txt` and `output.wav` → `response.txt`. Uses **NVIDIA Riva ASR** (gRPC); model `parakeet-1.1b-en-US-asr-offline` (override with `--model`). Point `--host` and `--port` at your Riva endpoint (cloud or local). Requires `nvidia-riva-client`, `grpcio`.

   ### Riva ASR offline deployment

   **Offline Docker deploy**: [Parakeet CTC 1.1B ASR – Deploy](https://build.nvidia.com/nvidia/parakeet-ctc-1_1b-asr/deploy) with `NIM_TAGS_SELECTOR=mode=ofl`. Prerequisites: Docker, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), [NGC API key](https://catalog.ngc.nvidia.com/).

   ```bash
   export NVIDIA_API_KEY=YOUR_NGC_OR_NVIDIA_API_KEY
   docker run -it --rm --name riva-asr-parakeet-ofl \
     --runtime=nvidia --gpus '"device=0"' --shm-size=8GB \
     -e NGC_API_KEY=${NVIDIA_API_KEY} \
     -e NIM_HTTP_API_PORT=9000 -e NIM_GRPC_API_PORT=50051 \
     -e NIM_TAGS_SELECTOR=mode=ofl \
     -p 9000:9000 -p 50051:50051 \
     nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us:1.3.0
   ```

   Then run `transcribe.py` with `--host` and `--port` (default 50051):

   ```bash
   uv run python transcribe.py --input_dir ./datasets/bigbench_audio
   ```

5. **Run evaluation**

   Compares `response.txt` and `question.txt` to the official answer via an **LLM judge**, writes `result.txt` (CORRECT/INCORRECT) per sample. Judge: **Anthropic Claude 3.5 Sonnet (Oct '24)** per [Artificial Analysis methodology](https://huggingface.co/blog/big-bench-audio-release#evaluation-methodology). Set `EVAL_API_URL` and `EVAL_API_KEY` for your judge endpoint.

   ```bash
   EVAL_API_URL=https://.../invoke EVAL_API_KEY=your_key uv run python eval.py --input_dir ./datasets/bigbench_audio
   ```

6. **Find LLM judge anomalies**

   List samples where `result.txt` is not exactly CORRECT or INCORRECT:

   ```bash
   uv run python find_invalid_results.py --input_dir ./datasets/bigbench_audio
   ```

   The script prints a suggested retry command with `--retry_samples`.

7. **Final accuracy**

   Report counts and accuracy percentage:

   ```bash
   uv run python analyze_results.py --input_dir ./datasets/bigbench_audio
   ```

## Experiment 2: Text-inference pipeline

Transcript → LLM text response → LLM judge → accuracy.

1. **Transcribe dataset inputs**

   Produce `question.txt` from each `input.wav`. Same setup as speech step 4: Riva ASR; see [Riva ASR offline deployment](#riva-asr-offline-deployment). Need `input.wav` first: run download + `speech-inference.py --preprocess` if needed.

   ```bash
   uv run python transcribe.py --input_dir ./datasets/bigbench_audio
   ```

2. **Run text inference**

   Point at the dataset dir and your local or cloud LLM endpoint (configure in the script or env). This writes `response.txt` per sample.

   ```bash
   uv run python text-inference.py --input_dir ./datasets/bigbench_audio
   ```

3. **Evaluate** (same as speech steps 5–7; set `EVAL_API_URL`, `EVAL_API_KEY`):

   ```bash
   uv run python eval.py --input_dir ./datasets/bigbench_audio
   uv run python find_invalid_results.py --input_dir ./datasets/bigbench_audio
   uv run python eval.py --input_dir ./datasets/bigbench_audio --retry_samples <ids>
   uv run python analyze_results.py --input_dir ./datasets/bigbench_audio
   ```

## Reference Results

Accuracy (%) on Big Bench Audio for text-only (standalone LLM) vs speech-to-speech (LLM in voice agent pipeline):

| Model / API | Reasoning Mode | Text Only Standalone LLM (%) | LLM In Voice Agent Pipeline (%) |
|-------------|----------------|-----------------------------|--------------------------------|
| Nemotron 49B (`nvidia/llama-3.3-nemotron-super-49b-v1.5`) | Reasoning ON | 91.90 | 81.30 |
| Nemotron 49B | Reasoning OFF | 82.70 | 60.30 |
| Nemotron 30B (`nvidia/nemotron-3-nano`) | Reasoning ON, Budget 500 | 78.76 | 75.60 |
| Nemotron 30B | Reasoning OFF | 56.50 | 50.40 |
