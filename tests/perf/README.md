# Performance Testing

This directory contains tools for evaluating the voice agent pipeline's latency and scalability/throughput under various loads. These tests simulate real-world scenarios where multiple users interact with the voice agent simultaneously.

## Prerequisites: Provide Your Own Audio Files

You must prerecord and add your own audio files before running the performance tests.

- Create an `audio_files/` directory in this folder (`tests/perf/`) if it does not exist.
- Add one or more WAV files containing spoken queries (see [Create your audio file dataset](#create-your-audio-file-dataset) below for format and recording guidelines).

## What the Tests Do

The performance tests:

- Open WebSocket clients that simulate user interactions
- Use pre-recorded audio files from `audio_files/` as user queries
- Send these queries to the voice agent pipeline and measure response times
- Track various latency metrics including end-to-end latency, component-wise breakdowns
- Can simulate multiple concurrent clients to test scaling
- Detect any audio glitches or reverse barge-in's during processing

## Running Performance Tests

### 1. Start the Voice Agent Pipeline

First, start the voice agent pipeline and capture server logs for analysis.
See the prerequisites and setup instructions in `examples/voice_agent_websocket/README.md` before proceeding.

#### If Using Docker

From examples/voice_agent_websocket/ directory run:

```bash
# Start the services
docker compose up -d

# Capture logs and save them into a file
docker compose logs -f python-app > bot_logs_test1.txt 2>&1 &
```

Before starting a new performance run:

```bash
# Stop the previous background log capture process
# Find the process ID and kill it
pkill -f "docker compose logs -f python-app"
# Clear existing Docker logs
sudo truncate -s 0 /var/lib/docker/containers/$(docker compose ps -q python-app)/$(docker compose ps -q python-app)-json.log
# Restart the python-app container
docker compose restart python-app
```

#### If Using Python Environment

From examples/voice_agent_websocket/ directory run:

```bash
python bot.py > bot_logs_test1.txt 2>&1 &
```

### 2. Run the Multi-Client Benchmark

Ensure `audio_files/` exists and contains at least one WAV file (see [Create your audio file dataset](#create-your-audio-file-dataset)).

```bash
./run_multi_client_benchmark.sh --host 0.0.0.0 --port 8100 --clients 10 --test-duration 150
```

Parameters:

- `--host`: The host address (default: 0.0.0.0)
- `--port`: The port where your voice agent is running (default: 8100)
- `--clients`: Number of concurrent clients to simulate (default: 1)
- `--test-duration`: Duration of the test in seconds (default: 150)

The script will:

1. Start the specified number of concurrent clients
2. Simulate user interactions using audio files
3. Measure latencies and detect audio glitches
4. Save detailed results in the `results` directory as JSON files
5. Output a summary to the console

### 3. Analyze Component-wise Latency

After the benchmark completes, analyze the server logs for detailed latency breakdowns:

```bash
python ttfb_analyzer.py <relative_path_to_bot_logs_test1.txt>
```

This will show:

- Per-client latency metrics for LLM, TTS, and ASR components
- Number of calls made by each client
- Overall averages and P95 values
- Component-wise timing breakdowns

## Understanding the Results

The metrics measured include:

- **LLM TTFB**: Time to first byte from the LLM model
- **TTS TTFB**: Time to first byte from the TTS model
- **ASR Lat**: Compute latency of the ASR model
- **LLM 1st**: Time taken to generate first complete sentence from LLM
- **Calls**: Number of API calls made to each service

The results help identify:

- Performance bottlenecks in specific components
- Scaling behavior under concurrent load
- Potential audio quality issues
- Potential reverse barge-in scenarios
- Overall system responsiveness

## Troubleshooting

If you see no results, possible reasons include:
- Model endpoints are unreachable
- Audio files are in the wrong format (must be 16 kHz, mono, linear PCM int16 WAV)

Check server logs for details.

## Create your audio file dataset

You must provide your own audio files in the `audio_files/` directory; Use it for generic queries or for your specific use case. Follow these guidelines:

- Record the query.
- Trim all trailing silence from the end (e.g., using Audacity). This is critical for correct latency measurement: the scripts measure latency from the end of the audio file to the time the bot’s response is received. Ensure the end of the file coincides with the end of the spoken query. The scripts will insert/send silence between files automatically.
- Save files as 16 kHz, mono, linear PCM (int16) WAV.
