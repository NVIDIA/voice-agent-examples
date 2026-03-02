<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD 2-Clause License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# NVIDIA Flowershop Voice Agent with NeMo Agent Toolkit

This example demonstrates an end-to-end (E2E) intelligent voice assistant using [NeMo Agent toolkit](https://docs.nvidia.com/aiqtoolkit/latest/index.html) with WebRTC for real-time speech-to-speech interaction. It showcases a flower shop assistant with custom function registration, voice processing pipeline using NVIDIA Pipecat, and comprehensive observability with Phoenix tracing.

## Key Features

- **Custom Function Registration**: Demonstrates custom function creation using the NeMo Agent toolkit registration system
- **Flower Shop Assistant**: Interactive menu browsing, pricing, and cart management functionality  
- **ReWOO Agent**: Uses planning-based approach for efficient task decomposition and execution
- **Voice-to-Voice Pipeline**: Real-time WebRTC-based speech interaction using NVIDIA Pipecat
- **RESTful API Deployment**: Production-ready API deployment using `nat serve`
- **Phoenix Tracing**: Comprehensive observability with Phoenix tracing and monitoring
- **Workflow Profiling**: Built-in profiling capabilities to analyze performance bottlenecks and optimize workflows
- **Evaluation System**: Comprehensive evaluation tools to validate and maintain accuracy of agentic workflows

## Prerequisites and Setup

1. Clone the voice-agent-examples repository:

   ```bash
   git clone https://github.com/NVIDIA/voice-agent-examples.git
   ```

2. Navigate to the example directory:

   ```bash
   cd voice-agent-examples/examples/nat_agent
   ```

3. Copy and configure the environment file:

   ```bash
   cp env.example .env  # and add your credentials
   ```

4. Setup API keys in .env file:
   
   Ensure you have the required API keys:
   - NVIDIA_API_KEY - Required for accessing NIM ASR, TTS and LLM models

   Refer to [https://build.nvidia.com/](https://build.nvidia.com/) for generating your API keys.

   Edit the .env file to add your keys or export using:

   ```bash
   export NVIDIA_API_KEY=<YOUR_API_KEY>
   ```

5. Deploy Coturn Server if required

    If you want to share widely or want to deploy on cloud platforms, you will need to setup coturn server. Follow instructions below for modifications required in example code for using coturn:

    Update HOST_IP_EXTERNAL with your machine IP and run the below command:

    ```bash
    docker run -d --network=host instrumentisto/coturn -n --verbose --log-file=stdout --external-ip=<HOST_IP_EXTERNAL>  --listening-ip=0.0.0.0  --lt-cred-mech --fingerprint --user=admin:admin --no-multicast-peers --realm=tokkio.realm.org --min-port=51000 --max-port=52000
    ```

    Add the following configuration to your `bot.py` file to use the coturn server:

    ```python
    ice_servers = [
        IceServer(
            urls="turn:<HOST_IP_EXTERNAL>:3478",
            username="admin",
            credential="admin"
        )
    ]
    ```

    Add the following configuration to your [`webrtc_ui/src/config.ts`](./webrtc_ui/src/config.ts) file to use the coturn server:

    ```typescript
    export const RTC_CONFIG: ConstructorParameters<typeof RTCPeerConnection>[0] = {
        iceServers: [
          {
            urls: "turn:<HOST_IP_EXTERNAL>:3478",
            username: "admin",
            credential: "admin",
          },
        ],
      };
    ```

    For more information, see the turn-server documentation at [https://webrtc.org/getting-started/turn-server](https://webrtc.org/getting-started/turn-server).

    
6. Deploy the application with either of the options (note: Phoenix tracing is enabled by default; see [Advanced: Phoenix Deployment](#advanced-phoenix-deployment) for details).


## Option 1: Deploy Using Docker

### Prerequisites

- You have access and are logged into NVIDIA NGC. For step-by-step instructions, refer to [the NGC Getting Started Guide](https://docs.nvidia.com/ngc/ngc-overview/index.html#registering-activating-ngc-account).

- You have access to an NVIDIA Turing™, NVIDIA Ampere (e.g., A100), NVIDIA Hopper (e.g., H100), NVIDIA Ada (e.g., L40S), or the latest NVIDIA GPU architectures. For more information, refer to [the Support Matrix](https://docs.nvidia.com/nim/riva/asr/latest/support-matrix.html).

- You have Docker installed with support for NVIDIA GPUs. For more information, refer to [the Support Matrix](https://docs.nvidia.com/nim/riva/asr/latest/support-matrix.html).

### Run

```bash
export NGC_API_KEY=nvapi-... # <insert your key>
docker login nvcr.io
```

From the examples/nat_agent directory, run below commands:

```bash
docker compose up --build -d
```

Docker deployment might take 30-45 minutes first time. Once all services are up and running, visit `http://<machine-ip>:9000/` in your browser to start interacting with the application. See the next sections for detailed instructions on interacting with the app.

## Option 2: Deploy Using Python Environment

### Requirements

- Python (>=3.11, <3.13)
- [uv](https://github.com/astral-sh/uv)

All Python dependencies are listed in separate `pyproject.toml` files for agent and bot components.

### Step 1: Deploy the NAT Agent

```bash
# Navigate to agent directory
cd agent
# Create a virtual environment
uv venv
# Install agent dependencies
source .venv/bin/activate
uv sync
uv pip install -e .

# Edit configs/config.yml to update LLM endpoints as per your deployment
# Update the 'llm' section with your specific model endpoints, API keys, and model names

# Start the NAT agent service
nat serve --config_file configs/config.yml --host 0.0.0.0 --port 8000
```

The agent service will start and be available at `http://localhost:8000`. You can view the auto-generated API documentation at `http://localhost:8000/docs`.

**Agent Deployment**: See [agent/README.md](agent/README.md) for comprehensive agent configuration, deployment options, troubleshooting, and advanced features

### Step 2: Start the Voice Agent pipeline

In a new terminal, from the main nat_agent directory:

```bash
# Install bot dependencies
source .venv/bin/activate
uv sync
uv pip install -e .

# Start the voice bot server
python bot.py
```

### Step 3: Start the Voice Bot Interface

Connect through the voice bot interface for real-time speech interaction. For detailed setup instructions, see [WebRTC UI README](../webrtc_ui/README.md)

visit `http://localhost:5173/` in your browser to start interacting with the application. See the next sections for detailed instructions on interacting with the app.

## Start interacting with the application

Note: To enable microphone access in Chrome, go to `chrome://flags/`, enable "Insecure origins treated as secure", add `http://<machine-ip>:9000` (for docker method), `http://localhost:5173/` (for python method) to the list, and restart Chrome.
You can interact with the application through:

1. **REST API**: Use HTTP requests to interact with the agent directly
2. **Console Interface**: Use `nat run` for text-based interaction

### Testing the API

Use curl to test the deployed NAT agent service:

```bash
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "input_message": "What flowers do you have available and what are your prices?"
  }'
```

### Testing with Console

Before testing, update the LLM endpoints in your configuration file to match your deployment:

```bash
cd agent
# Edit configs/config.yml to update LLM endpoints as per your deployment
# Update the 'llm' section with your specific model endpoints, API keys, and model names
```

Test your workflow using the `nat run` command:

```bash
nat run --config_file configs/config.yml --input "Hello, can you show me the menu?"
```

## Project Structure

This example is organized into two main components:

- **`agent/`** - Contains all files required for NAT agent deployment as a standalone service
  - Agent source code, configurations, and deployment files
  - Separate `pyproject.toml` with agent-specific dependencies (NAT, LangChain, etc.)
  - Docker setup for containerized deployment
  - Complete documentation in [agent/README.md](agent/README.md)
  
- **Root directory** - Contains the voice bot interface (`bot.py`, `bot_websocket.py`) 
  - Use `bot.py` for WebRTC-based voice interface for real-time interaction
  - Use `bot_websocket.py` for websocket based voice pipeline, recommended for evaluation and performance testing
  - Separate `pyproject.toml` with bot-specific dependencies (FastAPI, Pipecat, etc.)
  - Integration with NVIDIA Pipecat for voice processing

## Advanced Features

### Enable Phoenix Tracing

Phoenix tracing is already configured in your workflow. Start the Phoenix server to view traces:

```bash
# In a new terminal
phoenix serve
```

Phoenix will be available at:
- **Phoenix UI**: http://0.0.0.0:6006
- **Trace Endpoint**: http://0.0.0.0:6006/v1/traces

**Troubleshooting**: If `phoenix serve` shows any errors, consider deleting the Phoenix database file.

### Switch to WebSocket Pipecat transport
To run performance scripts and voice based evaluation, it will be better to use websocket transport based voice agent pipeline. You can make the switch by uncommenting code in `docker-compose.yml`.
- Change the command for the `python-app` service to use the websocket based NAT agent pipeline from `bot_websocket.py`
- Mount the [websocket based UI page](../voice_agent_websocket/static/) and update the `STATIC_PATH` environment variable in `python-app`
- Comment out the `ui-app` service used for WebRTC UI
- Deploy service using updated `docker-compose.yml` and access websocket UI page at `http://HOST_IP:8100/static/index.html`.

### Profiling and Evaluation

To run the profiler and evaluator, use the `nat eval` command with the workflow configuration file. The profiler will collect usage statistics and the evaluator will assess workflow accuracy, storing results in the output directory specified in the configuration file.

```bash
cd agent
nat eval --config_file configs/config.yml
```

## Bot pipeline customizations

  ### Speculative Speech Processing

  Speculative speech processing reduces bot response latency by working directly on Nemotron Speech ASR early interim user transcripts instead of waiting for final transcripts. This feature only works when using Nemotron Speech ASR. Currently set to true.

  - Toggle using the environment variable `ENABLE_SPECULATIVE_SPEECH`.
    - Docker Compose: set in `python-app.environment` (default is `true`)
      ```yaml
      environment:
        - ENABLE_SPECULATIVE_SPEECH=${ENABLE_SPECULATIVE_SPEECH:-false}
      ```
    - Local run: export before launching
      ```bash
      export ENABLE_SPECULATIVE_SPEECH=false  # or true
      python bot.py
      ```
  - The application will automatically switch processors based on this flag; no code edits needed.
  - See the [Documentation on Speculative Speech Processing](../../docs/SPECULATIVE_SPEECH_PROCESSING.md) for more details.

  ### Switching ASR and TTS Models

  You may customize ASR (Automatic Speech Recognition), Agent (Patient Front Desk Assistant), and TTS (Text-to-Speech) services by configuring environment variables. This allows you to switch between NIM cloud-hosted models and locally deployed models.

  The following environment variables control the endpoints and models:

  - `ASR_SERVER_URL`: Address of the Nemotron Speech ASR (speech-to-text) service (e.g., `localhost:50051` for local, "grpc.nvcf.nvidia.com:443" for [cloud endpoint](https://build.nvidia.com/)).
  - `TTS_SERVER_URL`: Address of the Nemotron Speech TTS (text-to-speech) service. (e.g., `localhost:50051` for local, "grpc.nvcf.nvidia.com:443" for [cloud endpoint](https://build.nvidia.com/)).

  You can set model, language, and voice using the `ASR_MODEL_NAME`, `TTS_MODEL_NAME`, `ASR_LANGUAGE`, `TTS_LANGUAGE`, and `TTS_VOICE_ID` environment variables.

  Update these variables in your Docker Compose configuration to match your deployment and desired models. For more details on available models and configuration options, refer to the [NIM NVIDIA Magpie](https://build.nvidia.com/nvidia/magpie-tts-multilingual), [NIM NVIDIA Parakeet](https://build.nvidia.com/nvidia/parakeet-ctc-1_1b-asr/api) documentation.

#### Example: Setting up Zero-shot Magpie Latest Model

Follow these steps to configure and use the latest Zero-shot Magpie TTS model:

1. **Update environment variables**

   Set `TTS_DOCKER_IMAGE` to actual image tag `<magpie-tts-zeroshot-image:version>`.

   Then, configure the settings found in the `Zero-shot TTS Magpie Model` section of your env file.

   Make sure your NVIDIA_API_KEY, with access to the zero-shot model, is correctly entered in your `.env` file.

2. **Configuring Zero-shot Audio Prompt**

   To use your own custom voice with zero-shot TTS:

   - Place your desired audio sample in the workspace directory.
   - Mount the audio file into your container by adding a volume in your `docker-compose.yml` under the `python-app` service:

     ```yaml
     services:
       python-app:
         # ... existing code ...
         volumes:
           - ./audio_prompts:/app/audio_prompts
     ```

   - In your `.env` file, set the `ZERO_SHOT_AUDIO_PROMPT` variable to its path (relative to your application's root):
     - `ZERO_SHOT_AUDIO_PROMPT=audio_prompts/voice_sample.wav`  # Path relative to app root

   Note: The zero-shot audio prompt is only required when using the Magpie Zero-shot model. For standard Magpie multilingual models, this configuration should be omitted.

3. **Set TTS Environment Variables**

   In `.env` (for `python-app`), update:

   ```bash
   TTS_VOICE_ID=Magpie-ZeroShot.Female-1
   TTS_MODEL_NAME=magpie_tts_ensemble-Magpie-ZeroShot
   ```

### Enabling Audio Dumps

To capture raw audio streams for debugging ASR/TTS quality issues:

```bash
# In .env file
ENABLE_ASR_AUDIO_DUMP=true   # Save input audio
ENABLE_TTS_AUDIO_DUMP=true   # Save output audio
AUDIO_DUMP_PATH=./audio_dumps
```

Audio files are saved as WAV format with stream ID for correlation.

**Permission Issues**: If Docker creates the `audio_dumps` folder with different user permissions, accessing it later via Python deployment or another Docker container may fail. To resolve:

- Pre-create the folder before enabling: `mkdir -p ./audio_dumps`
- Or fix existing permissions: `sudo chown -R $(id -u):$(id -g) ./audio_dumps`



