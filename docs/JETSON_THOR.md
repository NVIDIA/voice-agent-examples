# Deploying Voice Agent on Jetson Thor

This guide covers deploying the NVIDIA Voice Agent on Jetson Thor using Docker Compose.

## Prerequisites

- **Jetson Thor** flashed with **JetPack 7.0** via [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager) (with CUDA, CUDA-X, TensorRT, and NVIDIA Container Runtime components installed)
- [NGC CLI](https://org.ngc.nvidia.com/setup/installers/cli) installed and configured
- [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) and [Docker Compose](https://docs.docker.com/compose/install/linux/)
- [HuggingFace API token](https://huggingface.co/docs/hub/en/security-tokens) for downloading LLM models
- Network connectivity

## Project Structure

```
examples/voice_agent_webrtc/
├── docker-compose.jetson.yml   # Jetson-specific deployment
└── env.jetson.example          # Template for .env.jetson
```
> **Note:** This deployment uses vLLM for LLM inference instead of NVIDIA NIM. NIMs use TensorRT-LLM which provides optimized, pre-compiled inference engines for specific GPU architectures. Since Jetson Thor NIMs are not yet available, vLLM serves as a flexible alternative that can load HuggingFace models directly. Once Jetson Thor NIMs are released, they can be swapped in for improved inference performance.

## Step 1: Clone Project

On your Jetson Thor device:

```bash
git clone https://github.com/NVIDIA/voice-agent-examples.git
cd voice-agent-examples
```

## Step 2: Navigate to Example Folder

```bash
cd examples/voice_agent_webrtc
```

## Step 3: Configure Environment Variables

```bash
cp env.jetson.example .env.jetson
nano .env.jetson
```

## Step 4: Deploy Nemotron Speech ASR and TTS models


### Prerequisites

Ensure you meet the prerequisites before proceeding:
https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html#prerequisites

### Download and Deploy Nemotron Speech ASR and TTS models

Configure NGC CLI with your API key:

```bash
ngc config set
```

Download and Deploy ASR and TTS models using Quick Start scripts:

```bash
ngc registry resource download-version nvidia/riva/riva_quickstart_arm64:2.24.0
cd riva_quickstart_arm64_v2.24.0
bash riva_init.sh
bash riva_start.sh
```

> **Note:** Initialization may take 30-60 minutes on first run.

## Step 5: Start LLM Service and Voice Agent Application

```bash
cd /home/nvidia/voice-agent-examples/examples/voice_agent_webrtc

sudo docker compose -f docker-compose.jetson.yml up -d
```

## Step 6: Access the Application

Open in browser: `http://<jetson-ip>:8081`

## Switching LLM Models

Available models:

| NVIDIA_LLM_MODEL |
|------------------|
| `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16` |
| `nvidia/Nemotron-Mini-4B-Instruct` |
| `nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4` |

To switch:

```bash
# Update NVIDIA_LLM_MODEL in .env.jetson
nano .env.jetson

# Restart all services (no rebuild needed for model changes)
sudo docker compose -f docker-compose.jetson.yml down
sudo docker compose -f docker-compose.jetson.yml up -d

# Check LLM logs to verify new model is loading
sudo docker compose -f docker-compose.jetson.yml logs -f llm-nvidia-jetson
```

## Common Commands

```bash
# View logs
sudo docker compose -f docker-compose.jetson.yml logs -f python-app

# Stop all services
sudo docker compose -f docker-compose.jetson.yml down

# Rebuild after code changes
sudo docker compose -f docker-compose.jetson.yml up --build -d python-app
```
