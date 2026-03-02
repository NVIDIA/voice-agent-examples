# Nemotron Voice Agent Examples

This repository contains examples demonstrating how to build voice-enabled conversational AI agents using the NVIDIA services, built using [the Pipecat framework](https://github.com/pipecat-ai/pipecat). These examples demonstrate various implementation patterns, ranging from simple LLM-based conversations to complex agentic workflows, and from WebSocket-based solutions to advanced WebRTC implementations with real-time capabilities.

## Examples Overview

-  **[Voice Agent WebSocket](examples/voice_agent_websocket/)** : A simple voice assistant pipeline using WebSocket-based transport. This example demonstrates integration with NVIDIA LLM Service, Nemotron Speech ASR and TTS NIMS. 
- **[Voice Agent WebRTC](examples/voice_agent_webrtc/)** : A more advanced voice agent using WebRTC Transport with real-time transcripts, dynamic prompt configuration and TTS voice selection via UI.
- **[NAT Agent (NeMo Agent Toolkit)](examples/nat_agent/)** : An end-to-end intelligent voice assistant powered by NeMo Agent Toolkit. The ReWoo agent uses planning-based approach for efficient task decomposition and execution with custom tools for menu browsing, pricing and cart management.

We recommend starting with the Voice Agent WebSocket example for a simple introduction, then progressing to WebRTC-based examples for production use cases. More details on examples can be found in [examples README.md](examples/README.md).

## NVIDIA Pipecat

The NVIDIA Pipecat library augments [the Pipecat framework](https://github.com/pipecat-ai/pipecat) by adding additional frame processors and NVIDIA services. This includes the integration of NVIDIA services and NIMs such as [Nemotron Speech ASR Parakeet](https://build.nvidia.com/nvidia/parakeet-ctc-1_1b-asr), [Nemotron Speech TTS Magpie](https://build.nvidia.com/nvidia/magpie-tts-multilingual), [LLM NIMs](https://build.nvidia.com/models), [NAT (NeMo Agent Toolkit)](https://github.com/NVIDIA/NeMo-Agent-Toolkit), and [Foundational RAG](https://github.com/NVIDIA-AI-Blueprints/rag). It also introduces a few processors with a focus on improving the end-user experience for multimodal conversational agents, along with speculative speech processing to reduce latency for faster bot responses.


### Getting Started

The NVIDIA Pipecat package is released as a wheel on PyPI. Create a Python virtual environment and use the pip command to install the nvidia-pipecat package.

```bash
pip install nvidia-pipecat
```

You can start building pipecat pipelines utilizing services from the NVIDIA Pipecat package.

### Hacking on the framework itself

If you wish to work directly with the source code or modify services from the nvidia-pipecat package, you can utilize either the UV or Nix development setup as outlined below.

#### Using UV


To get started, first install the [UV package manager](https://docs.astral.sh/uv/#highlights). 

Then, create a virtual environment with all the required dependencies by running the following commands:
```bash
uv venv
source .venv/bin/activate
uv sync
```

Once the environment is set up, you can begin building pipelines or modifying the services in the source code.

If you wish to contribute your changes to the repository, please ensure you run the unit tests, linter, and formatting tool.

To run unit tests, use:
```
uv run pytest
```

To format the code, use:
```bash
ruff format
```

To run the linter, use:
```
ruff check
```


#### Using Nix

To set up your development environment using [the Nix](https://nixos.org/download/#nix-install-linux), follow these steps:

Initialize the development environment: Simply run the following command:
```bash
nix develop
```

This setup provides you with a fully configured environment, allowing you to focus on development without worrying about dependency management.

To ensure that all checks such as the formatting and linter for the repository are passing, use the following command:

```bash
nix flake check
```

## Agent Skills

This repository includes AI agent skills for deployment assistance. Install them for your coding agent with:

```bash
npx skills add .
```

## Documentation

The project documentation includes:

- **[Voice Agent Examples](./examples/README.md)** - Voice agents examples built using pipecat and NVIDIA services
- **[NVIDIA Pipecat](./docs/NVIDIA_PIPECAT.md)** - Custom Pipecat processors implemented for NVIDIA services
- **[Best Practices](./docs/BEST_PRACTICES.md)** - Performance optimization guidelines and production deployment strategies
- **[Speculative Speech Processing](./docs/SPECULATIVE_SPEECH_PROCESSING.md)** - Advanced speech processing techniques for reducing latency

## CONTRIBUTING

We invite contributions! Open a GitHub issue or pull request! See contributing guildelines [here](./CONTRIBUTING.md).

