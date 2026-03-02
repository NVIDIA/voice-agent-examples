# Voice Agent Examples

NVIDIA Pipecat provides a flexible framework for building real-time voice AI applications. These examples demonstrate various implementation patterns, ranging from simple LLM-based conversations to complex agentic workflows, and from WebSocket-based solutions to advanced WebRTC implementations with real-time capabilities. All examples leverage NVIDIA services including [Nemotron Speech ASR Parakeet](https://build.nvidia.com/nvidia/parakeet-ctc-1_1b-asr), [Nemotron Speech TTS Magpie](https://build.nvidia.com/nvidia/magpie-tts-multilingual), [LLM NIMs](https://build.nvidia.com/models), [NAT (NeMo Agent Toolkit)](https://github.com/NVIDIA/NeMo-Agent-Toolkit), and [Foundational RAG](https://github.com/NVIDIA-AI-Blueprints/rag).

Each example includes detailed setup instructions, configuration options, and deployment guides. We recommend starting with the Voice Agent WebSocket example for a simple introduction, then progressing to WebRTC-based examples for production use cases.

## Voice Agent WebSocket

A straightforward voice agent pipeline built on Pipecat's FastAPI WebSocket transport, ideal for getting started with voice AI applications.

**Key Features:**
- Simple WebSocket-based communication
- Integration with Nemotron Speech ASR and TTS models
- NVIDIA LLM Service support
- Flexible deployment via Docker or Python
- Quick setup and easy configuration

[View example →](./voice_agent_websocket/README.md)

## Voice Agent WebRTC

A production-grade, real-time voice assistant with live transcript capabilities using WebRTC for low-latency communication.

**Key Features:**
- WebRTC-based SmallWebRTCTransport for real-time streaming
- FastAPI backend with React frontend
- Live transcript display in the UI
- Dynamic prompt configuration and TTS voice selection via UI
- Nemotron Speech ASR and TTS integration
- NVIDIA LLM Service support
- Coturn server support for cloud deployments
- Flexible deployment via Docker or Python
- Support for multilingual ASR and TTS models
- Jetson deployment support with optimized configurations

[View example →](./voice_agent_webrtc/README.md)

## NAT Agent (NeMo Agent Toolkit)

An end-to-end intelligent voice assistant powered by NeMo Agent Toolkit, demonstrating how to build production-ready agentic voice applications with custom function calling, comprehensive observability, and modular architecture.

**Key Features:**
- [ReWOO agent](https://arxiv.org/abs/2305.18323) that implements Reasoning Without Observation, separating planning, execution, and solving into distinct phases
- Interactive flowershop assistant (menu browsing, pricing, cart management)
- Custom function registration via NeMo Agent Toolkit
- RESTful API deployment for NAT Agent using `nat serve`
- Phoenix tracing for comprehensive observability
- Built-in workflow profiling and evaluation tools
- Integration with [WebRTC UI](./webrtc_ui/README.md) frontend
- Modular architecture separating agent logic from pipeline components

[View example →](./nat_agent/README.md)

## Ambient Healthcare Agent for Patients

An agentic healthcare front desk can assist patients and healthcare professional staff by reducing the burden of the patient intake process, structuring responses into documentation and thus allowing for more patient-clinical staff quality time.

**Key Features:**
- Agentic AI for intelligent patient interactions
- NVIDIA Nemo Guardrails for safety to agent's interactions
- Built on [WebRTC Voice Agent](./voice_agent_webrtc/README.md) foundation
- Nemotron Speech ASR and TTS with speculative speech processing
- Comprehensive patient information collection workflows

Source code can be found at [NVIDIA-AI-Blueprints/ambient-patient](https://github.com/NVIDIA-AI-Blueprints/ambient-patient) GitHub Repository. Same repository includes an appointment making agent, medication information agent, and a full agent that combines the 3 specialized agents.
