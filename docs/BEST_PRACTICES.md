# Voice Agent Best Practices

Building production-grade voice agents requires careful consideration of multiple dimensions: technical performance, user experience, security, and operational excellence. This guide consolidates best practices and lessons learned from deploying voice agents at scale.

---
## Key Success Metrics

- **Latency**: Time from user speech end to bot response start (target: 600-1500ms)
- **Accuracy**: ASR word error rate (WER), factual correctness, LLM generation quality etc.
- **Scalability**: Concurrent streams supported without audio glitches or performance degradation, all models scale independently
- **Availability**: System uptime and fault tolerance (target: 99.9%+)
- **User Satisfaction**: Task completion rate and user feedback scores

---

## 1. Modular and Event-Driven Pipeline Design

Structure your voice agent as a composable pipeline of independent components:

```
Audio Input → VAD → ASR → Agent → TTS → Audio Output
```

Implement event-driven patterns for:
- Real-time transcription updates
- Intermediate processing results
- System health events
- User interaction events
- async/await patterns for non-blocking operations

**Benefits:**
- Easy to test and scale individual components
- Swap providers without full rewrites

---

## 2. Optimizing Pipeline Latency

For optimizing latency, first we need to measure e2e and component wise latency. Voice agent latency comes from multiple pipeline components. Understanding each contributor enables targeted optimization:

### 2.1 Audio Processing Latency

**Voice Activity Detection (VAD):**
- **Contribution**: 200-500ms (end of speech detection)
- **Optimization**: 
  - Use streaming VAD with shorter silence thresholds
  - Explore shorter EOU detection with Nemotron Speech ASR and open-source smart turn detection models
  - Implement adaptive VAD sensitivity based on environment noise

**Audio Buffering:**
- **Contribution**: 50-200ms (network buffering, codec processing)
- **Optimization**:
  - Use lower latency audio codecs (Opus at 20ms frames)
  - Minimize audio buffer sizes while maintaining quality
  - Implement jitter buffers for network variations
- **Scaling Audio Output for Concurrency:**  
  When scaling to multiple concurrent audio streams using either FastAPI WebSocket transport or WebRTC transport, consider increasing the output audio chunk size using the `audio_out_10ms_chunks` parameter up to 400ms to reduce audio glitches and enable smoother playback.

### 2.2 ASR (Automatic Speech Recognition) Latency

**Model Processing:**
- **Contribution**: 50-100 ms for Nemotron Speech ASR
- **Optimization**:
  - Prefer deploying Nemotron Speech ASR NIM locally
  - Utilize latest GPU hardware and optimized models
  - Maintain consistent latency performance when handling multiple concurrent requests
  - Use streaming ASR with interim results for early processing

### 2.3 Language Model (LLM) Processing Latency

**Model Inference:**
- **Contribution**: 200-800ms depending on model size and complexity
- **Optimization**:
  - **Model Selection**: Use smaller, faster models (8B vs 70B parameters)
  - **TRT LLM Optimized**: Use TRT LLM optimized NIM deployments
  - **Quantization**: INT8/FP16 models for 2-3x speedup
  - **KV-Cache Optimization**: Enable KV caching for lower TTFB and optimize based on use case

**Context Management:**
- **Contribution**: 50-200ms for large contexts
- **Optimization**:
  - Implement context truncation strategies
  - Enable KV caching with adequate cache size

### 2.4 TTS (Text-to-Speech) Latency

**Synthesis Time:**
- **Contribution**: 150-300ms for first audio chunk
- **Optimization**:
  - **Streaming TTS**: Start playback before full synthesis
  - **Local Nemotron Speech TTS**: 150-200ms with TRT optimized Magpie model
  - **Chunked Generation**: Process sentences as they're generated
  - **Batch Size**: Increasing the Magpie model batch size (e.g., to 64) can significantly boost throughput for high-volume or concurrent workloads.

**Audio Post-processing:**
- **Contribution**: 50-100ms (normalization, encoding)
- **Optimization**:
  - Minimize audio processing pipeline
  - Use hardware-accelerated audio codecs

### 2.5 Network and Infrastructure Latency

- **Geographic Distribution:** Distributed multi-node deployments based on user demographics
- **Load Balancing:** Use sticky sessions to avoid context switching
- **Monitoring:** Monitor key metrics in production deployment

### 2.6 Advanced Latency Reduction Techniques

**Speculative Speech Processing:**
- Process interim ASR transcripts before speech ends
- Pre-generate likely responses during user speech
- **Potential Savings**: 200-400ms reduction in perceived latency
- For more details, check [docs](SPECULATIVE_SPEECH_PROCESSING.md)

**Filler words or Intermediate responses:**
- Generate or use random filler words to reduce perceived latency
- For high latency agents or reasoning models, generate intermediate response based on function calls or thinking tokens

---

## 3. Designing User Experience

### 3.1 Conversation Design Principles

**Natural Turn-Taking:**
- Allow interruptions (barge-in)
- Implement proper silence handling
- Use conversational markers ("um", "let me check")

**Progressive Disclosure:**
```python
# Don't overwhelm with options
# Bad:
"You can check balance, transfer funds, pay bills, view history, 
update profile, set alerts, or lock your card. What would you like?"

# Good:
"What would you like to do today?"
# (Let user guide, offer suggestions if confused)
```
### 3.2 Persona & Tone Consistency

**Define Agent Personality:**
- Professional vs. casual
- Proactive vs. reactive
- Verbose vs. concise
- Empathetic vs. neutral

**Maintain Consistency:**
- Document persona guidelines
- Use system prompts for LLMs
- Implement tone checkers
- Regular quality reviews

### 3.3 Voice Selection

**Considerations:**
- Match voice to brand and use case
- Consider user demographics
- Regional accent preferences
- Gender neutrality options
- Custom IPA dictionary for mispronunciation

**Quality Metrics:**
- Naturalness (MOS score > 4.0)
- Prosody and intonation
- Emotional expressiveness
- Consistency across sessions

### 3.4 Response Optimization for Voice

**Voice-Specific Adaptations:**
- Keep responses concise (1-3 sentences per turn)
- Use conversational language (contractions, simple words)
- Structure information hierarchically
- Avoid lists with >3-4 items
- Use explicit transitions

### 3.5 Prompt Design

**System Prompt Instructions:**
- Include persona and tone guidelines directly in the system prompt for consistency
- Provide clear instructions to avoid outputting formatting (bullet points, markdown, URLs) that doesn't translate to voice
- Define conversation boundaries and scope to keep interactions focused and prevent rambling
- Include examples of ideal voice responses in the prompt for few-shot guidance
- Instructions for Progressive disclosure of options and Context-aware suggestions

### 3.6 ASR transcripts quality
- Implement custom vocabulary boosting for domain terms
- Use inverse text normalization (ITN) for proper formatting
- Make sure user audio quality is good
- Avoid resampling if possible
- Nemotron Speech ASR models are robust to noise, skip noise processing 
- Base critical decisions on final transcripts only
- Finetune ASR model on domain data if needed

### 3.7 User-Facing Error Handling

**Error Categories:**

```python
ERROR_MESSAGES = {
    "asr_failure": "I didn't catch that. Could you say that again?",
    "service_unavailable": "I'm having trouble connecting. Let me try again.",
    "timeout": "This is taking longer than expected. Please hold on.",
    "out_of_scope": "I'm not able to help with that, but I can help you with..."
}
```

**Recovery Strategies:**
- Offer alternative input methods (DTMF, transfer to human)
- Provide clear next steps
- Graceful conversation termination

### 3.8 Continuous testing
- Implement Unit and Integration Testing
- Load testing to find bottlenecks for latencies
- Prepare test data with different conversation scenarios
- A/B Testing to improve user experience
---

## 4. Scalability & Performance

### 4.1 Horizontal Scaling

**Stateless Services:**
- Deploy ASR/TTS behind load balancers
- Use container orchestration (Kubernetes)
- Auto-scaling based on CPU/memory/queue depth

**Stateful Services:**
- Use Sticky sessions
- Distributed session storage (Redis)

### 4.2 Resource Optimization

**Model Optimization:**
- Quantization (FP16, INT8) and TRT optimization for inference
- Smaller models selection for lower footprint
- Batch inference where possible
- GPU sharing and multiplexing

### 4.3 Network Optimization

**WebRTC Best Practices:**
- Use TURN servers for NAT traversal
- Implement adaptive bitrate
- Support multiple codecs (Opus preferred)
- Handle network transitions (WiFi to cellular)

---

## Conclusion

Building production voice agents requires a holistic approach balancing technical performance, user experience, and operational excellence. Key takeaways:

1. **Design for Latency**: Every millisecond counts in conversational AI
2. **Handle Errors Gracefully**: Users should never feel lost
3. **Monitor Everything**: You can't improve what you don't measure
4. **Test Thoroughly**: Automated testing catches issues before users do
5. **Iterate Based on Data**: Use real user feedback to improve
6. **Plan for Scale**: Design for 10x your current load
7. **Prioritize Security**: Protect user data as your top responsibility