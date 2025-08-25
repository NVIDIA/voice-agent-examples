class AudioProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        this.sampleRate = options.processorOptions.sampleRate || 16000;
        this.numChannels = options.processorOptions.numChannels || 1;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        
        if (input && input.length > 0 && input[0]) {
            const audioData = input[0]; // Get the first channel
            
            // Convert Float32Array to Int16Array (PCM S16)
            const pcmS16Array = new Int16Array(audioData.length);
            for (let i = 0; i < audioData.length; i++) {
                const sample = Math.max(-1, Math.min(1, audioData[i]));
                pcmS16Array[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
            }
            
            // Send the processed audio data to the main thread
            this.port.postMessage({
                type: 'audioData',
                data: pcmS16Array.buffer,
                sampleRate: this.sampleRate,
                numChannels: this.numChannels
            });
        }
        
        return true; // Keep the processor alive
    }
}

registerProcessor('audio-processor', AudioProcessor); 