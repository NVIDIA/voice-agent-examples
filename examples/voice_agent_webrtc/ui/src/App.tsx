// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD 2-Clause License

import { toast } from "sonner";
import { useEffect, useState } from "react";
import { AudioStream } from "./AudioStream";
import { AudioWaveForm } from "./AudioWaveForm";
import { Toaster } from "./components/ui/sonner";
import { RTC_CONFIG, RTC_OFFER_URL, DYNAMIC_PROMPT, POLL_PROMPT_URL } from "./config";
import usePipecatWebRTC from "./hooks/use-pipecat-webrtc";
import { Transcripts } from "./Transcripts";
import WebRTCButton from "./WebRTCButton";
import MicrophoneButton from "./MicrophoneButton";
import { PromptInput } from "./PromptInput";

function App() {
  const [showPromptInput, setShowPromptInput] = useState<boolean>(false); // Control PromptInput visibility
  const [currentPrompt, setCurrentPrompt] = useState<string>(""); // Store current prompt value
  
  const webRTC = usePipecatWebRTC({
    url: RTC_OFFER_URL,
    rtcConfig: RTC_CONFIG,
    onError: (e) => toast.error(e.message),
  });

  // Fetch and set the latest prompt when page loads - only if DYNAMIC_PROMPT is true
  useEffect(() => {
    if (DYNAMIC_PROMPT) {
      const fetchPrompt = async () => {
        try {
          console.log("Fetching latest prompt from API... (DYNAMIC_PROMPT mode)");
          const response = await fetch(POLL_PROMPT_URL);
          
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          
          const data = await response.json();
          console.log("Latest Prompt:", data);
          // Set the fetched prompt as current value
          setCurrentPrompt(data.prompt); // Initialize currentPrompt with API data
          console.log("Current prompt updated in PromptInput component");
        } catch (error) {
          console.error("Error fetching prompt:", error);
          toast.error("Failed to fetch latest prompt");
          // Keep the fallback default value on error
        }
      };

      fetchPrompt();
    } else {
      console.log("DYNAMIC_PROMPT is false - skipping API call");
    }
  }, []); // Empty dependency array - runs only on component mount (page reload)

  // Send current prompt IMMEDIATELY when WebRTC connection is established
  useEffect(() => {
    if (webRTC.status === "connected" && currentPrompt.trim()) {
      console.log("WebRTC connected! Sending prompt IMMEDIATELY:", currentPrompt);
      // Send without any delay to beat the LLM initialization
      webRTC.websocket.send(JSON.stringify({
        type: "context_reset",
        message: currentPrompt.trim(),
      }));
    }
  }, [webRTC.status]); // Triggers immediately when status becomes "connected"

  return (
    <div className="h-screen flex flex-col">
      <header className="border-b-1 border-gray-200 p-6 flex items-center">
        <img src="logo.png" alt="NVIDIA ACE Logo" className="h-16 mr-8" />
        <h1 className="text-2xl">Speech to Speech Demo</h1>
      </header>
      <section className="flex-1 flex">
        <div className="flex-1 p-5">
          <AudioStream
            streamOrTrack={webRTC.status === "connected" ? webRTC.stream : null}
          />
          <Transcripts
            websocket={webRTC.status === "connected" ? webRTC.websocket : null}
          />
        </div>
        <div className="p-5 border-l-1 border-gray-200 flex flex-col">
          <div className="flex-1 mb-4">
            <AudioWaveForm
              streamOrTrack={webRTC.status === "connected" ? webRTC.stream : null}
            />
          </div>
          {showPromptInput && (
            <div className="flex-7">
              <PromptInput
                defaultValue={currentPrompt}
                onChange={(prompt) => setCurrentPrompt(prompt)}
                disabled={webRTC.status === "connected"}
              />
            </div>
          )}
        </div>
      </section>
      <footer className="border-t-1 border-gray-200 p-6 flex items-center justify-between">
        <div className="flex items-center">
          <WebRTCButton {...webRTC} />
          {webRTC.status === "connected" && (
            <MicrophoneButton stream={webRTC.micStream} />
          )}
        </div>
        {DYNAMIC_PROMPT && (
          <button
            type="button"
            className="bg-nvidia px-4 py-2 rounded-lg text-white"
            onClick={() => {
              setShowPromptInput(!showPromptInput);
            }}
          >
            {showPromptInput ? "Hide Prompt" : "Show Prompt"}
          </button>
        )}
      </footer>
      <Toaster />
    </div>
  );
}

export default App;