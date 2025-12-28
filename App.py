#!/usr/bin/env python3
"""Simple Whisper Web UI - uses openai-whisper with ROCm/PyTorch"""

import tempfile
import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import whisper
import torch

app = FastAPI()

# Load model on startup (uses ROCm GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Whisper model on {device}...")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
model = whisper.load_model("medium", device=device)  # Change to "small", "medium", "large" as needed
print("Model loaded!")

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WHISPER//TRANSCRIBE</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap" rel="stylesheet">
    <style>
        /* ============================================
           CSS Variables & Reset
           ============================================ */
        :root {
            --neon-cyan: #00ffff;
            --neon-magenta: #ff00ff;
            --neon-green: #39ff14;
            --neon-pink: #ff0080;
            --bg-deep: #0a0a0f;
            --bg-primary: #0d0d1a;
            --bg-panel: rgba(15, 15, 30, 0.85);
            --text-primary: #e0e0e0;
            --text-dim: #666;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        /* ============================================
           Base Styles & Typography
           ============================================ */
        body {
            font-family: 'Share Tech Mono', 'Courier New', monospace;
            background: var(--bg-deep);
            background-image:
                radial-gradient(ellipse at 50% 0%, rgba(0, 255, 255, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(255, 0, 255, 0.05) 0%, transparent 40%);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* ============================================
           CRT Effects
           ============================================ */
        .scanlines {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 1000;
            background: repeating-linear-gradient(
                0deg,
                rgba(0, 0, 0, 0.1),
                rgba(0, 0, 0, 0.1) 1px,
                transparent 1px,
                transparent 2px
            );
        }

        .vignette {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 999;
            background: radial-gradient(
                ellipse at center,
                transparent 0%,
                transparent 50%,
                rgba(0, 0, 0, 0.5) 100%
            );
        }

        /* ============================================
           Layout Container
           ============================================ */
        .container {
            max-width: 650px;
            margin: 0 auto;
            padding: 30px 20px;
            position: relative;
            z-index: 1;
        }

        /* ============================================
           Glass Panel Component
           ============================================ */
        .glass-panel {
            background: var(--bg-panel);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 255, 255, 0.2);
            border-radius: 12px;
            box-shadow:
                0 8px 32px rgba(0, 0, 0, 0.4),
                inset 0 0 30px rgba(0, 255, 255, 0.03),
                0 0 15px rgba(0, 255, 255, 0.1);
            margin-bottom: 20px;
            overflow: hidden;
        }

        /* ============================================
           Header Section
           ============================================ */
        .header-panel {
            padding: 25px 30px;
            text-align: center;
            border-bottom: 1px solid rgba(0, 255, 255, 0.1);
        }

        .title {
            font-family: 'Orbitron', sans-serif;
            font-size: 2rem;
            font-weight: 900;
            letter-spacing: 0.15em;
            color: #fff;
            text-shadow:
                0 0 5px #fff,
                0 0 10px #fff,
                0 0 20px var(--neon-cyan),
                0 0 40px var(--neon-cyan),
                0 0 80px var(--neon-cyan);
            margin-bottom: 8px;
        }

        .subtitle {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.85rem;
            color: var(--neon-cyan);
            letter-spacing: 0.2em;
            opacity: 0.8;
        }

        /* ============================================
           Visualizer Section
           ============================================ */
        .visualizer-panel {
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 180px;
            position: relative;
            transition: all 0.3s ease;
        }

        .visualizer-panel.active {
            border-color: rgba(255, 0, 255, 0.4);
            box-shadow:
                0 8px 32px rgba(0, 0, 0, 0.4),
                inset 0 0 40px rgba(255, 0, 255, 0.05),
                0 0 25px rgba(255, 0, 255, 0.2);
        }

        #waveformCanvas {
            width: 100%;
            height: 140px;
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.3);
        }

        .idle-text {
            position: absolute;
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.9rem;
            color: var(--text-dim);
            letter-spacing: 0.1em;
        }

        /* ============================================
           Control Section
           ============================================ */
        .control-section {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 25px;
            gap: 15px;
        }

        .cyber-button {
            position: relative;
            padding: 18px 50px;
            font-family: 'Orbitron', sans-serif;
            font-size: 14px;
            font-weight: 700;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            color: #fff;
            background: linear-gradient(135deg, rgba(0, 255, 255, 0.1) 0%, rgba(0, 100, 100, 0.2) 100%);
            border: 2px solid var(--neon-cyan);
            border-radius: 4px;
            cursor: pointer;
            overflow: hidden;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow:
                0 0 15px rgba(0, 255, 255, 0.3),
                0 0 30px rgba(0, 255, 255, 0.15),
                inset 0 0 20px rgba(0, 255, 255, 0.1);
        }

        .cyber-button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(0, 255, 255, 0.4), transparent);
            transition: left 0.6s ease;
        }

        .cyber-button:hover::before {
            left: 100%;
        }

        .cyber-button:hover {
            transform: translateY(-2px);
            box-shadow:
                0 0 25px rgba(0, 255, 255, 0.5),
                0 0 50px rgba(0, 255, 255, 0.25),
                0 0 75px rgba(0, 255, 255, 0.15),
                inset 0 0 30px rgba(0, 255, 255, 0.15);
        }

        .cyber-button.recording {
            border-color: var(--neon-magenta);
            background: linear-gradient(135deg, rgba(255, 0, 128, 0.15) 0%, rgba(100, 0, 80, 0.25) 100%);
            animation: recordPulse 1.2s ease-in-out infinite;
            box-shadow:
                0 0 20px rgba(255, 0, 128, 0.5),
                0 0 40px rgba(255, 0, 128, 0.3),
                0 0 60px rgba(255, 0, 128, 0.15),
                inset 0 0 25px rgba(255, 0, 128, 0.15);
        }

        @keyframes recordPulse {
            0%, 100% {
                box-shadow:
                    0 0 20px rgba(255, 0, 128, 0.5),
                    0 0 40px rgba(255, 0, 128, 0.3),
                    inset 0 0 25px rgba(255, 0, 128, 0.15);
            }
            50% {
                box-shadow:
                    0 0 35px rgba(255, 0, 128, 0.7),
                    0 0 70px rgba(255, 0, 128, 0.4),
                    0 0 100px rgba(255, 0, 128, 0.2),
                    inset 0 0 35px rgba(255, 0, 128, 0.2);
            }
        }

        .status-text {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.85rem;
            color: var(--text-dim);
            letter-spacing: 0.15em;
        }

        .status-text.processing {
            color: var(--neon-cyan);
        }

        .spinner {
            display: inline-block;
            width: 14px;
            height: 14px;
            border: 2px solid rgba(0, 255, 255, 0.3);
            border-top-color: var(--neon-cyan);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 8px;
            vertical-align: middle;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        /* ============================================
           Result Section
           ============================================ */
        .result-panel {
            padding: 0;
        }

        .panel-header {
            padding: 12px 20px;
            background: rgba(0, 0, 0, 0.3);
            border-bottom: 1px solid rgba(0, 255, 255, 0.1);
        }

        .terminal-prompt {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.8rem;
            color: var(--neon-green);
            letter-spacing: 0.1em;
        }

        .result-content {
            padding: 20px;
            min-height: 100px;
            font-size: 1rem;
            line-height: 1.6;
            color: var(--text-primary);
            white-space: pre-wrap;
            word-break: break-word;
        }

        .cursor {
            display: inline-block;
            width: 10px;
            height: 1.2em;
            background: var(--neon-cyan);
            margin-left: 2px;
            animation: blink 1s step-end infinite;
            vertical-align: text-bottom;
        }

        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }

        .result-content.has-text .cursor {
            display: none;
        }

        /* ============================================
           Copy Button
           ============================================ */
        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .copy-btn {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.7rem;
            color: var(--neon-cyan);
            background: transparent;
            border: 1px solid rgba(0, 255, 255, 0.3);
            border-radius: 3px;
            padding: 4px 10px;
            cursor: pointer;
            letter-spacing: 0.1em;
            transition: all 0.2s ease;
            opacity: 0.7;
        }

        .copy-btn:hover {
            opacity: 1;
            border-color: var(--neon-cyan);
            box-shadow: 0 0 10px rgba(0, 255, 255, 0.3);
        }

        .copy-btn.copied {
            color: var(--neon-green);
            border-color: var(--neon-green);
            box-shadow: 0 0 10px rgba(57, 255, 20, 0.3);
        }

        .copy-btn.attention {
            animation: copyPulse 0.6s ease-in-out 3;
            opacity: 1;
            border-color: var(--neon-cyan);
        }

        @keyframes copyPulse {
            0%, 100% {
                box-shadow: 0 0 5px rgba(0, 255, 255, 0.3);
                transform: scale(1);
            }
            50% {
                box-shadow: 0 0 20px rgba(0, 255, 255, 0.8), 0 0 30px rgba(0, 255, 255, 0.4);
                transform: scale(1.1);
            }
        }

        /* ============================================
           Responsive Design
           ============================================ */
        @media (max-width: 600px) {
            .container {
                padding: 20px 15px;
            }

            .title {
                font-size: 1.4rem;
                letter-spacing: 0.1em;
            }

            .subtitle {
                font-size: 0.75rem;
            }

            .cyber-button {
                padding: 15px 35px;
                font-size: 12px;
            }

            #waveformCanvas {
                height: 100px;
            }
        }
    </style>
</head>
<body>
    <!-- CRT Effects -->
    <div class="scanlines"></div>
    <div class="vignette"></div>

    <div class="container">
        <!-- Header -->
        <div class="glass-panel header-panel">
            <h1 class="title">WHISPER//TRANSCRIBE</h1>
            <div class="subtitle">[ NEURAL VOICE DECODER v2.0 ]</div>
        </div>

        <!-- Visualizer -->
        <div class="glass-panel visualizer-panel" id="visualizerPanel">
            <canvas id="waveformCanvas"></canvas>
            <span class="idle-text" id="idleText">AWAITING INPUT...</span>
        </div>

        <!-- Controls -->
        <div class="glass-panel control-section">
            <button id="recordBtn" class="cyber-button">HOLD TO RECORD</button>
            <div id="status" class="status-text">[ SYSTEM READY ]</div>
        </div>

        <!-- Result -->
        <div class="glass-panel result-panel">
            <div class="panel-header">
                <span class="terminal-prompt">&gt; OUTPUT_</span>
                <button id="copyBtn" class="copy-btn">COPY</button>
            </div>
            <div id="result" class="result-content"><span class="cursor"></span></div>
        </div>
    </div>

    <script>
        /* ============================================
           DOM References
           ============================================ */
        const recordBtn = document.getElementById('recordBtn');
        const status = document.getElementById('status');
        const result = document.getElementById('result');
        const canvas = document.getElementById('waveformCanvas');
        const visualizerPanel = document.getElementById('visualizerPanel');
        const idleText = document.getElementById('idleText');
        const copyBtn = document.getElementById('copyBtn');
        const ctx = canvas.getContext('2d');

        /* ============================================
           Audio Context & Analyzer
           ============================================ */
        let audioContext;
        let analyser;
        let dataArray;
        let animationId;
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;

        // Set canvas size
        function resizeCanvas() {
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * window.devicePixelRatio;
            canvas.height = rect.height * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        }
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        // Draw idle state
        function drawIdle() {
            const width = canvas.width / window.devicePixelRatio;
            const height = canvas.height / window.devicePixelRatio;

            ctx.fillStyle = 'rgba(10, 10, 15, 0.1)';
            ctx.fillRect(0, 0, width, height);

            // Draw subtle center line
            ctx.strokeStyle = 'rgba(0, 255, 255, 0.15)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, height / 2);
            ctx.lineTo(width, height / 2);
            ctx.stroke();
        }
        drawIdle();

        /* ============================================
           Waveform Visualization
           ============================================ */
        function drawWaveform() {
            if (!isRecording) return;

            const width = canvas.width / window.devicePixelRatio;
            const height = canvas.height / window.devicePixelRatio;

            analyser.getByteFrequencyData(dataArray);

            // Fade effect for trails
            ctx.fillStyle = 'rgba(10, 10, 15, 0.25)';
            ctx.fillRect(0, 0, width, height);

            const barCount = 64;
            const gap = 3;
            const totalGap = gap * (barCount - 1);
            const barWidth = (width - totalGap) / barCount;
            const centerY = height / 2;
            const maxBarHeight = (height / 2) - 10;

            for (let i = 0; i < barCount; i++) {
                // Map to frequency data (focus on lower/mid frequencies)
                const dataIndex = Math.floor(i * (dataArray.length * 0.6) / barCount);
                const value = dataArray[dataIndex] / 255;
                const barHeight = Math.max(2, value * maxBarHeight);

                // Create gradient
                const gradient = ctx.createLinearGradient(0, centerY - barHeight, 0, centerY + barHeight);
                gradient.addColorStop(0, '#00ffff');
                gradient.addColorStop(0.3, '#00ccff');
                gradient.addColorStop(0.5, '#ff00ff');
                gradient.addColorStop(0.7, '#00ccff');
                gradient.addColorStop(1, '#00ffff');

                ctx.fillStyle = gradient;
                ctx.shadowBlur = 10;
                ctx.shadowColor = value > 0.5 ? '#ff00ff' : '#00ffff';

                const x = i * (barWidth + gap);

                // Draw mirrored bars
                ctx.fillRect(x, centerY - barHeight, barWidth, barHeight);
                ctx.fillRect(x, centerY, barWidth, barHeight);
            }

            ctx.shadowBlur = 0;
            animationId = requestAnimationFrame(drawWaveform);
        }

        /* ============================================
           Copy to Clipboard
           ============================================ */
        async function copyToClipboard(text) {
            if (!text) return false;
            try {
                await navigator.clipboard.writeText(text);
                copyBtn.textContent = 'COPIED';
                copyBtn.classList.add('copied');
                setTimeout(() => {
                    copyBtn.textContent = 'COPY';
                    copyBtn.classList.remove('copied');
                }, 2000);
                return true;
            } catch (err) {
                console.error('Copy failed:', err);
                return false;
            }
        }

        copyBtn.addEventListener('click', () => {
            const text = result.textContent;
            if (text && !result.querySelector('.cursor')) {
                copyToClipboard(text);
            }
        });

        function highlightForCopy() {
            // Auto-select the result text
            const selection = window.getSelection();
            const range = document.createRange();
            range.selectNodeContents(result);
            selection.removeAllRanges();
            selection.addRange(range);

            // Pulse the copy button
            copyBtn.classList.remove('attention');
            void copyBtn.offsetWidth; // Trigger reflow to restart animation
            copyBtn.classList.add('attention');
            setTimeout(() => copyBtn.classList.remove('attention'), 2000);
        }

        /* ============================================
           Recording Logic
           ============================================ */
        recordBtn.addEventListener('mousedown', startRecording);
        recordBtn.addEventListener('mouseup', stopRecording);
        recordBtn.addEventListener('mouseleave', stopRecording);
        recordBtn.addEventListener('touchstart', (e) => { e.preventDefault(); startRecording(); });
        recordBtn.addEventListener('touchend', stopRecording);

        async function startRecording() {
            if (isRecording) return;

            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

                // Setup audio analyzer
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioContext.createAnalyser();
                analyser.fftSize = 256;
                analyser.smoothingTimeConstant = 0.8;
                dataArray = new Uint8Array(analyser.frequencyBinCount);

                const source = audioContext.createMediaStreamSource(stream);
                source.connect(analyser);

                // Setup recorder
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
                mediaRecorder.onstop = sendAudio;

                mediaRecorder.start();
                isRecording = true;

                // Update UI
                recordBtn.classList.add('recording');
                recordBtn.textContent = 'RECORDING...';
                status.textContent = '[ CAPTURING AUDIO ]';
                visualizerPanel.classList.add('active');
                idleText.style.display = 'none';

                // Start visualization
                drawWaveform();

            } catch (err) {
                status.textContent = '[ ERROR: ' + err.message + ' ]';
            }
        }

        function stopRecording() {
            if (!isRecording) return;
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                isRecording = false;
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(t => t.stop());

                if (audioContext) {
                    audioContext.close();
                }

                if (animationId) {
                    cancelAnimationFrame(animationId);
                }

                // Update UI
                recordBtn.classList.remove('recording');
                recordBtn.textContent = 'HOLD TO RECORD';
                visualizerPanel.classList.remove('active');

                // Clear canvas
                setTimeout(() => {
                    const width = canvas.width / window.devicePixelRatio;
                    const height = canvas.height / window.devicePixelRatio;
                    ctx.clearRect(0, 0, width, height);
                    drawIdle();
                    idleText.style.display = 'block';
                }, 100);
            }
        }

        /* ============================================
           API Communication
           ============================================ */
        async function sendAudio() {
            status.innerHTML = '<span class="spinner"></span>PROCESSING...';
            status.classList.add('processing');
            idleText.textContent = 'DECODING NEURAL PATTERNS...';

            const blob = new Blob(audioChunks, { type: 'audio/webm' });
            const formData = new FormData();
            formData.append('file', blob, 'recording.webm');

            try {
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                // Update result with typing effect simulation
                const text = data.text || data.error || 'No transcription available';
                result.innerHTML = text;
                result.classList.add('has-text');

                // Highlight text and copy button for easy copying
                if (data.text) {
                    highlightForCopy();
                }

                status.classList.remove('processing');
                status.textContent = `[ COMPLETE: ${data.duration || '?'}s audio / ${data.processing_time || '?'}s processing ]`;
                idleText.textContent = 'AWAITING INPUT...';

            } catch (err) {
                result.innerHTML = 'ERROR: ' + err.message;
                result.classList.add('has-text');
                status.classList.remove('processing');
                status.textContent = '[ TRANSMISSION FAILED ]';
                idleText.textContent = 'AWAITING INPUT...';
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    import time
    start = time.time()
    
    # Save uploaded audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Transcribe with whisper
        result = model.transcribe(tmp_path)
        processing_time = round(time.time() - start, 2)
        
        return {
            "text": result["text"].strip(),
            "language": result.get("language"),
            "duration": round(result.get("segments", [{}])[-1].get("end", 0), 1) if result.get("segments") else None,
            "processing_time": processing_time
        }
    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
