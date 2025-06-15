# Real-Time Audio Steganography with GPU Acceleration

A system for hiding secret audio messages within another audio stream in real-time, using steganographic techniques accelerated on the GPU.

## 🔍 Core Features

- **Real-time processing**: <10ms embedding/extraction time per chunk
- **GPU acceleration**: Uses PyAudio and TorchAudio for parallel audio processing
- **Multiple steganography techniques**:
  - LSB (Least Significant Bit) embedding in time domain
  - Frequency domain embedding using FFT
  - Echo hiding (optional)
- **High-quality output**: Achieves >30dB SNR for minimal audio degradation
- **Live audio support**: Works with microphone input and audio playback devices
- **Cross-platform**: Works on major operating systems with appropriate GPU support


![diagram](https://github.com/user-attachments/assets/a623f668-1169-4eee-8029-05522780d636)

## 🛠️ Installation

### Prerequisites

- Python 3.7+ or C++ compiler
- NVIDIA GPU with CUDA support (or OpenCL compatible GPU)
- Audio development libraries (PortAudio, etc.)

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/x0prc/RTAS.git
   cd RTAS
   ```

2. Create a virtual environment (Python version):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Performance

| Metric | Target | Typical Results |
|--------|--------|----------------|
| Embedding Time per Chunk | < 10 ms | 3-8 ms |
| Audio Latency (Total) | < 50 ms | 30-45 ms |
| Stego Quality (SNR) | > 30 dB | 32-40 dB |
| Payload Capacity | 4-16 kbps | ~10 kbps |


## 🔒 Security Considerations

While this system can hide audio data in real-time, it's important to note:

- The steganography techniques used here prioritize real-time performance over absolute security
- For highly sensitive data, consider adding encryption before embedding
- The system may be vulnerable to dedicated steganalysis techniques
- Higher embedding strengths improve security but reduce audio quality


## 📚 References and Inspiration

- Audio signal processing techniques from MIT OpenCourseWare
- Torchaudio Documentation for data processing
- Papers on audio steganography with spectral modifications


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
