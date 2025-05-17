# Real-Time Audio Steganography with GPU Acceleration

A system for hiding secret audio messages within another audio stream in real-time, using steganographic techniques accelerated on the GPU.

## 🔍 Core Features

- **Real-time processing**: <10ms embedding/extraction time per chunk
- **GPU acceleration**: Uses PyAudio with CUDA for parallel audio processing
- **Multiple steganography techniques**: 
  - LSB (Least Significant Bit) embedding in time domain
  - Frequency domain embedding using FFT
  - Echo hiding (optional)
- **High-quality output**: Achieves >30dB SNR for minimal audio degradation
- **Live audio support**: Works with microphone input and audio playback devices
- **Cross-platform**: Works on major operating systems with appropriate GPU support


## ⚙️ Technologies Used

| Layer | Tools |
|-------|-------|
| Audio I/O | PortAudio, PyAudio, ALSA |
| Signal Processing | FFT/Inverse FFT (cuFFT / custom CUDA FFT) |
| GPU Acceleration | CUDA |
| Optional GUI | PyQt, Dear ImGui |
| Dev Language | C++ with CUDA|

## 🛠️ Installation

### Prerequisites

- Python 3.7+ or C++ compiler with CUDA support
- NVIDIA GPU with CUDA support (or OpenCL compatible GPU)
- CUDA Toolkit 11.0+ or OpenCL SDK
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

4. Install GPU dependencies:
   - For CUDA: `pip install cupy-cuda11x pycuda`  
     (Replace `11x` with your CUDA version)
   - For OpenCL: `pip install pyopencl`

5. Compile CUDA kernels:
   ```bash
   cd src/gpu
   make
   ```

## 🚀 Usage

### Basic Usage

```bash
# Encode a secret audio file into a cover file
python src/main.py encode --cover path/to/cover.wav --secret path/to/secret.wav --output path/to/output.wav --method fft

# Decode a stego audio file to extract the secret
python src/main.py decode --stego path/to/stego.wav --output path/to/extracted.wav --method fft

# Real-time encoding with microphone input
python src/main.py realtime --method fft --mode encode

# Real-time decoding with microphone input
python src/main.py realtime --method fft --mode decode
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
- NVIDIA cuFFT library: [cuFFT Documentation](https://docs.nvidia.com/cuda/cufft/index.html)
- Papers on audio steganography with spectral modifications


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
