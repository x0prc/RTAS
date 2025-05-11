import numpy as np
from audio_io import load_audio, save_audio

def extract_frequency(stego, cover, alpha):
    """Extract secret audio from stego audio using FFT."""
    stego_fft = np.fft.fft(stego)
    cover_fft = np.fft.fft(cover)
    secret_fft = (stego_fft - cover_fft) / alpha
    secret_audio = np.fft.ifft(secret_fft).real
    return secret_audio / np.max(np.abs(secret_audio))

def main():
    import yaml, os
    config_path = os.path.join(os.path.dirname(__file__), "../../config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f)

    stego_audio, sr = load_audio(config["audio"]["stego_audio_path"])
    cover_audio, _ = load_audio(config["audio"]["cover_audio_path"])

    secret_audio = extract_frequency(stego_audio, cover_audio, config["audio"]["alpha"])
    save_audio(config["audio"]["extracted_audio_path"], secret_audio, sr)
    print(f"Extracted audio saved to {config['audio']['extracted_audio_path']}")

if __name__ == "__main__":
    main()
