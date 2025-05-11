import numpy as np
from audio_io import load_audio, save_audio

def embed_frequency(cover, secret, alpha):
    cover_fft = np.fft.fft(cover)
    secret_fft = np.fft.fft(secret)
    stego_fft = cover_fft + alpha * secret_fft
    stego_audio = np.fft.ifft(stego_fft).real
    return stego_audio / np.max(np.abs(stego_audio))

def main():
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    cover_audio, sr = load_audio(config["audio"]["cover_audio_path"])
    secret_audio, _ = load_audio(config["audio"]["secret_audio_path"])

    max_len = max(len(cover_audio), len(secret_audio))
    cover_audio = np.pad(cover_audio, (0, max_len - len(cover_audio)))
    secret_audio = np.pad(secret_audio, (0, max_len - len(secret_audio)))

    stego_audio = embed_frequency(cover_audio, secret_audio, config["audio"]["alpha"])
    save_audio(config["audio"]["stego_audio_path"], stego_audio, sr)
    print(f"Stego audio saved to {config['audio']['stego_audio_path']}")

    if __name__ == "__main__":
        main()
