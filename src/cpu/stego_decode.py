import torch
import torchaudio
import torchaudio.transforms as T

class Decode:
    def __init__(self, device='cuda', frame_size=2048):
        self.device = device
        self.frame_size = frame_size
        self.hop_length = frame_size // 4
        self.spectrogram = T.Spectrogram(n_fft=frame_size, hop_length=self.hop_length, power=None).to(device)
        self.inverse_spectrogram = T.InverseSpectrogram(n_fft=frame_size, hop_length=self.hop_length).to(device)

    def decode_lsb(self, stego_audio: torch.Tensor, num_bits=2):
        stego_int = (stego_audio * 32767).short()
        mask = (1 << num_bits) - 1
        extracted = (stego_int & mask).float() / mask
        return extracted

    def decode_fft(self, stego_audio: torch.Tensor, strength=0.01):
        spec = self.spectrogram(stego_audio)
        phase = torch.angle(spec)
        secret_phase = phase / strength
        secret_spec = torch.abs(spec) * torch.exp(1j * secret_phase)
        extracted = self.inverse_spectrogram(secret_spec, length=stego_audio.shape[-1])
        return extracted

    def decode_echo(self, stego_audio: torch.Tensor, delay=0.1, decay=0.3, sample_rate=16000):
        delay_samples = int(delay * sample_rate)
        echo_kernel = torch.zeros(delay_samples + 1, device=self.device)
        echo_kernel[0] = 1.0
        echo_kernel[-1] = -decay
        extracted = torchaudio.functional.fftconvolve(stego_audio, echo_kernel, mode='same')
        return extracted / (1 + decay)

    def decode_adaptive(self, stego_audio: torch.Tensor):
        # Try all methods, return the one with highest variance (simple heuristic)
        lsb = self.decode_lsb(stego_audio)
        fft = self.decode_fft(stego_audio)
        echo = self.decode_echo(stego_audio)
        candidates = [lsb, fft, echo]
        scores = [c.var().item() for c in candidates]
        return candidates[scores.index(max(scores))]
