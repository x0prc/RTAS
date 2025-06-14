import torch
import torchaudio
import torchaudio.transforms as T
import sys
from typing import Optional

class Encode:
    def __init__(self, device='cuda', frame_size=2048):
        self.device = device
        self.frame_size = frame_size
        self.hop_length = frame_size // 4

        self.spectrogram = T.Spectrogram(
            n_fft=frame_size,
            hop_length=self.hop_length,
            power=None
        ).to(device)

        self.inverse_spectrogram = T.InverseSpectrogram(
            n_fft=frame_size,
            hop_length=self.hop_length
        ).to(device)

    def lsb_embed(self, cover: torch.Tensor, secret: torch.Tensor, num_bits: int = 2) -> torch.Tensor:
        scale_factor = 2**(16 - num_bits)
        secret_quantized = torch.round(secret * scale_factor).short()
        mask = (0xFF << (8 - num_bits)) & 0xFF

        cover_int = cover.short()
        stego_int = (cover_int & ~mask) | (secret_quantized << (8 - num_bits))
        return stego_int.float() / 32767.0

    def fft_embed(self, cover: torch.Tensor, secret: torch.Tensor, strength: float = 0.01) -> torch.Tensor:
        spec = self.spectrogram(cover)
        secret_spec = self.spectrogram(secret)

        phase_noise = strength * secret_spec.angle()
        modified_spec = spec * torch.exp(1j * phase_noise)
        return self.inverse_spectrogram(modified_spec, cover.shape[-1])

    def echo_hide(self, cover: torch.Tensor, secret: torch.Tensor, delay: float = 0.1, decay: float = 0.3) -> torch.Tensor:
        delay_samples = int(delay * 16000)  # 16kHz sample rate
        echo_filter = torch.zeros(delay_samples + 1, device=self.device)
        echo_filter[0] = 1
        echo_filter[-1] = decay

        secret_echo = torchaudio.functional.fftconvolve(secret, echo_filter)
        padded_secret = torch.nn.functional.pad(secret_echo, (0, cover.shape[-1] - secret_echo.shape[-1]))
        return cover + padded_secret * 0.1


class RealTimeProcessor:
    def __init__(self, method: str = 'fft', device: str = 'cuda'):
        self.method = method
        self.device = device
        self.stegano = Encode(device=device)
        self.stream: Optional[torchaudio.io.StreamReader] = None

        try:
            self.stream = torchaudio.io.StreamReader(
                src=":0",
                format="alsa" if 'linux' in sys.platform else "avfoundation"
            )
            self.stream.add_basic_audio_stream(
                frames_per_chunk=2048,
                buffer_chunk_size=4,
                sample_rate=16000
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize audio stream: {str(e)}") from e

    def realtime_encode(self, secret_audio: torch.Tensor) -> None:
        if self.stream is None:
            raise RuntimeError("Audio stream not initialized")

        for chunk in self.stream.stream():
            if chunk is None or len(chunk) == 0:
                continue

            try:
                cover = chunk[0].to(self.device)
                secret = secret_audio[:cover.shape[-1]]

                processed: Optional[torch.Tensor] = None
                if self.method == 'lsb':
                    processed = self.stegano.lsb_embed(cover, secret)
                elif self.method == 'fft':
                    processed = self.stegano.fft_embed(cover, secret)

                if processed is not None:
                    torchaudio.io.play_audio(processed.cpu(), 16000)
            except Exception as e:
                print(f"Error processing audio chunk: {str(e)}")
                continue

    def realtime_decode(self) -> None:
        if self.stream is None:
            raise RuntimeError("Audio stream not initialized")

        for chunk in self.stream.stream():
            if chunk is None or len(chunk) == 0:
                continue

            try:
                stego = chunk[0].to(self.device)
                extracted: Optional[torch.Tensor] = None

                if self.method == 'lsb':
                    extracted = self._lsb_extract(stego)
                elif self.method == 'fft':
                    extracted = self._fft_extract(stego)

                if extracted is not None:
                    torchaudio.io.play_audio(extracted.cpu(), 16000)
            except Exception as e:
                print(f"Error decoding audio chunk: {str(e)}")
                continue

    def _lsb_extract(self, stego: torch.Tensor) -> torch.Tensor:
        return (stego.short() & 0x03).float() / 3.0

    def _fft_extract(self, stego: torch.Tensor) -> torch.Tensor:
        spec = self.stegano.spectrogram(stego)
        return torch.angle(spec) / 0.01
