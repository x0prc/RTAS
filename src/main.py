import argparse
import torch
import torchaudio
from cpu.stego_encode import Encode
from cpu.stego_decode import Decode

def main():
    parser = argparse.ArgumentParser(description="Real-Time Audio Steganography (PyTorchAudio, GPU)")
    parser.add_argument('--mode', choices=['encode', 'decode'], required=True)
    parser.add_argument('--method', choices=['lsb', 'fft', 'echo'], default='fft')
    parser.add_argument('--input', type=str, required=True, help="Input audio file path")
    parser.add_argument('--secret', type=str, help="Secret audio file path (for encoding)")
    parser.add_argument('--output', type=str, required=True, help="Output file path")
    parser.add_argument('--gpu', type=int, default=0, help="GPU device index")

    args = parser.parse_args()
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    if args.mode == 'encode':
        if not args.secret:
            print("Secret audio file required for encoding.")
            return
        cover, sr = torchaudio.load(args.input)
        secret, sr2 = torchaudio.load(args.secret)
        if sr2 != sr:
            secret = torchaudio.functional.resample(secret, sr2, sr)
        cover, secret = cover.to(device), secret.to(device)
        steg = Encode(device=device)
        if args.method == 'lsb':
            stego = steg.lsb_embed(cover, secret)
        elif args.method == 'fft':
            stego = steg.fft_embed(cover, secret)
        else:
            stego = steg.echo_hide(cover, secret)
        torchaudio.save(args.output, stego.cpu(), sr)
        print(f"Stego audio saved to {args.output}")

    elif args.mode == 'decode':
        stego, sr = torchaudio.load(args.input)
        stego = stego.to(device)
        decoder = Decode(device=device)
        if args.method == 'lsb':
            extracted = decoder.decode_lsb(stego)
        elif args.method == 'fft':
            extracted = decoder.decode_fft(stego)
        else:
            extracted = decoder.decode_echo(stego)
        torchaudio.save(args.output, extracted.cpu(), sr)
        print(f"Extracted audio saved to {args.output}")

if __name__ == "__main__":
    main()
