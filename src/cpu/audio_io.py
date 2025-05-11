import soundfile as sf
import numpy as np
import yaml
import os

config_path = os.path.join(os.path.dirname(__file__), "../../config.yaml")
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def load_audio(file_path):
    data, samplerate = sf.read(file_path)
    return data / np.max(np.abs(data)), samplerate

def save_audio(file_path, data, samplerate):
    sf.write(file_path, data, samplerate)
