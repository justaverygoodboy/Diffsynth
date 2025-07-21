import wave
import numpy as np
import torch


def load_audio_features(audio_path: str, fps: int, num_frames: int, target_sr: int = 16000) -> torch.Tensor:
    """Load an audio file and return simple per-frame features.

    Each frame corresponds to 1/fps seconds of audio. For each frame
    we compute the mean and standard deviation of the waveform values.
    This minimal representation avoids external dependencies.
    """
    with wave.open(audio_path, 'rb') as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)
    audio = audio.astype(np.float32) / 32768.0

    if sr != target_sr:
        ratio = target_sr / sr
        indices = np.arange(0, len(audio) * ratio, dtype=np.float32) / ratio
        audio = np.interp(indices, np.arange(len(audio)), audio)
        sr = target_sr

    total_len = int(num_frames / fps * sr)
    audio = audio[:total_len]
    step = sr // fps
    feats = []
    for i in range(num_frames):
        start = i * step
        end = start + step
        chunk = audio[start:end]
        if len(chunk) == 0:
            break
        feats.append([float(chunk.mean()), float(chunk.std())])
    if not feats:
        return torch.zeros((num_frames, 2), dtype=torch.float32)
    return torch.tensor(feats, dtype=torch.float32)

