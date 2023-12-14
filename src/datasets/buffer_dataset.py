import os
import librosa
import numpy as np
import scipy
import os
import librosa
import torch
from torch.utils.data import Dataset
from torchaudio.transforms import LFCC
from torchaudio.functional import compute_deltas
import torch.nn.functional as F


def compute_log_spectral_energy(audio, frame_length, hop_length, n_fft):
    stft = librosa.stft(
        audio, n_fft=n_fft, hop_length=hop_length, win_length=frame_length
    )
    spectral_energy = np.sum(np.abs(stft) ** 2, axis=0)
    log_spectral_energy = np.log(spectral_energy + 1e-8)
    return log_spectral_energy


def _calc_stft(wave) -> torch.Tensor:
    stft = torch.stft(
        wave,
        n_fft=1724,
        hop_length=int(16_000 * 0.0081),
        window=torch.blackman_window(1724),
        return_complex=True,
    )
    amplitude = torch.abs(stft)
    ref_value = amplitude.max()
    eps = 1e-6
    amp_db = 20 * torch.log10(amplitude / (ref_value + eps) + eps)
    return amp_db.type(torch.float32)


class LADataset(Dataset):
    def __init__(
        self,
        wav_dir,
        txt_path,
        sr=16_000,
        max_len=600,
        max_audio_len=4,
        use_buffer=False,
        type="STFT",
        **kwargs,
    ):
        self.wav_dir = wav_dir
        self.sr = sr
        self.max_length = max_len
        self.max_audio_len = max_audio_len

        try:
            self.protocols = self._load_protocols(txt_path)
        except Exception as e:
            print(e)
            self.protocols = []
        self.transform = self._create_transform(sr, type)

        self.buffer = []
        self.type = type
        if use_buffer:
            if type in ["STFT", "LFCC"]:
                self.buffer = [self._process_audio(entry) for entry in self.protocols]
            else:
                self.buffer = [self._just_load_audio(entry) for entry in self.protocols]

    def __len__(self):
        return len(self.protocols)

    def __getitem__(self, idx):
        if self.buffer:
            return self.buffer[idx]
        elif self.type in ["STFT", "LFCC"]:
            return self._process_audio(self.protocols[idx])
        else:
            return self._just_load_audio(self.protocols[idx])

    def _load_protocols(self, txt_path):
        with open(txt_path, "r") as f:
            return [
                {
                    "path": line.strip().split()[1] + ".flac",
                    "type": line.strip().split()[-1],
                }
                for line in f
            ]

    def _create_transform(self, sr, type):
        if type == "LFCC":
            return LFCC(
                sample_rate=sr,
                n_lfcc=60,
                n_filter=20,
                log_lf=True,
                norm="ortho",
                dct_type=2,
                speckwargs={
                    "n_fft": 512,
                    "win_length": int(sr * 0.02),
                    "hop_length": int(sr * 0.01),
                    "normalized": False,
                    "window_fn": torch.hamming_window,
                },
            )
        return lambda x: _calc_stft(x)

    def _just_load_audio(self, entry):
        target_frames = self.sr * self.max_audio_len
        audio, _ = librosa.load(
            os.path.join(self.wav_dir, entry["path"]),
            sr=self.sr,
            duration=self.max_audio_len,
        )
        audio = np.pad(
            audio, pad_width=(0, target_frames - audio.shape[0]), mode="wrap"
        )[:target_frames]
        return {
            "wave": torch.from_numpy(audio),
            "target": int(entry["type"] == "bonafide"),
        }

    def _process_audio(self, entry):
        audio, _ = librosa.load(
            os.path.join(self.wav_dir, entry["path"]),
            sr=self.sr,
            duration=self.max_audio_len,
        )
        mel = self.transform(torch.from_numpy(audio))
        if self.type == "LFCC":
            mel = self.transform(torch.from_numpy(audio))
            log_energy = compute_log_spectral_energy(
                audio, int(self.sr * 0.02), int(self.sr * 0.01), 512
            )
            mel[0, :] = torch.from_numpy(log_energy)
        mel = F.pad(mel, (0, max(0, self.max_length - mel.shape[1])))[
            :, : self.max_length
        ]
        if self.type == "LFCC":
            deltas = compute_deltas(mel)
            delta_deltas = compute_deltas(deltas)
            mel = torch.concat([mel, deltas, delta_deltas], dim=0)

        return {
            "target": int(entry["type"] == "bonafide"),
            "mel": mel,
        }
