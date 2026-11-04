import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union
from transformers import WhisperFeatureExtractor, WhisperModel


class WhisperTorchFeatureExtractor(nn.Module):
    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        chunk_length: int = 30,
        n_fft: int = 400,
        padding_value: float = 0.0,
        dither: float = 0.0,
        mel_filters: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_fft = n_fft
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.padding_value = padding_value
        self.dither = dither

        if mel_filters is None:
            from transformers.audio_utils import mel_filter_bank

            mel_filters = mel_filter_bank(
                num_frequency_bins=1 + n_fft // 2,
                num_mel_filters=feature_size,
                min_frequency=0.0,
                max_frequency=sampling_rate / 2.0,
                sampling_rate=sampling_rate,
                norm="slaney",
                mel_scale="slaney",
            )
        if isinstance(mel_filters, np.ndarray):
            mel_filters = torch.from_numpy(mel_filters).float()

        self.register_buffer("mel_filters", mel_filters, persistent=False)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    @classmethod
    def from_pretrained(cls, whisper_path: str) -> "WhisperTorchFeatureExtractor":
        fe = WhisperFeatureExtractor.from_pretrained(whisper_path)
        return cls(
            feature_size=fe.feature_size,
            sampling_rate=fe.sampling_rate,
            hop_length=fe.hop_length,
            chunk_length=fe.chunk_length,
            n_fft=fe.n_fft,
            padding_value=fe.padding_value,
            dither=getattr(fe, "dither", 0.0),
            mel_filters=fe.mel_filters,
        )

    def forward(
        self,
        waveform: torch.Tensor,
        return_attention_mask: bool = True,
    ) -> dict:
        if not isinstance(waveform, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor input, got {type(waveform)}")
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 2:
            raise ValueError(
                "Expected waveform of shape (batch, num_samples) or (num_samples,), "
                f"got {tuple(waveform.shape)}"
            )

        batch_size, input_length = waveform.shape
        device = waveform.device

        if input_length > self.n_samples:
            waveform = waveform[:, : self.n_samples]
            input_length = self.n_samples

        if input_length < self.n_samples:
            pad = waveform.new_full(
                (batch_size, self.n_samples - input_length),
                fill_value=self.padding_value,
            )
            waveform_padded = torch.cat([waveform, pad], dim=1)
        else:
            waveform_padded = waveform

        waveform_padded = waveform_padded.float()
        if self.dither != 0.0:
            waveform_padded = waveform_padded + self.dither * torch.randn_like(waveform_padded)

        stft = torch.stft(
            waveform_padded,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2

        mel_spec = self.mel_filters.T @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()

        max_val = log_spec.amax(dim=(-2, -1), keepdim=True)
        log_spec = torch.maximum(log_spec, max_val - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        outputs = {"input_features": log_spec}

        if return_attention_mask:
            attention_mask_samples = torch.zeros(
                batch_size, self.n_samples, dtype=torch.long, device=device
            )
            attention_mask_samples[:, :input_length] = 1
            rescaled = attention_mask_samples[:, :: self.hop_length]

            if self.n_samples % self.hop_length != 0:
                rescaled = rescaled[:, :-1]
            outputs["attention_mask"] = rescaled

        return outputs


class WhisperTeacher(nn.Module):
    def __init__(self, whisper_path: str = "openai/whisper-large-v3"):
        super().__init__()
        self.feature_extractor = WhisperTorchFeatureExtractor.from_pretrained(whisper_path)
        self.model = WhisperModel.from_pretrained(whisper_path).get_encoder()

    def forward(self, audio: torch.Tensor):
        if audio.ndim == 3:
            audio = audio.squeeze(1)
            
        inputs = self.feature_extractor(audio, return_attention_mask=True)
        outputs = self.model(input_features=inputs["input_features"], attention_mask=inputs["attention_mask"]).last_hidden_state

        return outputs