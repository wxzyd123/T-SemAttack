import torch
from typing import List, Optional, Union

from modules.dac import DAC


class ROSETok(DAC):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 5, 8],
        latent_dim: Optional[int] = 1280,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 5, 4, 2],

        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,

        sample_rate: int = 16000,
    ):
        super().__init__(
            encoder_dim=encoder_dim,
            encoder_rates=encoder_rates,
            latent_dim=latent_dim,
            decoder_dim=decoder_dim,
            decoder_rates=decoder_rates,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
            sample_rate=sample_rate,
        )

        self._metadata = {
            "encoder_dim": encoder_dim,
            "encoder_rates": encoder_rates,
            "latent_dim": latent_dim,
            "decoder_dim": decoder_dim,
            "decoder_rates": decoder_rates,
            "n_codebooks": n_codebooks,
            "codebook_size": codebook_size,
            "codebook_dim": codebook_dim,
            "quantizer_dropout": quantizer_dropout,
            "sample_rate": sample_rate,
        }


    @staticmethod
    def _to_mono_batch(audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 1:
            return audio.unsqueeze(0).unsqueeze(1)
        if audio.ndim == 2:
            return audio.unsqueeze(1)
        if audio.ndim == 3 and audio.shape[1] == 1:
            return audio
        raise ValueError(f"Expected audio shape (T,), (B,T), or (B,1,T), got {tuple(audio.shape)}")


    def forward(
        self,
        audio_data: Optional[torch.Tensor] = None,
        clean_audio: Optional[torch.Tensor] = None,
        sample_rate: Optional[int] = None,
        n_quantizers: Optional[int] = None,
    ):
        if clean_audio is None:
            clean_audio = audio_data

        audio_data = self._to_mono_batch(audio_data)
        clean_audio = self._to_mono_batch(clean_audio)
        target_length = clean_audio.shape[-1]

        audio_data = self.preprocess(audio_data, sample_rate)
        z = self.encoder(audio_data)

        z_q, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            z, n_quantizers
        )
        audio_hat = self.decode(z_q)[..., :target_length]

        return {
            "audio": audio_hat,
            "z": z_q,
            "encoded": z,
            "codes": codes,
            "latents": latents,
            "codec/commitment_loss": commitment_loss,
            "codec/codebook_loss": codebook_loss,
        }