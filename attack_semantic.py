import warnings
warnings.filterwarnings("ignore")

import torch
import os
import torch
import torch.nn.functional as F
import librosa
import torchaudio
import s3tokenizer

import soundfile as sf
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from transformers import (
    HubertModel,
    Wav2Vec2FeatureExtractor,
    WhisperModel,
    WhisperFeatureExtractor,
)

from masker import Masker

class DifferentiableWhisperMel(torch.nn.Module):
    def __init__(self, feature_extractor: WhisperFeatureExtractor, device):
        super().__init__()
        self.n_fft = feature_extractor.n_fft
        self.hop_length = feature_extractor.hop_length
        self.n_mels = feature_extractor.feature_size
        self.sampling_rate = feature_extractor.sampling_rate
        self.n_samples = feature_extractor.n_samples
        self.max_frames = self.n_samples // self.hop_length

        mel_filters = librosa.filters.mel(
            sr=self.sampling_rate, n_fft=self.n_fft, n_mels=self.n_mels
        )
        self.register_buffer("mel_filters", torch.from_numpy(mel_filters).float())

        window = torch.hann_window(self.n_fft)
        self.register_buffer("window", window)

        self.to(device)

    def forward(self, wave_16k):
        wav = wave_16k.squeeze(0)
        target_len = self.n_samples
        if wav.shape[-1] < target_len:
            wav = F.pad(wav, (0, target_len - wav.shape[-1]))
        else:
            wav = wav[..., :target_len]

        stft = torch.stft(
            wav, n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, return_complex=True, center=True,
        )
        magnitudes = stft.abs() ** 2
        mel_spec = self.mel_filters @ magnitudes
        log_mel = torch.clamp(mel_spec, min=1e-10).log10()
        log_mel = torch.clamp((log_mel + 4.0) / 4.0, min=0.0)

        if log_mel.dim() == 2:
            log_mel = log_mel.unsqueeze(0)
        if log_mel.shape[-1] < self.max_frames:
            log_mel = F.pad(log_mel, (0, self.max_frames - log_mel.shape[-1]))
        else:
            log_mel = log_mel[..., :self.max_frames]

        return log_mel


class T_SemAttack():
    def __init__(self, epsilon, max_items, device):
        self.device = device

        self.epsilon = epsilon
        self.max_items = max_items

        self.initialize()

        self.lr_optim = 1e-3
        self.sampling_rate = 16000

        self.criterion = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

        self.build_tokenizer_encoders()


    def param_no_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False


    def initialize(self):
        self.plateau_length = 5
        self.max_lr = 0.001
        self.min_lr = 1e-6
        self.plateau_drop = 2.
        self.last_ls = []
        self.lr = self.max_lr


    def save_audio(self, output_path, wave, sr=None):
        if isinstance(wave, torch.Tensor):
            wave = wave.detach().cpu().numpy()

        sf.write(output_path, wave, samplerate=self.sr)


    def calculate_snr(self, wave, adv_wave):
        if isinstance(wave, np.ndarray):
            wave = torch.from_numpy(wave)
        if isinstance(adv_wave, np.ndarray):
            adv_wave = torch.from_numpy(adv_wave)

        noise = adv_wave - wave

        signal_power = torch.sum(wave ** 2)
        noise_power = torch.sum(noise ** 2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))

        return snr


    def build_tokenizer_encoders(self):
        # 1. Semantic Tokenizer: S3 Tokenizer (CosyVoice v1)
        # https://huggingface.co/model-scope/CosyVoice-300M/blob/main/speech_tokenizer_v1.onnx
        tokenizer_path = "path_to_speech_tokenizer/speech_tokenizer_v1.onnx"
        self.speech_tokenizer = s3tokenizer.load_model(tokenizer_path).to(self.device)
        self.encoder_s3 = self.speech_tokenizer.encoder
        self.param_no_grad(self.encoder_s3)

        # 2. Semantic Tokenizer: HuBERT-Large
        hubert_path = "facebook/hubert-large-ll60k"
        self.hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_path)
        self.encoder_hubert = HubertModel.from_pretrained(hubert_path, torch_dtype=torch.float32).to(self.device)
        self.encoder_hubert.eval()
        self.param_no_grad(self.encoder_hubert)

        self.hubert_do_normalize = self.hubert_feature_extractor.do_normalize

        # 3. Semantic Tokenizer: Whisper-Large-v3 (encoder only)
        whisper_path = "openai/whisper-large-v3"
        whisper_fe = WhisperFeatureExtractor.from_pretrained(whisper_path)
        self.whisper_mel = DifferentiableWhisperMel(whisper_fe, self.device)
        self.encoder_whisper = WhisperModel.from_pretrained(whisper_path, torch_dtype=torch.float32).to(self.device).encoder
        self.encoder_whisper.eval()
        self.param_no_grad(self.encoder_whisper)

        self.whisper_valid_frames = None

        # 4. Acoustic Encoder: MFCC
        self.encoder_mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.sampling_rate, n_mfcc=40
        ).to(self.device)

        print("[Guard] Loaded: S3 Tokenizer, HuBERT-Large, Whisper-Large-v3, MFCC")


    def _extract_hubert(self, wave_16k):
        wav = wave_16k.squeeze(0)
        if self.hubert_do_normalize:
            wav = (wav - wav.mean()) / (wav.std() + 1e-7)
        input_values = wav.unsqueeze(0)

        outputs = self.encoder_hubert(input_values=input_values)
        return outputs.last_hidden_state


    def _extract_whisper(self, wave_16k):
        actual_samples = wave_16k.shape[-1]
        actual_frames = actual_samples // self.whisper_mel.hop_length
        self.whisper_valid_frames = min(actual_frames, self.whisper_mel.max_frames)

        input_features = self.whisper_mel(wave_16k)
        outputs = self.encoder_whisper(input_features=input_features)
        return outputs.last_hidden_state


    def _extract_s3(self, wave_16k):
        mel = s3tokenizer.log_mel_spectrogram(wave_16k.squeeze(0)).to(self.device)
        mel, mel_len = s3tokenizer.padding([mel])
        mel_len = mel_len.to(self.device)

        hidden, _ = self.encoder_s3(mel, mel_len)
        return hidden.transpose(2, 1)


    def _extract_mfcc(self, wave_16k):
        return self.encoder_mfcc(wave_16k)


    def get_vectors(self, wave_16k):
        vec_s3 = self._extract_s3(wave_16k)
        vec_hubert = self._extract_hubert(wave_16k)
        vec_whisper = self._extract_whisper(wave_16k)
        vec_mfcc = self._extract_mfcc(wave_16k)

        return [vec_s3, vec_hubert, vec_whisper, vec_mfcc]


    def compute_feature_loss(self, vecs_1, vecs_2):
        loss = 0.0
        weight_list = [1.0, 1.0, 1.0, 1.0]

        for index, (vec_1, vec_2) in enumerate(zip(vecs_1, vecs_2)):
            if vec_1.dim() == 2:
                vec_1 = vec_1.unsqueeze(0)
            if vec_2.dim() == 2:
                vec_2 = vec_2.unsqueeze(0)

            if index == 2 and self.whisper_valid_frames is not None:
                n_valid = self.whisper_valid_frames
                n_valid_enc = max(1, n_valid // 2)
                vec_1 = vec_1[:, :n_valid_enc, :]
                vec_2 = vec_2[:, :n_valid_enc, :]

            try:
                score = self.criterion(vec_1, vec_2).mean()
            except RuntimeError:
                min_len = min(vec_1.shape[-1], vec_2.shape[-1])
                min_seq = min(vec_1.shape[-2], vec_2.shape[-2])
                vec_1 = vec_1[..., :min_seq, :min_len]
                vec_2 = vec_2[..., :min_seq, :min_len]
                score = self.criterion(vec_1, vec_2).mean()

            loss = loss + score * weight_list[index]

        return loss



    def start_attack(self, audio_path, verbose=False):
        wave, sr = torchaudio.load(audio_path)
        wave = wave.to(self.device)
        self.sr = sr

        noise = torch.zeros(wave.shape).to(self.device)
        adv_wave = torch.clamp(wave + noise, min=-1., max=1.)
        adv_wave.requires_grad = True

        transform_16k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(self.device)
        wave_16k = transform_16k(wave)
        wave_vectors = self.get_vectors(wave_16k)

        opt_noise = optim.SGD([adv_wave], lr=self.lr_optim, weight_decay=0.95)

        # Psychoacoustic masking
        msker = Masker(device=self.device, sample_rate=sr)
        theta, original_max_psd = msker._compute_masking_threshold(wave[0].cpu().numpy())
        theta = torch.FloatTensor(theta.transpose(1, 0))
        original_max_psd = torch.FloatTensor([original_max_psd])
        theta_batch = theta.unsqueeze(0)
        original_max_psd_batch = original_max_psd.unsqueeze(0)

        for step in tqdm(range(self.max_items)):
            self.step = step
            adv_wave_16k = transform_16k(adv_wave)

            ###### Semantic Loss ######
            adv_vectors = self.get_vectors(adv_wave_16k)
            loss_feature = self.compute_feature_loss(wave_vectors, adv_vectors)

            ###### Psychoacoustic Loss ######
            loss_psy = msker.batch_forward_2nd_stage(
                local_delta_rescale=(adv_wave - wave).squeeze(1),
                theta_batch=theta_batch.to(self.device),
                original_max_psd_batch=original_max_psd_batch.to(self.device),
            )

            loss_psy_max = 1.0e7
            if torch.isnan(loss_psy) or torch.isinf(loss_psy):
                loss_psy = torch.FloatTensor([loss_psy_max]).to(self.device)
            elif loss_psy > loss_psy_max:
                loss_psy = torch.FloatTensor([loss_psy_max]).to(self.device)

            ###### L2 norm ######
            loss_l2 = torch.norm(adv_wave - wave, p=2)

            ###### Total Loss ######
            weight_feature = 100
            weight_psy = 1e-3
            weight_l2 = 0.1

            loss = loss_feature * weight_feature + loss_psy * weight_psy + loss_l2 * weight_l2
            loss_items = {
                "loss_emb": f"{loss_feature.item():.6f}",
                "loss_psy": f"{loss_psy.item():.6f}",
                "loss_l2": f"{loss_l2.item():.6f}",
            }

            if step % 50 == 0 and verbose:
                print(step, loss_items, self.lr)

            opt_noise.zero_grad()
            loss.backward()
            opt_noise.step()


            ### Plateau LR scheduling
            self.last_ls.append(loss)
            self.last_ls = self.last_ls[-self.plateau_length:]
            if self.last_ls[-1] > self.last_ls[0] and len(self.last_ls) == self.plateau_length:
                if self.lr > self.min_lr:
                    self.lr = max(self.lr / self.plateau_drop, self.min_lr)
                self.last_ls = []


            ### Projected Gradient Descent
            delta = self.lr * torch.sign(adv_wave.grad) * -1.
            adv_wave = delta + adv_wave
            noise = torch.clamp(adv_wave.data - wave, min=-self.epsilon, max=self.epsilon)
            adv_wave = torch.clamp(wave + noise, min=-1., max=1.)
            adv_wave.requires_grad = True

        snr = self.calculate_snr(wave, adv_wave.detach()).item()
        print(f"The SNR is {snr:.3f}")

        return adv_wave.squeeze().detach().cpu().numpy(), loss_items, snr


if __name__ == "__main__":
    audio_path = "examples/0_train_117.wav"
    output_path = "examples/0_train_117_semantic.wav"

    epsilon = 8 / 255
    max_items = 500
    device = torch.device("cuda")
    verbose = False

    defender = T_SemAttack(epsilon, max_items, device)

    adv_wave, loss_items, snr = defender.start_attack(audio_path, verbose=verbose)
    defender.save_audio(output_path, adv_wave)
