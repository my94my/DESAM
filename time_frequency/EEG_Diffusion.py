from typing import Iterable, Tuple

import numpy as np
import torch
from librosa.beat import beat_track
from PIL import Image
from tqdm.auto import tqdm

from pipeline_eeg_diffusion import EEGDiffusionPipeline


class EEGDiffusion:
    def __init__(
        self,
        model_path: str = None,
        cuda: bool = True,
        progress_bar: Iterable = tqdm,
    ):
        """Class for generating EEG spectrogram using De-noising Diffusion Probabilistic Models.

        Args:
            model_path (String): name of model (local directory or Hugging Face Hub)
            cuda (bool): use CUDA?
            progress_bar (iterable): iterable callback for progress updates or None
        """
        self.model_path = model_path
        self.pipe = EEGDiffusionPipeline.from_pretrained(self.model_path)
        if cuda:
            self.pipe.to("cuda")
        self.progress_bar = progress_bar or (lambda _: _)

    def generate_spectrogram(
        self,
        steps: int = None,
        generator: torch.Generator = None,
        step_generator: torch.Generator = None,
        eta: float = 0,
        noise: torch.Tensor = None,
        encoding: torch.Tensor = None,
    ) -> Image.Image:
        """Generate random mel spectrogram and convert to audio.

        Args:
            steps (int): number of de-noising steps (defaults to 50 for DDIM, 1000 for DDPM)
            generator (torch.Generator): random number generator or None
            step_generator (torch.Generator): random number generator used to de-noise or None
            eta (float): parameter between 0 and 1 used with DDIM scheduler
            noise (torch.Tensor): noisy image or None
            encoding (`torch.Tensor`): for UNet2DConditionModel shape (batch_size, seq_length, cross_attention_dim)

        Returns:
            PIL Image: spectrogram
        """
        images = self.pipe(
            batch_size=1,
            steps=steps,
            generator=generator,
            step_generator=step_generator,
            eta=eta,
            noise=noise,
            encoding=encoding,
            return_dict=False,
        )
        return images[0]

    def generate_spectrogram_from_spectrogram(
        self,
        eeg_image_file: str = None,
        eeg_fragment_length: float = 0,
        start_step: int = 0,
        steps: int = None,
        generator: torch.Generator = None,
        mask_start_secs: float = 0,
        mask_end_secs: float = 0,
        step_generator: torch.Generator = None,
        eta: float = 0,
        encoding: torch.Tensor = None,
        noise: torch.Tensor = None,
    ) -> Tuple[Image.Image, Tuple[int, np.ndarray]]:
        """Generate random mel spectrogram from audio input and convert to audio.

        Args:
            eeg_image_file (str): must be a file on disk due to Librosa limitation
            slice (int): slice number of audio to convert
            start_step (int): step to start from
            steps (int): number of de-noising steps (defaults to 50 for DDIM, 1000 for DDPM)
            generator (torch.Generator): random number generator or None
            mask_start_secs (float): number of seconds of audio to mask (not generate) at start
            mask_end_secs (float): number of seconds of audio to mask (not generate) at end
            step_generator (torch.Generator): random number generator used to de-noise or None
            eta (float): parameter between 0 and 1 used with DDIM scheduler
            encoding (`torch.Tensor`): for UNet2DConditionModel shape (batch_size, seq_length, cross_attention_dim)
            noise (torch.Tensor): noisy image or None

        Returns:
            PIL Image:
        """

        images = self.pipe(
            batch_size=1,
            eeg_image_file=eeg_image_file,
            eeg_fragment_length=eeg_fragment_length,
            start_step=start_step,
            steps=steps,
            generator=generator,
            mask_start_secs=mask_start_secs,
            mask_end_secs=mask_end_secs,
            step_generator=step_generator,
            eta=eta,
            noise=noise,
            encoding=encoding,
            return_dict=False,
        )
        return images[0]

    @staticmethod
    def loop_it(audio: np.ndarray, sample_rate: int, loops: int = 12) -> np.ndarray:
        """Loop audio

        Args:
            audio (np.ndarray): audio as numpy array
            sample_rate (int): sample rate of audio
            loops (int): number of times to loop

        Returns:
            (float, np.ndarray): sample rate and raw audio or None
        """
        _, beats = beat_track(y=audio, sr=sample_rate, units="samples")
        beats_in_bar = (len(beats) - 1) // 4 * 4
        if beats_in_bar > 0:
            return np.tile(audio[beats[0] : beats[beats_in_bar]], loops)
        return None
