"""
Implementation of the basic_pitch model in PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from typing import Optional
import nnAudio.features.cqt as nnAudio
import utils

MIDI_OFFSET = 21


class harmonic_stacking(nn.Module):
    """
    Harmonic stacking layer

    Input: (batch, freq_bins, time_frames)

    Args:
        bins_per_semitone: The number of bins per semitone in the CQT
        harmonics: The list of harmonics to stack
        n_output_freqs: The number of output frequencies to return in each harmonic layer

    Returns:
        (batch, n_harmonics, out_freq_bins, time_frames)
    """

    def __init__(self, bins_per_semitone: int, harmonics: int, n_output_freqs: int):
        super(harmonic_stacking, self).__init__()

        self.bins_per_semitone = bins_per_semitone
        self.harmonics = harmonics
        self.n_output_freqs = n_output_freqs
        self.shifts = [
            int(np.round(12 * self.bins_per_semitone * np.log2(float(h))))
            for h in self.harmonics
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        torch._assert(len(x.shape) == 3, "x must be (batch, freq_bins, time_frames)")
        channels = []
        for shift in self.shifts:
            if shift == 0:
                channels.append(x)
            elif shift > 0:
                channels.append(F.pad(x[:, shift:, :], (0, 0, 0, shift)))
            elif shift < 0:
                channels.append(F.pad(x[:, :shift, :], (0, 0, -shift, 0)))
            else:
                raise ValueError("shift must be non-zero")
        x = torch.stack(channels, dim=1)
        x = x[:, :, : self.n_output_freqs, :]
        return x


class Conv2dSame(torch.nn.Conv2d):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def constrain_frequency(
    onsets: torch.Tensor,
    frames: torch.Tensor,
    max_freq: Optional[float],
    min_freq: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Constrain the frequency range of the pitch predictions by zeroing out bins outside the range

    Args:
        onsets: The onset predictions (batch, freq, time_frames)
        frames: The frame predictions (batch, freq, time_frames)
        max_freq: The maximum frequency to allow
        min_freq: The minimum frequency to allow

    Returns:
        The constrained onset and frame predictions
    """
    # check inputs are batched
    torch._assert(len(onsets.shape) == 3, "onsets must be (batch, freq, time_frames)")
    torch._assert(len(frames.shape) == 3, "frames must be (batch, freq, time_frames)")

    if max_freq is not None:
        max_freq_idx = int(np.round(utils.hz_to_midi(max_freq) - MIDI_OFFSET))
        onsets[:, max_freq_idx:, :].zero_()
        frames[:, max_freq_idx:, :].zero_()
    if min_freq is not None:
        min_freq_idx = int(np.round(utils.hz_to_midi(min_freq) - MIDI_OFFSET))
        onsets[:, :min_freq_idx, :].zero_()
        frames[:, :min_freq_idx, :].zero_()

    return onsets, frames


def get_infered_onsets(
    onsets: torch.Tensor, frames: torch.Tensor, n_diff: int = 2
) -> torch.Tensor:
    """
    Infer onsets from large changes in frame amplutude

    Args:
        onsets: The onset predictions (batch, freq, time_frames)
        frames: The frame predictions (batch, freq, time_frames)
        n_diff: DDifferences used to detect onsets

    Returns:
        The maximum between the predicted onsets and its differences
    """
    # check inputs are batched
    torch._assert(len(onsets.shape) == 3, "onsets must be (batch, freq, time_frames)")
    torch._assert(len(frames.shape) == 3, "frames must be (batch, freq, time_frames)")

    diffs = []
    for n in range(1, n_diff + 1):
        # Use PyTorch's efficient padding and slicing
        frames_padded = torch.nn.functional.pad(frames, (0, 0, n, 0))
        diffs.append(frames_padded[:, n:, :] - frames_padded[:, :-n, :])

    frame_diff = torch.min(torch.stack(diffs), dim=0)[0]
    frame_diff = torch.clamp(frame_diff, min=0)  # Replaces negative values with 0
    frame_diff[:, :n_diff, :].zero_()  # Set the first n_diff frames to 0

    # Rescale to have the same max as onsets
    frame_diff = frame_diff * torch.max(onsets) / torch.max(frame_diff)

    # Use the max of the predicted onsets and the differences
    max_onsets_diff = torch.max(onsets, frame_diff)

    return max_onsets_diff


def argrelmax(x: torch.Tensor) -> torch.Tensor:
    """
    Emulate scipy.signal.argrelmax with axis 1 and order 1 in torch

    Args:
        x: The input tensor (batch, freq_bins, time_frames)

    Returns:
        The indices of the local maxima
    """
    # Check inputs are batched
    torch._assert(len(x.shape) == 3, "x must be (batch, freq, time_frames)")

    frames_padded = torch.nn.functional.pad(x, (1, 1))

    diff1 = x - frames_padded[:, :, :-2]
    diff2 = x - frames_padded[:, :, 2:]

    diff1[:, :, :1].zero_()
    diff1[:, :, -1:].zero_()

    return torch.nonzero((diff1 > 0) * (diff2 > 0), as_tuple=True)


class basic_pitch_torch(nn.Module):
    """
    Port of basic_pitch pitch prediction to pytorch

    input: (batch, samples)
    returns: {"onset": (batch, freq_bins, time_frames),
              "contour": (batch, freq_bins, time_frames),
              "note": (batch, freq_bins, time_frames)}
    """

    def __init__(
        self,
        sr: int = 16000,
        hop_length: int = 256,
        annotation_semitones: int = 88,
        annotation_base: float = 27.5,
        n_harmonics: int = 8,
        n_filters_contour: int = 32,
        n_filters_onsets: int = 32,
        n_filters_notes: int = 32,
        no_contours: bool = False,
        contour_bins_per_semitone: int = 3,
        device: str = "mps",
    ):
        super(basic_pitch_torch, self).__init__()

        self.sr = sr
        self.hop_length = hop_length
        self.annotation_semitones = annotation_semitones
        self.annotation_base = annotation_base
        self.n_harmonics = n_harmonics
        self.n_filters_contour = n_filters_contour
        self.n_filters_onsets = n_filters_onsets
        self.n_filters_notes = n_filters_notes
        self.no_contours = no_contours
        self.contour_bins_per_semitone = contour_bins_per_semitone
        self.n_freq_bins_contour = annotation_semitones * contour_bins_per_semitone
        self.device = device

        max_semitones = int(
            np.floor(12.0 * np.log2(0.5 * self.sr / self.annotation_base))
        )

        n_semitones = np.min(
            [
                int(
                    np.ceil(12.0 * np.log2(self.n_harmonics))
                    + self.annotation_semitones
                ),
                max_semitones,
            ]
        )

        self.cqt = nnAudio.CQT2010v2(
            sr=self.sr,
            fmin=self.annotation_base,
            hop_length=self.hop_length,
            n_bins=n_semitones * self.contour_bins_per_semitone,
            bins_per_octave=12 * self.contour_bins_per_semitone,
            verbose=False,
        )

        self.contour_1 = nn.Sequential(
            # nn.Conv2d(self.n_harmonics, self.n_filters_contour, (5, 5), padding="same"),
            # nn.BatchNorm2d(self.n_filters_contour),
            # nn.ReLU(),
            # nn.Conv2d(self.n_filters_contour, 8, (3 * 13, 3), padding="same"),
            nn.Conv2d(self.n_harmonics, 8, (3 * 13, 3), padding="same"),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.contour_2 = nn.Sequential(
            nn.Conv2d(8, 1, (5, 5), padding="same"), nn.Sigmoid()
        )

        self.note_1 = nn.Sequential(
            Conv2dSame(1, self.n_filters_notes, (7, 7), (3, 1)),
            nn.ReLU(),
            nn.Conv2d(self.n_filters_notes, 1, (3, 7), padding="same"),
            nn.Sigmoid(),
        )

        self.onset_1 = nn.Sequential(
            Conv2dSame(self.n_harmonics, self.n_filters_onsets, (5, 5), (3, 1)),
            nn.BatchNorm2d(self.n_filters_onsets),
            nn.ReLU(),
        )

        self.onset_2 = nn.Sequential(
            nn.Conv2d(self.n_filters_onsets + 1, 1, (3, 3), padding="same"),
            nn.Sigmoid(),
        )

    def normalised_to_db(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert spectrogram to dB and normalise

        Args:
            audio: The spectrogram input. (batch, 1, freq_bins, time_frames)
                or (batch, freq_bins, time_frames)

        Returns:
            the spectogram in dB in the same shape as the input
        """
        power = torch.square(audio)
        log_power = 10.0 * torch.log10(power + 1e-10)

        log_power_min = torch.min(log_power, keepdim=True, dim=-2)[0]
        log_power_min = torch.min(log_power_min, keepdim=True, dim=-1)[0]
        log_power_offset = log_power - log_power_min
        log_power_offset_max = torch.max(log_power_offset, keepdim=True, dim=-2)[0]
        log_power_offset_max = torch.max(log_power_offset_max, keepdim=True, dim=-1)[0]

        log_power_normalised = log_power_offset / log_power_offset_max
        return log_power_normalised

    def get_cqt(self, audio: torch.Tensor, use_batchnorm: bool) -> torch.Tensor:
        """
        Compute CQT from audio

        Args:
            audio: The audio input. (batch, samples)
            n_harmonics: The number of harmonics to capture above the maximum output frequency.
                Used to calculate the number of semitones for the CQT.
            use_batchnorm: If True, applies batch normalization after computing the CQT


        Returns:
            the log-normalised CQT of audio (batch, freq_bins, time_frames)
        """

        torch._assert(
            len(audio.shape) == 2, "audio has multiple channels, only mono is supported"
        )
        x = self.cqt(audio)
        x = self.normalised_to_db(x)

        if use_batchnorm:
            x = torch.unsqueeze(x, 1)
            x = nn.BatchNorm2d(1).to(self.device)(x)
            x = torch.squeeze(x, 1)

        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.tensor]:
        x = self.get_cqt(x, use_batchnorm=True)

        if self.n_harmonics > 1:
            x = harmonic_stacking(
                self.contour_bins_per_semitone,
                [0.5] + list(range(1, self.n_harmonics)),
                self.n_freq_bins_contour,
            )(x)
        else:
            x = harmonic_stacking(
                self.contour_bins_per_semitone, [1], self.n_freq_bins_contour
            )(x)

        # contour layers
        x_contours = self.contour_1(x)

        if not self.no_contours:
            x_contours = self.contour_2(x_contours)
            x_contours = torch.squeeze(x_contours, 1)
            x_contours_reduced = torch.unsqueeze(x_contours, 1)
        else:
            x_contours_reduced = x_contours

        # notes layers
        x_notes_pre = self.note_1(x_contours_reduced)
        x_notes = torch.squeeze(x_notes_pre, 1)

        # onsets layers
        x_onset = self.onset_1(x)
        x_onset = torch.cat([x_notes_pre, x_onset], dim=1)
        x_onset = self.onset_2(x_onset)
        x_onset = torch.squeeze(x_onset, 1)

        return {"onset": x_onset, "contour": x_contours, "note": x_notes}
