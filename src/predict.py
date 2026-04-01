
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn

from souldClassifier_final_client.src import sound_to_tensor
from souldClassifier_final_client.src.models import AudioResNet


class _LegacyDepthwiseSeparableConv2d(nn.Module):
    """Block name layout matches yamnet checkpoint keys (`bn_dw`, `bn_pw`)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.bn_pw(x)
        x = self.act(x)
        return x


class _YAMNetBackbone(nn.Module):
    """Checkpoint-compatible backbone for legacy yamnet.pt (`backbone.*` keys)."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        cfg = [
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (512, 2),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (1024, 2),
            (1024, 1),
        ]

        blocks = []
        in_channels = 32
        for out_channels, stride in cfg:
            blocks.append(_LegacyDepthwiseSeparableConv2d(in_channels, out_channels, stride))
            in_channels = out_channels

        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class _LegacyYAMNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = _YAMNetBackbone()
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.dropout(x)
        return self.classifier(x)


_MODEL_CACHE: dict[str, nn.Module] = {}


def _model_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "model"


def _load_rasnet_model() -> nn.Module:
    if "rasnet" in _MODEL_CACHE:
        return _MODEL_CACHE["rasnet"]

    state = torch.load(_model_dir() / "rasnet.pt", map_location="cpu")
    input_dim = int(state["input_layer.weight"].shape[1])
    num_classes = int(state["output_layer.weight"].shape[0])

    hidden_dims = [int(state["input_layer.weight"].shape[0])]
    idx = 0
    while f"res_blocks.{idx}.fc1.weight" in state:
        hidden_dims.append(int(state[f"res_blocks.{idx}.fc1.weight"].shape[0]))
        idx += 1

    model = AudioResNet(input_dim=input_dim, num_classes=num_classes, hidden_dims=hidden_dims)
    model.load_state_dict(state, strict=True)
    model.eval()
    _MODEL_CACHE["rasnet"] = model
    return model


def _load_yamnet_model() -> nn.Module:
    if "yamnet" in _MODEL_CACHE:
        return _MODEL_CACHE["yamnet"]

    ckpt = torch.load(_model_dir() / "yamnet.pt", map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    num_classes = int(state["classifier.weight"].shape[0])

    model = _LegacyYAMNet(num_classes=num_classes)
    model.load_state_dict(state, strict=True)
    model.eval()
    _MODEL_CACHE["yamnet"] = model
    return model


def _prepare_rasnet_input(soundclip: np.ndarray | torch.Tensor) -> torch.Tensor:
    arr = soundclip.detach().cpu().numpy() if isinstance(soundclip, torch.Tensor) else np.asarray(soundclip)
    features = sound_to_tensor.extract_enhanced_features(arr)
    return torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)


def _prepare_yamnet_input(soundclip: np.ndarray | torch.Tensor) -> torch.Tensor:
    arr = soundclip.detach().cpu().numpy() if isinstance(soundclip, torch.Tensor) else np.asarray(soundclip)

    if arr.ndim == 1:
        mel = librosa.feature.melspectrogram(
            y=arr.astype(np.float32),
            sr=16000,
            n_fft=400,
            hop_length=160,
            n_mels=64,
            power=2.0,
        )
        log_mel = np.log(mel + 1e-6).T
        if log_mel.shape[0] < 96:
            log_mel = np.pad(log_mel, ((0, 96 - log_mel.shape[0]), (0, 0)), mode="constant")
        else:
            log_mel = log_mel[:96]
        return torch.from_numpy(log_mel.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    tensor = torch.as_tensor(arr, dtype=torch.float32)
    if tensor.dim() == 2:
        if tensor.shape[-1] == 64:
            return tensor.unsqueeze(0).unsqueeze(0)
        if tensor.shape[0] == 64:
            return tensor.transpose(0, 1).unsqueeze(0).unsqueeze(0)
    if tensor.dim() == 3 and tensor.shape[-1] == 64:
        return tensor.unsqueeze(1)
    if tensor.dim() == 4:
        return tensor

    raise ValueError("yamnet input must be a waveform or a mel tensor with 64 mel bins.")


def predict(soundclip, model: str):
    model_name = model.lower().strip()

    if model_name == "yamnet":
        loaded_model = _load_yamnet_model()
        indata = _prepare_yamnet_input(soundclip)
    elif model_name == "rasnet":
        loaded_model = _load_rasnet_model()
        indata = _prepare_rasnet_input(soundclip)
    else:
        raise ValueError(f"Unsupported model '{model}'. Use 'yamnet' or 'rasnet'.")

    with torch.no_grad():
        return loaded_model(indata)