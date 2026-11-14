"""开发图研究用的一维分类器。

所有模型接收 ``(批量, 1, 光谱长度)``，输出未经过 softmax 的类别 logits。
模型不绑定 CPU 或 GPU，也不加载预训练权重。
"""
from __future__ import annotations

import torch
from torch import nn


class _SpectrumClassifier(nn.Module):
    """统一输入约定，避免误将光谱长度当作通道数。"""

    def __init__(self, n_classes: int, input_length: int) -> None:
        super().__init__()
        if not isinstance(n_classes, int) or n_classes < 2:
            raise ValueError('n_classes 必须是至少为 2 的整数。')
        if not isinstance(input_length, int) or input_length < 16:
            raise ValueError('input_length 必须是至少为 16 的整数。')
        self.n_classes = n_classes
        self.input_length = input_length

    def _check_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[2] != self.input_length:
            raise ValueError(
                f'期望输入形状为 (N, 1, {self.input_length})，实际为 {tuple(x.shape)}。'
            )


class PlainCNN(_SpectrumClassifier):
    """与主项目 SpectralCNN 保持相同的卷积层、分类头和权重名称。"""

    def __init__(self, n_classes: int, input_length: int = 1024) -> None:
        super().__init__(n_classes, input_length)
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        # 保留 16 个光谱位置区段，避免全局平均池化抹去绝对峰位置。
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        return self.classifier(self.features(x))


class SpectralMLP(_SpectrumClassifier):
    """将完整光谱直接送入全连接层，作为卷积结构的对照。"""

    def __init__(self, n_classes: int, input_length: int = 1024) -> None:
        super().__init__(n_classes, input_length)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_length, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        # 训练时 BatchNorm 需要每批至少 2 条；推理阶段允许批量为 1。
        return self.classifier(x)


class ResidualBlock1D(nn.Module):
    """两层 3 点卷积；通道数或步幅改变时使用投影捷径。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class SmallResNet1D(_SpectrumClassifier):
    """4 个通道阶段，每阶段 2 个残差块的轻量一维网络。"""

    def __init__(self, n_classes: int, input_length: int = 1024) -> None:
        super().__init__(n_classes, input_length)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.stages = nn.Sequential(
            self._make_stage(32, 32, stride=1),
            self._make_stage(32, 64, stride=2),
            self._make_stage(64, 128, stride=2),
            self._make_stage(128, 256, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, n_classes),
        )

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            ResidualBlock1D(in_channels, out_channels, stride=stride),
            ResidualBlock1D(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        return self.classifier(self.pool(self.stages(self.stem(x))))


def build_model(
    name: str, n_classes: int, input_length: int = 1024
) -> nn.Module:
    """按固定名称创建模型；设备转移由调用方通过 ``model.to(device)`` 完成。"""
    constructors = {
        'cnn': PlainCNN,
        'mlp': SpectralMLP,
        'resnet': SmallResNet1D,
    }
    if not isinstance(name, str) or name.lower().strip() not in constructors:
        raise ValueError(f'未知模型 {name!r}，可选名称为 cnn、mlp、resnet。')
    return constructors[name.lower().strip()](n_classes, input_length)
