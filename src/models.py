"""保留峰位置的一维卷积网络，约30万参数，适合单卡课程实验。"""
import torch
from torch import nn


class SpectralCNN(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv1d(1,32,9,stride=2,padding=4),nn.BatchNorm1d(32),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(32,64,7,padding=3),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(64,128,5,padding=2),nn.BatchNorm1d(128),nn.ReLU(),nn.AdaptiveAvgPool1d(16),
        )
        self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(128*16,128),nn.ReLU(),nn.Dropout(.25),nn.Linear(128,n_classes))

    def forward(self,x):
        return self.classifier(self.features(x))


def augment_spectra(x):
    # 只扰动训练批次，不保存为新独立样品；不跨样品混合标签。
    x=x*(.9+.2*torch.rand((len(x),1,1),device=x.device))
    x=x+.02*torch.randn_like(x)
    # 同批次最多平移2个采样点，边缘复制，避免循环卷绕伪峰。
    shift=int(torch.randint(-2,3,(1,),device=x.device).item())
    if shift>0:
        x=torch.cat([x[:,:,:1].expand(-1,-1,shift),x[:,:,:-shift]],dim=2)
    elif shift<0:
        shift=-shift
        x=torch.cat([x[:,:,shift:],x[:,:,-1:].expand(-1,-1,shift)],dim=2)
    return x
