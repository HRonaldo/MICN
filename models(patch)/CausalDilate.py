import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalDilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalDilatedConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=(kernel_size - 1) * dilation, dilation=dilation)
    def forward(self, x):
        # 进行因果卷积，去掉填充的未来信息部分
        x = self.conv(x)
        return x[:, :, :-self.conv.padding[0]]

class WaveNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(WaveNetBlock, self).__init__()
        self.causal_conv = CausalDilatedConv1d(in_channels, out_channels, kernel_size, dilation)
        self.residual_conv = nn.Conv1d(out_channels, in_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(out_channels, in_channels, kernel_size=1)

    def forward(self, x):
        out = torch.tanh(self.causal_conv(x))
        residual = self.residual_conv(out)
        skip = self.skip_conv(out)
        return residual + x, skip

class SimpleWaveNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilations):
        super(SimpleWaveNet, self).__init__()
        self.blocks = nn.ModuleList([
            WaveNetBlock(in_channels, out_channels, kernel_size, d) for d in dilations
        ])
        self.final_conv = nn.Conv1d(len(dilations) * in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        skip_connections = []
        for block in self.blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        x = torch.cat(skip_connections, dim=1)
        x = self.final_conv(x)
        return x
    
# # # 定义简单的WaveNet模型
# dilations = [1, 2, 4, 8, 16]
# model = SimpleWaveNet(in_channels=1, out_channels=1, kernel_size=2, dilations=dilations)

# # 模拟输入序列 (batch_size=1, length=20, channels=1)
# x = torch.randn(1, 9, 1)
# # 执行前向传播
# output = model(x)
# print("输入形状:", x.shape)
# print("输出形状:", output.shape)
