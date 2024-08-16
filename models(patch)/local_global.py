import torch.nn as nn
import torch
from models.CausalDilate import SimpleWaveNet
# 对于输入的序列做平均池化
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    def forward(self, x):
        # x shape: batch,seq_len,channels
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1) # 提取第一个序列
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)   # 提取最后一个序列
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

# 分解为残差和移动平均
class series_decomp(nn.Module):
    """
    Series decomposition block, seasonal 和 trend 的一次分解
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

# 扩展的"series_decomp", 多个池化卷积核，进行提取之后求平均值
class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size] # 创建多个卷积核，并且装在列表中

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg)
            sea = x - moving_avg
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean

# 前馈神经网络
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)
        self.initialize_weight(self.layer1)
        self.initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)

# 合并以上模块
class MIC(nn.Module):
    """
    MIC layer to extract local and global features
    """
    def __init__(self, feature_size=512, n_heads=8, dropout=0.05, decomp_kernel=[32], conv_kernel=[24], isometric_kernel=[18, 6], device='cuda'):
        super(MIC, self).__init__()
        self.src_mask = None
        self.conv_kernel = conv_kernel
        self.isometric_kernel = isometric_kernel
        self.device = device
        # isometric convolution
        self.isometric_conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                                   kernel_size=i,padding=0,stride=1)
                                        for i in isometric_kernel])
        # causal-dilate convolution
        dilations = [1, 2, 4, 8, 16]
        self.Causal_DilateCov = SimpleWaveNet(in_channels=feature_size, out_channels=feature_size, kernel_size=2, dilations=dilations)
        self.Causal_DilateCov.to("cuda")
        # downsampling convolution: padding=i//2, stride=i
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                             kernel_size=i,padding=i//2,stride=i)
                                  for i in conv_kernel])
        # upsampling convolution
        self.conv_trans = nn.ModuleList([nn.ConvTranspose1d(in_channels=feature_size, out_channels=feature_size,
                                                            kernel_size=i,padding=0,stride=i)
                                        for i in conv_kernel])
        self.decomp = nn.ModuleList([series_decomp(k) for k in decomp_kernel])
        self.merge = torch.nn.Conv2d(in_channels=feature_size, out_channels=feature_size, kernel_size=(len(self.conv_kernel), 1))
        self.fnn = FeedForwardNetwork(feature_size, feature_size*4, dropout)
        self.fnn_norm = torch.nn.LayerNorm(feature_size)
        self.norm = torch.nn.LayerNorm(feature_size)
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(0.05)

    # 对于输入进行处理
    def conv_trans_conv(self, input, conv1d, conv1d_trans, isometric):
        batch, seq_len, channel = input.shape
        x = input.permute(0, 2, 1)

        # downsampling convolution
        x1 = self.drop(self.act(conv1d(x)))
        x = x1

        # isometric convolution 
        # zeros = torch.zeros((x.shape[0], x.shape[1], x.shape[2]-1), device=self.device)
        # x = torch.cat((zeros, x), dim=-1)
        # x = self.drop(self.act(isometric(x)))
        # 需要修改的就是这部分内容
        # print(x.shape)
        # print(x)
        # 因果膨胀卷积
        

        x = self.drop(self.act(self.Causal_DilateCov(x)))
        
        x = self.norm((x+x1).permute(0, 2, 1)).permute(0, 2, 1)

        # upsampling convolution
        x = self.drop(self.act(conv1d_trans(x)))
        x = x[:, :, :seq_len]   # truncate

        x = self.norm(x.permute(0, 2, 1) + input)
        return x

    # 提取信息，然后再合并
    def forward(self, src):
        # multi-scale
        multi = []  
        for i in range(len(self.conv_kernel)):
            # 分解
            src_out, trend1 = self.decomp[i](src)
            # 卷积操作
            src_out = self.conv_trans_conv(src_out, self.conv[i], self.conv_trans[i], self.isometric_conv[i])
            multi.append(src_out)  
        # merge
        mg = torch.tensor([], device = self.device)
        for i in range(len(self.conv_kernel)):
            mg = torch.cat((mg, multi[i].unsqueeze(1)), dim=1)
        mg = self.merge(mg.permute(0,3,1,2)).squeeze(-2).permute(0,2,1)
        
        return self.fnn_norm(mg + self.fnn(mg))

# 多个MIC再合并，接受输入的信息，最后投影
class Seasonal_Prediction(nn.Module):
    def __init__(self, embedding_size=512, n_heads=8, dropout=0.05, d_layers=1, decomp_kernel=[32], c_out=1,
                conv_kernel=[2, 4], isometric_kernel=[18, 6], device='cuda'):
        super(Seasonal_Prediction, self).__init__()
        self.mic = nn.ModuleList([MIC(feature_size=embedding_size, n_heads=n_heads,
                                                   decomp_kernel=decomp_kernel,conv_kernel=conv_kernel, isometric_kernel=isometric_kernel, device=device)
                                      for i in range(d_layers)])
        self.projection = nn.Linear(embedding_size, c_out)
    def forward(self, dec):
        for mic_layer in self.mic:
            dec = mic_layer(dec)
        return self.projection(dec)

