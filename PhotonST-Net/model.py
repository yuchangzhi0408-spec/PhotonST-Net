import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义卷积块
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# 定义下采样模块
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_block(in_channels, out_channels, padding=padding)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# 定义上采样模块
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = conv_block(in_channels, out_channels, padding=1)

    def forward(self, x, x_copy):
        x = self.up(x)

        diffY = x_copy.size()[2] - x.size()[2]
        diffX = x_copy.size()[3] - x.size()[3]
        x = F.pad(x, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2
        ])

        x = torch.cat([x_copy, x], dim=1)
        return self.conv(x)

# 定义 ConvLSTM 单元
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

    def forward(self, x, h_cur, c_cur):
        combined = torch.cat([x, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

# 集成 ConvLSTM 的 U-Net
class UNetWithDynamicOutputs(nn.Module):
    def __init__(self, n_channels):
        super(UNetWithDynamicOutputs, self).__init__()
        self.n_channels = n_channels

        padding = 1
        expansion = 2
        inplanes = 64
        chns = [inplanes, inplanes * expansion, inplanes * expansion ** 2, inplanes * expansion ** 3,
                inplanes * expansion ** 4]

        self.inc = conv_block(n_channels, chns[0], padding)
        self.down1 = DownSample(chns[0], chns[1], padding)
        self.down2 = DownSample(chns[1], chns[2], padding)
        self.down3 = DownSample(chns[2], chns[3], padding)
        self.down4 = DownSample(chns[3], chns[4], padding)

        # ConvLSTM网络
        self.convlstm = ConvLSTMCell(input_dim=chns[4], hidden_dim=chns[4], kernel_size=3)

        # 解码器部分
        self.up1 = UpSample(chns[-1], chns[-2])
        self.up2 = UpSample(chns[-2], chns[-3])
        self.up3 = UpSample(chns[-3], chns[-4])
        self.up4 = UpSample(chns[-4], chns[-5])

        # 添加目标数量分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(chns[-5], 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # 输出2个值，表示是单目标还是双目标的概率
            nn.Softmax(dim=1)
        )

        # 输出层
        self.outc1 = nn.Conv2d(chns[-5], 1, kernel_size=1)  # 第一个目标
        self.outc2 = nn.Conv2d(chns[-5], 1, kernel_size=1)  # 第二个目标（如果需要）
        self.mask1 = nn.Conv2d(chns[-5], 1, kernel_size=1)
        self.mask2 = nn.Conv2d(chns[-5], 1, kernel_size=1)

    def forward(self, x, h_cur=None, c_cur=None):
        # 编码部分
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)

        # ConvLSTM提取时序特征
        if h_cur is None or c_cur is None:
            h_cur = torch.zeros_like(e5)
            c_cur = torch.zeros_like(e5)

        h_next, c_next = self.convlstm(e5, h_cur, c_cur)

        # 解码部分
        x = self.up1(h_next, e4)
        x = self.up2(x, e3)
        x = self.up3(x, e2)
        x = self.up4(x, e1)

        # 分类预测
        class_pred = self.classifier(x)

        # 生成输出
        output1 = self.outc1(x)
        output2 = self.outc2(x)
        mask1 = torch.sigmoid(self.mask1(x))
        mask2 = torch.sigmoid(self.mask2(x))

        output1 = output1 * mask1
        output2 = output2 * mask2

        output1 = torch.tanh(output1)
        output2 = torch.tanh(output2)

        return output1, output2, h_next, c_next, class_pred
