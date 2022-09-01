import torch
import torch.nn as nn
import compressai
from models.gdn import GDN
from models.convlstm import ConvLSTM, ConvLSTMCell


class MvanalysisNet(nn.Module):
    def __init__(self, device):
        super(MvanalysisNet, self).__init__()
        self.device = device
        self.conv1 = self.conv_gdn(2, 128)
        self.conv2 = self.conv_gdn(128, 128)
        self.rnn = ConvLSTM(128, 128, 3, 2, True, True, False)
        self.conv3 = self.conv_gdn(128, 128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True)
        self.entropy_model = compressai.entropy_models.entropy_models.EntropyBottleneck(128).to(device)

    def forward(self, input, h_state=None):
        x = self.conv1(input)
        x = self.conv2(x)
        x, h = self.rnn(x, h_state)
        x = self.conv3(x)
        x = self.conv4(x)
        x, self.likelihood = self.entropy_model(x)
        return x, h

    def conv_gdn(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv2d(feat_in, feat_out, kernel_size=3, stride=2, padding=1, bias=True),
            GDN(feat_out, self.device)
        )


class MvsynthesisNet(nn.Module):
    def __init__(self, device):
        super(MvsynthesisNet, self).__init__()
        self.device = device
        self.conv1 = self.conv_gdn(128, 128)
        self.conv2 = self.conv_gdn(128, 128)
        self.rnn = ConvLSTM(128, 128, 3, 1, True, True, False)
        self.conv3 = self.conv_gdn(128, 128)
        self.conv4 = nn.ConvTranspose2d(128, 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)

    def forward(self, input, h_state=None):
        x = self.conv1(input)
        x = self.conv2(x)
        x, h = self.rnn(x, h_state)
        x = self.conv3(x)
        x = self.conv4(x)
        return x, h

    def conv_gdn(self, feat_in, feat_out):
        return nn.Sequential(
            nn.ConvTranspose2d(feat_in, feat_out, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            GDN(feat_out, self.device, inverse=True)
        )


if __name__ == '__main__':
    gpu_id = 5
    device = torch.device('cuda:{}'.format(gpu_id))
    input_frame = torch.rand(2, 7, 2, 256, 256).to(device)
    mv_encoder = MvanalysisNet(device).to(device)
    mv_decoder = MvsynthesisNet(device).to(device)
    code = mv_encoder(input_frame)
    res = mv_decoder(code)
    print(res.shape)