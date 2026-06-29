import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class SPADE(nn.Module):
    """Компонент модуляции слоев на основе маски разметки (NVIDIA GauGAN принцип)"""
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        # Изменяем разрешение маски под текущий шаг архитектуры генератора
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return normalized * (1 + gamma) + beta

class SPADEResnetBlock(nn.Module):
    """Резидуальный блок, управляемый геометрическим условием дефекта"""
    def __init__(self, fin, fout, label_nc):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
            self.norm_s = SPADE(fin, label_nc)
            
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)

    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)
        dx = self.conv_0(F.leaky_relu(self.norm_0(x, seg), 0.2))
        dx = self.conv_1(F.leaky_relu(self.norm_1(dx, seg), 0.2))
        return x_s + dx

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            return self.conv_s(self.norm_s(x, seg))
        return x

class GeneratorSPADE(nn.Module):
    def __init__(self, in_channels=3, num_classes=6, latent_dim=256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        
        # Пошаговое развертывание разрешения: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
        self.head_0 = SPADEResnetBlock(512, 512, num_classes)
        self.up_1 = nn.Upsample(scale_factor=2)
        self.head_1 = SPADEResnetBlock(512, 512, num_classes)
        self.up_2 = nn.Upsample(scale_factor=2)
        self.head_2 = SPADEResnetBlock(512, 256, num_classes)
        self.up_3 = nn.Upsample(scale_factor=2)
        self.head_3 = SPADEResnetBlock(256, 128, num_classes)
        self.up_4 = nn.Upsample(scale_factor=2)
        self.head_4 = SPADEResnetBlock(128, 64, num_classes)
        self.up_5 = nn.Upsample(scale_factor=2)
        self.head_5 = SPADEResnetBlock(64, 32, num_classes)
        
        self.conv_img = nn.Conv2d(32, in_channels, kernel_size=3, padding=1)

    def forward(self, z, cond):
        x = self.fc(z).view(-1, 512, 4, 4)
        
        x = self.head_0(x, cond)
        x = self.up_1(x)
        x = self.head_1(x, cond)
        x = self.up_2(x)
        x = self.head_2(x, cond)
        x = self.up_3(x)
        x = self.head_3(x, cond)
        x = self.up_4(x)
        x = self.head_4(x, cond)
        x = self.up_5(x)
        x = self.head_5(x, cond)
        
        x = self.conv_img(F.leaky_relu(x, 0.2))
        return torch.tanh(x)

class DiscriminatorSN(nn.Module):
    """Дискриминатор PatchGAN, возвращающий промежуточные признаки для Feature Matching Loss"""
    def __init__(self, in_channels=3, num_classes=6):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels + num_classes, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.final_layer = spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))

    def forward(self, img, cond):
        x = torch.cat([img, cond], dim=1)
        
        feats = []
        x = self.layer1(x); feats.append(x)
        x = self.layer2(x); feats.append(x)
        x = self.layer3(x); feats.append(x)
        x = self.layer4(x); feats.append(x)
        
        logits = self.final_layer(x)
        return logits, feats
