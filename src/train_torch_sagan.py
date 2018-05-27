import argparse
import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='folder', help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=5000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=-1, help='number of GPUs to use')
parser.add_argument('--netG', default='./results/netG_epoch.pth', help="path to netG (to continue training)")
parser.add_argument('--netD', default='./results/netD_epoch.pth', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./results', help='folder to output images and model checkpoints')
parser.add_argument('--n_critic', type=int, default=1, help='number of training steps for discriminator per iter')
parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    Path(opt.outf).mkdir(parents=True, exist_ok=True)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.ColorJitter(brightness=0.4,
                                                          contrast=0.4,
                                                          saturation=0.4),
                                   # transforms.CenterCrop(opt.imageSize),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())

assert dataset
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=opt.batchSize,
    shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3

# Loss weight for gradient penalty
lambda_gp = 10

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def init_linear(linear):
    init.xavier_uniform_(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class SpectralNorm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()
        sigma = u @ weight_mat @ v
        weight_sn = weight / sigma
        # weight_sn = weight_sn.view(*size)

        return weight_sn, u

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)


def spectral_norm(module, name='weight'):
    SpectralNorm.apply(module, name)

    return module


class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,
                 n_class=None,
                 padding=1, post=True, resize=True,
                 normalize=True, self_attention=False):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                         padding=padding, bias=False)
        init_conv(conv)
        self.conv = spectral_norm(conv)

        self.resize = resize
        self.post = post
        if self.post:
            self.bn = nn.BatchNorm2d(out_channel, affine=False)

            self.embed = nn.Embedding(n_class, out_channel * 2)
            self.embed.weight.data[:, :out_channel] = 1
            self.embed.weight.data[:, out_channel:] = 0

        self.attention = self_attention
        if self_attention:
            self.query = nn.Conv1d(out_channel, out_channel // 8, 1)
            self.key = nn.Conv1d(out_channel, out_channel // 8, 1)
            self.value = nn.Conv1d(out_channel, out_channel, 1)
            self.gamma = nn.Parameter(torch.tensor(0.0))

            init_conv(self.query)
            init_conv(self.key)
            init_conv(self.value)

    def forward(self, input, class_id=None):
        out = input
        if self.resize:
            out = self.upsample(input)
        out = self.conv(out)
        if self.post:
            out = self.bn(out)
            embed = self.embed(class_id)
            gamma, beta = embed.chunk(2, 1)
            #print(out.shape, gamma.shape, beta.shape)
            gamma = gamma.unsqueeze(2).unsqueeze(3)
            beta = beta.unsqueeze(2).unsqueeze(3)
            out = gamma * out + beta
            out = activation(out)

        if self.attention:
            shape = out.shape
            flatten = out.view(shape[0], shape[1], -1)
            query = self.query(flatten).permute(0, 2, 1)
            key = self.key(flatten)
            value = self.value(flatten)
            #print(key.shape, value.shape)
            query_key = torch.bmm(query, key)
            attn = F.softmax(query_key, 1)
            attn = torch.bmm(value, attn)
            attn = attn.view(*shape)
            #print(out.shape, attn.shape)
            out = self.gamma * attn + out

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,
                 stride=1, padding=1, bn=False,
                 self_attention=False):
        super().__init__()

        conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                         stride, padding, bias=False)
        init_conv(conv)
        self.conv = spectral_norm(conv)
        self.use_bn = bn
        if bn:
            self.bn = nn.BatchNorm2d(out_channel, affine=True)

        self.attention = self_attention
        if self_attention:
            query = nn.Conv1d(out_channel, out_channel // 8, 1)
            key = nn.Conv1d(out_channel, out_channel // 8, 1)
            value = nn.Conv1d(out_channel, out_channel, 1)
            self.gamma = nn.Parameter(torch.tensor(0.0))

            init_conv(query)
            init_conv(key)
            init_conv(value)

            self.query = spectral_norm(query)
            self.key = spectral_norm(key)
            self.value = spectral_norm(value)

    def forward(self, input):
        out = self.conv(input)
        if self.use_bn:
            out = self.bn(out)
        out = F.leaky_relu(out, negative_slope=0.2)

        if self.attention:
            shape = out.shape
            flatten = out.view(shape[0], shape[1], -1)
            query = self.query(flatten).permute(0, 2, 1)
            key = self.key(flatten)
            value = self.value(flatten)
            query_key = torch.bmm(query, key)
            attn = F.softmax(query_key, 1)
            attn = torch.bmm(value, attn)
            attn = attn.view(*shape)
            out = self.gamma * attn + out

        return out


class Generator(nn.Module):
    def __init__(self, code_dim=100, n_class=None):
        super().__init__()
        self.c4, self.c8, self.c16, self.c32, self.c64 = 512, 256, 128, 64, 32
        self.s4 = 4

        self.lin_code = nn.Linear(code_dim, self.s4*self.s4*self.c4)
        self.conv1 = UpsampleConvBlock(self.c4, self.c8, 3, post=False)
        self.conv2 = UpsampleConvBlock(self.c8, self.c8, 3,
                                       self_attention=True, post=False)
        self.conv3 = UpsampleConvBlock(self.c8, self.c16, 3, post=False)
        self.conv4 = UpsampleConvBlock(self.c16, self.c32, 3, post=False)
        self.conv5 = UpsampleConvBlock(self.c32, 3, 3, 1,
                                       resize=False, post=False)
        init_linear(self.lin_code)

    def forward(self, input):
        out = F.relu(self.lin_code(input))
        out = out.view(-1, self.c4, self.s4, self.s4)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        return F.tanh(out)


netG = Generator()
if opt.cuda:
    netG.cuda()

# netG.apply(weights_init)
if Path(opt.netG).exists():
    netG.load_state_dict(torch.load(opt.netG))
# print(netG)


class Discriminator(nn.Module):
    def __init__(self, n_class=18):
        super().__init__()
        self.c2, self.c4, self.c8, self.c16 = 64, 128, 256, 512

        self.conv = nn.Sequential(ConvBlock(3, self.c2, 3, 2),
                                  ConvBlock(self.c2, self.c4, 3, 2),
                                  ConvBlock(self.c4, self.c8, 3, 2),
                                  ConvBlock(self.c8, self.c16, 3, 2),
                                  ConvBlock(self.c16, self.c16, 3,
                                            self_attention=True))
        linear = nn.Linear(512, 1)
        init_linear(linear)
        self.linear = spectral_norm(linear)

        embed = nn.Embedding(n_class, 512)
        embed.weight.data.uniform_(-0.1, 0.1)
        self.embed = spectral_norm(embed)

    def forward(self, input):
        out = self.conv(input)
        out = out.view(out.size(0), out.size(1), -1)
        out = out.sum(2)
        out_linear = self.linear(out).squeeze()
        return out_linear


netD = Discriminator()
if opt.cuda:
    netD.cuda()

# netD.apply(weights_init)
if Path(opt.netD).exists():
    netD.load_state_dict(torch.load(opt.netD))
# print(netD)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def main():
    # fixed_noise = torch.FloatTensor(opt.batchSize, nz).uniform_(-1, 1)

    # setup optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0, 0.9))
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0, 0.9))

    requires_grad(netG, False)
    requires_grad(netD, True)

    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            real_images = data[0]
            optimizerD.zero_grad()

            # train with fake
            noise = torch.randn(real_images.shape[0], nz)
            fake = netG(noise).detach()

            real_validity = netD(real_images)
            fake_validity = netD(fake)

            loss_D = F.relu(1 + fake_validity).mean()
            loss_D += F.relu(1 - real_validity).mean()
            disc_loss_val = loss_D.detach().item()

            loss_D.backward()
            optimizerD.step()

            if i % opt.n_critic == 0:
                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                optimizerG.zero_grad()
                requires_grad(netG, True)
                requires_grad(netD, False)

                fake = netG(torch.randn(real_images.shape[0], nz))
                # Adversarial loss
                loss_G = -netD(fake).mean()
                gen_loss_val = loss_G.detach().item()
                loss_G.backward()
                optimizerG.step()

                requires_grad(netG, False)
                requires_grad(netD, True)

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                      % (epoch, opt.niter, i, len(dataloader),
                         disc_loss_val, gen_loss_val))

            if i % 50 == 0:
                vutils.save_image(real_images,
                        '%s/real_samples.png' % opt.outf,
                        normalize=True)
                fake = netG(torch.randn(opt.batchSize, nz))
                vutils.save_image(fake.detach(),
                        '%s/fake_samples_epoch.png' % (opt.outf),
                        normalize=True)

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch.pth' % (opt.outf))
        torch.save(netD.state_dict(), '%s/netD_epoch.pth' % (opt.outf))


if __name__ == '__main__':
    main()
