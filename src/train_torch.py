import argparse
import os
from pathlib import Path
import random
import torch
import torch.nn as nn
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
parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=-1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./results', help='folder to output images and model checkpoints')
parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
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


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c4, self.c8, self.c16, self.c32, self.c64 = 512, 256, 128, 64, 32
        self.s4 = 4

        self.g_fc1 = nn.Linear(100, self.s4*self.s4*self.c4)
        self.g_h1 = nn.ConvTranspose2d(self.c4, self.c8, 5, 2, padding=2)
        self.g_h2 = nn.ConvTranspose2d(self.c8, self.c16, 5, 2, padding=1)
        self.g_h3 = nn.ConvTranspose2d(self.c16, self.c32, 5, 2, padding=1)
        self.g_h4 = nn.ConvTranspose2d(self.c32, 3, 4, 2, padding=0)

        self.bn0 = nn.BatchNorm2d(self.c4)
        self.bn1 = nn.BatchNorm2d(self.c8)
        self.bn2 = nn.BatchNorm2d(self.c16)
        self.bn3 = nn.BatchNorm2d(self.c32)
        self.bn4 = nn.BatchNorm2d(3)

    def forward(self, input):
        h_z = self.g_fc1(input)
        h0 = h_z.view(-1, self.c4, self.s4, self.s4)
        h0 = F.relu(self.bn0(h0))

        h1 = self.g_h1(h0)
        h1 = F.relu(self.bn1(h1))

        h2 = self.g_h2(h1)
        h2 = F.relu(self.bn2(h2))

        h3 = self.g_h3(h2)
        h3 = F.relu(self.bn3(h3))

        h4 = self.g_h4(h3)

        h4 = F.tanh(self.bn4(h4))
        return h4


netG = Generator()
if opt.cuda:
    netG.cuda()

netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
# print(netG)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c2, self.c4, self.c8, self.c16 = 64, 128, 256, 512

        self.d_h0_conv = nn.Conv2d(3, self.c2, 5, 2, padding=0)
        self.d_h1_conv = nn.Conv2d(self.c2, self.c4, 5, 2, padding=0)
        self.d_h2_conv = nn.Conv2d(self.c4, self.c8, 5, 2, padding=0)
        self.d_h3_conv = nn.Conv2d(self.c8, self.c16, 5, 2, padding=0)
        self.d_fc = nn.Linear(self.c16, 1)

        self.bn0 = nn.BatchNorm2d(self.c2)
        self.bn1 = nn.BatchNorm2d(self.c4)
        self.bn2 = nn.BatchNorm2d(self.c8)
        self.bn3 = nn.BatchNorm2d(self.c16)

    def forward(self, input):
        n_batch = input.shape[0]

        h0 = self.d_h0_conv(input)  # 32
        h0 = F.leaky_relu(self.bn0(h0))

        h1 = self.d_h1_conv(h0)  # 16
        h1 = F.leaky_relu(self.bn1(h1))

        h2 = self.d_h2_conv(h1)  # 8
        h2 = F.leaky_relu(self.bn2(h2))

        h3 = self.d_h3_conv(h2)  # 4
        h3 = F.leaky_relu(self.bn3(h3))

        h3_flatten = h3.view(n_batch, -1)
        # out = F.sigmoid(self.d_fc(h3_flatten).squeeze(1))
        return self.d_fc(h3_flatten).squeeze(1)


netD = Discriminator()
if opt.cuda:
    netD.cuda()

netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
# print(netD)


def main():
    # fixed_noise = torch.randn(opt.batchSize, nz)  # .clamp_(-1., 1.).detach()
    fixed_noise = torch.FloatTensor(opt.batchSize, nz).uniform_(-1, 1)

    # setup optimizer
    # optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    # optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.RMSprop(netG.parameters(), lr=5e-5)
    optimizerD = optim.RMSprop(netD.parameters(), lr=5e-5)

    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            real_images = data[0]
            optimizerD.zero_grad()

            # train with fake
            # noise = torch.randn(opt.batchSize, nz)  # .clamp_(-1., 1.).detach()
            noise = torch.FloatTensor(opt.batchSize, nz).uniform_(-1, 1)
            fake = netG(noise).detach()
            # Adversarial loss
            # err1 = -torch.mean(netD(real_images))
            # err2 = torch.mean(netD(fake))
            # loss_D = err1 + err2
            # print(err2.item(), err1.item())
            loss_D = -torch.mean(netD(real_images)) + torch.mean(netD(fake))
            loss_D.backward()
            optimizerD.step()

            for p in netD.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)

            if i % opt.n_critic == 0:
                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                optimizerG.zero_grad()
                fake = netG(fixed_noise)
                # Adversarial loss
                loss_G = -torch.mean(netD(fake))
                loss_G.backward()
                optimizerG.step()

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                      % (epoch, opt.niter, i, len(dataloader),
                         loss_D.item(), loss_G.item()))

            if i % 500 == 0:
                vutils.save_image(real_images,
                        '%s/real_samples.png' % opt.outf,
                        normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                        '%s/fake_samples_epoch.png' % (opt.outf),
                        normalize=True)

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch.pth' % (opt.outf))
        torch.save(netD.state_dict(), '%s/netD_epoch.pth' % (opt.outf))


if __name__ == '__main__':
    main()
