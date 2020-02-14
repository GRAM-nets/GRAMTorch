from __future__ import print_function
import argparse
import wandb
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--model', required=True, help='gramnet | gan')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--showSize', type=int, default=100, help='size of display batch')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--nk', type=int, default=100, help='size of the projected k vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ncf', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--clip_ratio', action='store_true', help='apply ratio clipping as suggested by one of the reviewer')
parser.add_argument('--eps_ratio', type=float, default=0.001, help='add eps to the diagonal before solving')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--gpu_id', type=int, default=0, help='default GPU ID to use')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netF', default='', help="path to netF (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--monitor_heuristic', action='store_true', help='monitor heuristic σ')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nowandb', action='store_true', help='disables wandb')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')

opt = parser.parse_args()
print(opt)
if not opt.nowandb:
    wandb.init(project="gramtorch", config=opt)

try:
    os.makedirs(opt.outf)
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
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nc=3
elif opt.dataset == 'lsun':
    classes = [ c + '_train' for c in opt.classes.split(',')]
    dataset = dset.LSUN(root=opt.dataroot, classes=classes,
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc=3
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nc=3

elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
        nc=1

elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
    nc=3

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device(f"cuda:{opt.gpu_id}" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
nk = int(opt.nk)
ngf = int(opt.ngf)
ncf = int(opt.ncf)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def save_img(x_data, x_gen, epoch):
    vutils.save_image(
        x_data[0:opt.showSize],
        f'{opt.outf}/{opt.dataset}-{opt.model}-data.png',
        normalize=True
    )
    vutils.save_image(
        x_gen[0:opt.showSize],
        f'{opt.outf}/{opt.dataset}-{opt.model}-samples-epoch={epoch:03d}.png',
        normalize=True
    )
    if not opt.nowandb:
        wandb.log({"samples" : [wandb.Image(i) for i in x_gen[0:opt.showSize]]}, commit=False)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Critic(nn.Module):
    def __init__(self, ngpu, nout):
        super(Critic, self).__init__()
        self.ngpu = ngpu
        self.nout = nout
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ncf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ncf) x 32 x 32
            nn.Conv2d(ncf, ncf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ncf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ncf*2) x 16 x 16
            nn.Conv2d(ncf * 2, ncf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ncf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ncf*4) x 8 x 8
            nn.Conv2d(ncf * 4, ncf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ncf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ncf*8) x 4 x 4
        )
        final = nn.Linear(ncf * 8 * 4 * 4, nout)
        if nout == 1:
            final = nn.Sequential(final, nn.Sigmoid())
        self.final = final

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            output = output.view(-1, ncf * 8 * 4 * 4)
            output = nn.parallel.data_parallel(self.final, output, range(self.ngpu))
        else:
            output = self.main(input)
            output = output.view(-1, ncf * 8 * 4 * 4)
            output = self.final(output)
        if self.nout == 1:
            output = output.view(-1, 1).squeeze(1)
        return output


class GAN:
    def __init__(self, ngpu):
        netG = Generator(ngpu).to(device)
        netG.apply(weights_init)
        if opt.netG != '':
            netG.load_state_dict(torch.load(opt.netG))
        print(netG)
        
        netD = Critic(ngpu, 1).to(device)
        netD.apply(weights_init)
        if opt.netD != '':
            netD.load_state_dict(torch.load(opt.netD))
        print(netD)

        self.netG = netG
        self.netD = netD

    def train(self):
        netG = self.netG
        netD = self.netD

        criterion = nn.BCELoss()

        fixed_noise = torch.randn(opt.showSize, nz, 1, 1, device=device)
        real_label = 1
        fake_label = 0

        # setup optimizer
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        for epoch in range(opt.n_epochs):
            for i, data in enumerate(dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
                netD.zero_grad()
                real_cpu = data[0].to(device)
                batch_size = real_cpu.size(0)
                label = torch.full((batch_size,), real_label, device=device)

                output = netD(real_cpu)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                # train with fake
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake = netG(noise)
                label.fill_(fake_label)
                output = netD(fake.detach())
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                output = netD(fake)
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                    % (epoch, opt.n_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                if i % 100 == 0:
                    netG.train(False)
                    save_img(real_cpu, netG(fixed_noise).detach(), epoch)
                    netG.train(True)
                if not opt.nowandb:
                    wandb.log({
                        "lossG" : errG, "lossF" : errD, 
                        "D(x)" : D_G_z1, "D(G(z))" : D_G_z2,
                    })

            # do checkpointing
            torch.save(netG.state_dict(), f'{opt.outf}/{opt.model}-netG-epoch={epoch}.pth')
            torch.save(netD.state_dict(), f'{opt.outf}/{opt.model}-netD-epoch={epoch}.pth')


### MMD utilities

def euclidsq(x, y):
    return torch.pow(torch.cdist(x, y), 2)

def gaussian_gramian(esq, σ):
    return torch.exp(torch.div(-esq, 2 * σ**2))

def prepare(x_de, x_nu):
    return euclidsq(x_de, x_de), euclidsq(x_de, x_nu), euclidsq(x_nu, x_nu)

USE_SOLVE = True

def kmm_ratios(Kdede, Kdenu, λ):
    n_de, n_nu = Kdenu.shape
    if λ > 0:
        A = Kdede + λ * torch.eye(n_de).to(device)
    else:
        A = Kdede
    # Equivalent implement based on 1) solver and 2) matrix inversion
    if USE_SOLVE:
        B = torch.sum(Kdenu, 1, keepdim=True)
        return (n_de / n_nu) * torch.solve(B, A).solution
    else:
        B = Kdenu
        return torch.matmul(torch.matmul(torch.inverse(A), B), torch.ones(n_nu, 1).to(device))

def mmdsq_of(Kdede, Kdenu, Knunu):
    return torch.mean(Kdede) - 2 * torch.mean(Kdenu) + torch.mean(Knunu)

def estimate_ratio_compute_mmd(x_de, x_nu, σs):
    dsq_dede, dsq_denu, dsq_nunu = prepare(x_de, x_nu)
    if opt.monitor_heuristic:
        sigma = torch.sqrt(
            torch.median(torch.cat([dsq_dede.squeeze(), dsq_denu.squeeze(), dsq_nunu.squeeze()], 1))
        )
        wandb.log({"heuristic_sigma" : sigma})
        #print("heuristic sigma: ", sigma)
    is_first = True
    ratio = None
    mmdsq = None
    for σ in σs:
        Kdede = gaussian_gramian(dsq_dede, σ)
        Kdenu = gaussian_gramian(dsq_denu, σ)
        Knunu = gaussian_gramian(dsq_nunu, σ)
        if is_first:
            ratio = kmm_ratios(Kdede, Kdenu, opt.eps_ratio)
            mmdsq = mmdsq_of(Kdede, Kdenu, Knunu)
            is_first = False
        else:
            ratio += kmm_ratios(Kdede, Kdenu, opt.eps_ratio)
            mmdsq += mmdsq_of(Kdede, Kdenu, Knunu)
    
    raito = ratio / len(σs)
    ratio = torch.relu(ratio) if opt.clip_ratio else ratio
    mmd = torch.sqrt(torch.relu(mmdsq))
    
    return ratio, mmd

def extract_grad(m):
    gs = []
    for p in m.parameters():
        gs.append(p.grad.clone())
    return gs

def assign_grad(m, gs):
    for p, g in zip(m.parameters(), gs):
        p.grad = g

def sim_step(optimizer1, optimizer2, m1, m2, loss1, loss2):
    loss1.backward(retain_graph=True)
    gs1 = extract_grad(m1)
    m1.zero_grad()
    m2.zero_grad()
    loss2.backward()
    optimizer2.step()
    assign_grad(m1, gs1)
    optimizer1.step()

class GRAMnet:
    def __init__(self, ngpu):
        netG = Generator(ngpu).to(device)
        netG.apply(weights_init)
        if opt.netG != '':
            netG.load_state_dict(torch.load(opt.netG))
        print(netG)
        
        netF = Critic(ngpu, nk).to(device)
        netF.apply(weights_init)
        if opt.netF != '':
            netF.load_state_dict(torch.load(opt.netF))
        print(netF)

        self.netG = netG
        self.netF = netF

        if opt.dataset == "mnist":
            sigma_list = np.sqrt([10, 50, 100, 500])
        elif opt.dataset = "cifar10":
            sigma_list = np.sqrt([1, 2, 4, 8, 16])
        else:
            sigma_list = [1, 5, 10, 50, 100]
        self.sigma_list = sigma_list

    def train(self):

        netG = self.netG
        netF = self.netF

        fixed_noise = torch.randn(opt.showSize, nz, 1, 1, device=device)

        # setup optimizer
        optimizerF = optim.Adam(netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        for epoch in range(opt.n_epochs):
            for i, data in enumerate(dataloader, 0):
                x_data = data[0].to(device)
                batch_size = x_data.size(0)

                netF.zero_grad()
                netG.zero_grad()
                # Generate samples
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                x_gen = netG(noise)
                # Project to low-dimensional space
                fx_data = netF(x_data)
                fx_gen = netF(x_gen)
                # Compute ratio and mmd
                ratio, mmd = estimate_ratio_compute_mmd(fx_gen, fx_data, self.sigma_list)
                pearson_divergence = torch.mean(torch.pow(ratio - 1, 2))
                lossG = mmd
                lossF = -pearson_divergence
                # Add positivity regularizer if not clipping
                if not opt.clip_ratio:  
                    lossF -= torch.mean(ratio)
                # Update G and F simultaneously
                sim_step(optimizerG, optimizerF, netG, netF, lossG, lossF)

                print('[%d/%d][%d/%d] Loss_F: %.4f Loss_G: %.4f'
                    % (epoch, opt.n_epochs, i, len(dataloader), lossF.item(), lossG.item()))
                if i % 100 == 0:
                    netG.train(False)
                    save_img(x_data, netG(fixed_noise).detach(), epoch)
                    netG.train(True)
                if not opt.nowandb:
                    wandb.log({
                        "lossG" : lossG, "lossF" : lossF, "mmd" : mmd,
                        "pearson_divergence" : pearson_divergence,
                    })

            # do checkpointing
            torch.save(netG.state_dict(), f'{opt.outf}/{opt.model}-netG-epoch={epoch}.pth')
            torch.save(netF.state_dict(), f'{opt.outf}/{opt.model}-netF-epoch={epoch}.pth')


if opt.model == "gan":
    model = GAN(ngpu)
elif opt.model == "gramnet":
    model = GRAMnet(ngpu)

model.train()
