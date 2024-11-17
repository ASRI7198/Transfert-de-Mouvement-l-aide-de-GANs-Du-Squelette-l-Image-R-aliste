import numpy as np
import cv2
import os
import pickle
import sys
import math

import matplotlib.pyplot as plt

from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenVanillaNN import *


class Discriminator(nn.Module):
    def __init__(self, ngpu=0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        # Input: (batch_size, 3, 64, 64)
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),  # (3,64,64)==>(3,32,32)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # (64,32,32)==>(128,16,16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # (128,16,16)==>(256,8,8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # (256,8,8)==>(512,4,4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),  # (512,4,4)==>(1,1,1)
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)


class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """

    def __init__(self, videoSke, loadFromFile=False):
        self.netG = GenNNSkeToImage()
        self.netD = Discriminator()
        self.real_label = 1.
        self.fake_label = 0.
        self.filename = 'C:/s`Users/DELL/Downloads/tp_dance_start/dance_start/data/DanceGenGAN.pth'
        tgt_transform = transforms.Compose(
            [transforms.Resize((64, 64)),
             # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             transforms.CenterCrop(64),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: Load=", self.filename, "   Current Working Directory=", os.getcwd())
            self.netG = torch.load(self.filename)

    def train(self, n_epochs=20):
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.netG = self.netG.to(device)
        self.netD = self.netD.to(device)

        criterion = nn.BCELoss()
        criterion2 = nn.MSELoss()

        optimD = optim.Adam(self.netD.parameters(), lr=0.001, betas=(0.5, 0.999))
        optimG = optim.Adam(self.netG.parameters(), lr=0.001, betas=(0.5, 0.999))

        for epoch in range(n_epochs):
            G_losses = 0
            D_losses = 0
            for i, data in enumerate(self.dataloader, 0):

                images_real = data[1].to(device)  # images
                print("images_real : ", images_real.shape)
                ske = data[0].to(device)
                batch_size = images_real.size(0)

                images_real_bruit = images_real + torch.randn_like(images_real) * 0.1
                self.real_label = torch.ones(batch_size, device=device, dtype=torch.float32) * 0.9
                self.fake_label = torch.zeros(batch_size, device=device, dtype=torch.float32) + 0.1

                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                # Train with all-real batch
                self.netD.zero_grad()

                output_real = self.netD(images_real_bruit).view(-1)
                loss_D = criterion(output_real, self.real_label)  # real

                # Train with all-fake batch
                images_fake = self.netG(ske)
                images_fake_bruit = images_fake + torch.randn_like(images_fake) * 0.1
                output_fake = self.netD(images_fake_bruit.detach()).view(-1)
                loss_D_G = criterion(output_fake, self.fake_label)  # fake # log(1 - D(G(z)))

                err_D = (loss_D + loss_D_G) * 0.5

                gradient = self.gradient(images_real, images_fake.detach())
                err_D = err_D + 10.0 * gradient

                err_D.backward()
                optimD.step()

                # (2) Update G network: maximize log(D(G(z)))
                self.netG.zero_grad()
                output_fake = self.netD(images_fake).view(-1)
                loss_D_G = criterion(output_fake, self.real_label)  # fake # log(1 - D(G(z)))

                loss_pixel = criterion2(images_fake, images_real)

                loss_G = loss_D_G + loss_pixel * 100
                loss_G.backward()
                optimG.step()

                G_losses += loss_G.item()
                D_losses += loss_D.item()

                if i % 20 == 0:
                    print(f'[{epoch + 1}/{n_epochs}][{i}/{len(self.dataloader)}] '
                          f'Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f} '
                          f'D(x): {output_real.mean().item():.4f} D(G(z)): {output_fake.mean().item():.4f}')

            lossD = D_losses / len(self.dataloader)
            lossG = G_losses / len(self.dataloader)
            print(f'Epoch [{epoch + 1}/{n_epochs}], Loss D : {lossD:.4f}, Loss G : {lossG:.4f}')

        torch.save(self.netG, self.filename)
        print("Saved !!!")

    def gradient(self, real_samples, fake_samples):
        """Compute gradient penalty for WGAN-GP"""
        device = real_samples.device
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        alpha = alpha.expand_as(real_samples)

        # Interpolate between real and fake samples
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)

        # Get discriminator output for interpolated images
        d_interpolated = self.netD(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated,
                                        grad_outputs=torch.ones_like(d_interpolated),
                                        create_graph=True, retain_graph=True)[0]

        # Calculate gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def generate(self, ske):  # TP-TODO
        """ generator of image from skeleton """

        self.netG.eval()
        with torch.no_grad():
            ske_t = torch.from_numpy(ske.__array__(reduced=True).flatten())
            ske_t = ske_t.to(torch.float32)
            ske_t = ske_t.reshape(1, Skeleton.reduced_dim, 1, 1)
            normalized_output = self.netG(ske_t)
            res = self.dataset.tensor2image(normalized_output[0])
        return res


if __name__ == '__main__':
    force = False
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    # if False:
    if True:  # train or load
        # Train
        gen = GenGAN(targetVideoSke, False)
        gen.train(100)  # 5) #200)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)  # load from file

    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        # image = image*255
        nouvelle_taille = (256, 256)
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)
