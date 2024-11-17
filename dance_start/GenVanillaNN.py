import numpy as np
import cv2
import os
import pickle
import sys
import math

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.io import read_image

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# from tensorboardX import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton

import torch.optim as optim

torch.set_default_dtype(torch.float32)


class SkeToImageTransform:
    def __init__(self, image_size):
        self.imsize = image_size

    def __call__(self, ske):
        # image = Image.new('RGB', (self.imsize, self.imsize), (255, 255, 255))
        image = white_image = np.ones((self.imsize, self.imsize, 3), dtype=np.uint8) * 255
        ske.draw(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('Image', image)
        # key = cv2.waitKey(-1)
        return image


class VideoSkeletonDataset(Dataset):
    def __init__(self, videoSke, ske_reduced, source_transform=None, target_transform=None, optSkeOrImage=1):

        self.videoSke = videoSke
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced
        self.optSkeOrImage = optSkeOrImage
        self.skeToImageTransform = SkeToImageTransform(64)
        print("VideoSkeletonDataset: ",
              "ske_reduced =", ske_reduced, " =(", Skeleton.reduced_dim, " or ", Skeleton.full_dim, ")")

    def __len__(self):
        return self.videoSke.skeCount()

    def __getitem__(self, idx):
        # prepreocess skeleton (input)
        reduced = True
        ske = self.videoSke.ske[idx]
        if self.optSkeOrImage == 1:
            ske = self.preprocessSkeleton(ske)
            # prepreocess image (output)
            image = Image.open(self.videoSke.imagePath(idx))
            if self.target_transform:
                image = self.target_transform(image)
            return ske, image
        if self.optSkeOrImage == 2:
            ske_image = self.skeToImageTransform(ske)
            ske_image = np.array(ske_image, dtype=np.float32)
            ske_image = torch.tensor(ske_image).permute(2, 0, 1)  # Convert to (C, H, W)
            if self.source_transform:
                ske_image = self.source_transform(ske_image)
            image = Image.open(self.videoSke.imagePath(idx))
            if self.target_transform:
                image = self.target_transform(image)
            return ske_image, image
        else:
            print("Invalid option.")

    def preprocessSkeleton(self, ske):
        if self.source_transform:
            ske = self.source_transform(ske)
        else:
            ske = torch.from_numpy(ske.__array__(reduced=self.ske_reduced).flatten())
            ske = ske.to(torch.float32)
            ske = ske.reshape(ske.shape[0], 1, 1)
        return ske

    def tensor2image(self, normalized_image):
        numpy_image = normalized_image.detach().numpy()
        # RÃ©organiser les dimensions (C, H, W) en (H, W, C)
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        # passage a des images cv2 pour affichage
        numpy_image = cv2.cvtColor(np.array(numpy_image), cv2.COLOR_RGB2BGR)
        denormalized_image = numpy_image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
        denormalized_output = denormalized_image * 1
        return denormalized_output


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GenNNSkeToImage(nn.Module):
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """

    def __init__(self, optSkeOrImage=1):
        super(GenNNSkeToImage, self).__init__()
        self.input_dim = Skeleton.reduced_dim
        self.optSkeOrImage = optSkeOrImage
        if self.optSkeOrImage == 1:
            self.model = nn.Sequential(
                nn.Linear(26, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 3 * 64 * 64)
            )
        if self.optSkeOrImage == 2:
            self.model = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # [batch_size, 64, 32, 32]
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [batch_size, 128, 16, 16]
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [batch_size, 256, 8, 8]
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # [batch_size, 512, 4, 4]
                nn.ReLU(),
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # Upsample to [batch_size, 256, 8, 8]
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Upsample to [batch_size, 64, 32, 32]
                nn.ReLU(),
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
                nn.Tanh()
            )
        else:
            print("Errure changer la option vers 1 ou 2 ")

        print(self.model)

    def forward(self, z):
        if self.optSkeOrImage == 1:
            img = self.model(z)
            img = img.view(img.size(0), 3, 64, 64)
            return img
        if self.optSkeOrImage == 2:
            return self.model(z)
        else:
            print("Erreur changer la option vers 1 ou 2 ")


class GenVanillaNN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """

    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        image_size = 64
        self.optSkeOrImage = optSkeOrImage
        self.netG = GenNNSkeToImage(self.optSkeOrImage)
        self.skeToImageTransform = SkeToImageTransform(image_size)
        src_transform = None
        self.filename = 'C:/Users/DELL/Downloads/tp_dance_start/dance_start/data/DanceGenVanillaFromSke.pth'

        tgt_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # [transforms.Resize((64, 64)),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform,
                                            source_transform=src_transform, optSkeOrImage=self.optSkeOrImage)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=16, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenVanillaNN: Load=", self.filename)
            print("GenVanillaNN: Current Working Directory: ", os.getcwd())
            self.netG = torch.load(self.filename)

    def train(self, n_epochs=20):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.netG.parameters(), lr=0.001)

        for epoch in range(n_epochs):
            loss = 0.0
            for i, (skeletons, images) in enumerate(self.dataloader):
                optimizer.zero_grad()

                if self.optSkeOrImage == 1:
                    skeletons = skeletons.view(skeletons.size(0), -1)

                images_g = self.netG(skeletons)
                loss = criterion(images_g, images)
                loss.backward()
                optimizer.step()

                loss += loss.item()

            print(f"Epoch {epoch + 1}/{n_epochs}, Loss = {loss:.4f}")
        torch.save(self.netG, self.filename)
        print("Saved !!!")

    def generate(self, ske):
        """ generator of image from skeleton """
        if self.optSkeOrImage == 2:
            ske_image = self.skeToImageTransform(ske)
            ske_image = np.array(ske_image, dtype=np.float32)
            ske_t = transforms.ToTensor()(ske_image).unsqueeze(0)
        elif self.optSkeOrImage == 1 :
            ske_t = self.dataset.preprocessSkeleton(ske)
            ske_t = ske_t.view(1, -1)
        else :
            print("Erreur changer la option vers 1 ou 2")

        normalized_output = self.netG(ske_t)

        normalized_output = normalized_output.squeeze(0)
        res = self.dataset.tensor2image(normalized_output)
        return res



if __name__ == '__main__':
    force = False
    optSkeOrImage = 2  # use as input a skeleton (1) or an image with a skeleton drawn (2)
    n_epoch = 100  # 200
    train = False

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenVanillaNN: Current Working Directory=", os.getcwd())
    print("GenVanillaNN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    if train:
        # Train
        gen = GenVanillaNN(targetVideoSke, loadFromFile=False)
        gen.train(n_epoch)
    else:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=True)  # load from file

    # Test with a second video
    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        # image = image*255
        nouvelle_taille = (256, 256)
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)