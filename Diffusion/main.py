import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import Generator, Discriminator, DiffAttackGAN
import numpy as np
from PIL import Image

def produce_graph(a, b, c, d):
    a = a.squeeze(0).cpu()
    b = b.squeeze(0).cpu()
    c = c.squeeze(0).cpu()
    d = d.squeeze(0).cpu()
    a = np.uint8(a * 255)
    b = np.uint8(b * 255)
    c = np.uint8(c * 255)
    d = np.uint8(d * 255)
    a = Image.fromarray(np.transpose(a, (1, 2, 0)))
    b = Image.fromarray(np.transpose(b, (1, 2, 0)))
    c = Image.fromarray(np.transpose(c, (1, 2, 0)))
    d = Image.fromarray(np.transpose(d, (1, 2, 0)))
    grid_image = Image.new('RGB', (64, 64))
    # Paste the images into the grid
    grid_image.paste(a, (0, 0))
    grid_image.paste(b, (32, 0))
    grid_image.paste(c, (0, 32))
    grid_image.paste(d, (32, 32))

    # Save the grid image as a PNG file
    grid_image.save('result.png')


def main():
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    trainloader = DataLoader(
        trainset,
        batch_size=80,
        shuffle=True,
        num_workers=2
    )

    valset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    valloader = DataLoader(
        valset,
        batch_size=80,
        shuffle=False,
        num_workers=2
    )

    attacker = DiffAttackGAN(image_size=32)

    print("Testing attack...")
    test_image, _ = next(iter(valloader))
    test_image = test_image[0].unsqueeze(0).cuda()

    adv_image = attacker.attack_vae(test_image)

    test_image = test_image * 0.5 + 0.5
    adv_image = adv_image * 0.5 + 0.5

    with torch.no_grad():
        mu, logvar = attacker.victim_encoder(test_image)
        std = torch.exp(0.5 * logvar)
        z = torch.randn_like(std) * std + mu
        recon_images = attacker.victim_decoder(z)
    MSE = torch.nn.MSELoss(reduction='sum')(recon_images, test_image)  # reconstruction loss

    with torch.no_grad():
        mu_adv, logvar_adv = attacker.victim_encoder(adv_image)
        std = torch.exp(0.5 * logvar_adv)
        z_adv = torch.randn_like(std) * std + mu_adv
        recon_adv_images = attacker.victim_decoder(z_adv)
    MSE_adv = torch.nn.MSELoss(reduction='sum')(adv_image, recon_adv_images)  # reconstruction loss

    print(f"Original VAE reconstruction loss: {MSE.item():.4f}")
    print(f"Adversarial VAE reconstruction loss: {MSE_adv.item():.4f}")
    print(f"Diff: {((MSE_adv.item() - MSE.item())/MSE.item()) * 100:.2f}%")
    produce_graph(test_image, adv_image, recon_images, recon_adv_images)


if __name__ == "__main__":
    main()
