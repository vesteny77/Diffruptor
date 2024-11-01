import numpy as np
import torch
import torch.nn as nn
import torch.optim.adam as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn.functional as F
import matplotlib.pyplot as plt

from VAE_Model import Encoder, Decoder, Discriminator, LogCoshLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download Dataset
BATCH_SIZE = 100

tf = transforms.Compose([
    transforms.ToTensor()
])
train_set = datasets.CIFAR10('./data', train=True, download=True, transform=tf)
train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
val_set = datasets.CIFAR10('./data', train=False, download=True, transform=tf)
val_loader = DataLoader(val_set, BATCH_SIZE)

encoder = Encoder().to(device)
decoder = Decoder().to(device)
discriminator = Discriminator().to(device)
log_cosh_loss = LogCoshLoss()

optimizer_vae = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0002)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=0.0002)


def vae_loss(recon_x, x, mu, logvar):
    # MSE = F.mse_loss(recon_x, x, reduction='sum')
    # LCOSHL = log_cosh_loss(recon_x, x)
    BCE = nn.BCELoss(reduction='sum')(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print(f"LCOSHL: {LCOSHL:.4f}, KLD: {KLD:.4f}")
    return BCE + 0.01*KLD


def show_images(original, reconstructed):
    # original = original / 2 + 0.5  # unnormalize
    # reconstructed = reconstructed / 2 + 0.5  # unnormalize

    # Create a subplot with 2 rows and 1 column, larger figure size
    fig, axes = plt.subplots(2, 1, figsize=(6, 5))  # Increased figsize

    # Display original images in the top row
    original_image = original.cpu().numpy()
    if original_image.shape[0] == 1:
        original_image = original_image.squeeze(0)
    else:
        original_image = np.transpose(original_image, (1, 2, 0))
    axes[0].imshow(original_image)
    axes[0].axis('off')

    # Display reconstructed images in the bottom row
    reconstructed_image = reconstructed.cpu().numpy()
    if reconstructed_image.shape[0] == 1:
        reconstructed_image = reconstructed_image.squeeze(0)
    else:
        reconstructed_image = np.transpose(reconstructed_image, (1, 2, 0))
    axes[1].imshow(reconstructed_image)
    axes[1].axis('off')

    plt.show()


def train():
    num_epochs = 50
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device=device)

            # VAE forward pass
            mu, logvar = encoder(images)
            std = torch.exp(0.5 * logvar)
            z = torch.randn_like(std) * std + mu
            recon_images = decoder(z)
            # Discriminator forward pass
            real_labels = torch.ones(images.size(0), 1).to(device)
            fake_labels = torch.zeros(images.size(0), 1).to(device)

            outputs = discriminator(images)
            d_loss_real = nn.functional.binary_cross_entropy(outputs, real_labels)

            outputs = discriminator(recon_images.detach())
            d_loss_fake = nn.functional.binary_cross_entropy(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake

            optimizer_discriminator.zero_grad()
            d_loss.backward()
            optimizer_discriminator.step()

            # VAE backward pass
            outputs = discriminator(recon_images)
            g_loss = nn.functional.binary_cross_entropy(outputs, real_labels)
            vae_loss_value = vae_loss(recon_images, images, mu, logvar)

            optimizer_vae.zero_grad()
            (vae_loss_value + g_loss*300).backward()
            # vae_loss_value.backward()
            optimizer_vae.step()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, VAE Loss: {vae_loss_value.item():.4f}')
                # print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], VAE Loss: {vae_loss_value.item():.4f}')
    torch.save(encoder.state_dict(), "encoder_state.pth")
    torch.save(decoder.state_dict(), "decoder_state.pth")

def display(loader):
    with torch.no_grad():
        for images, _ in loader:
            images = images.cuda()
            mu, logvar = encoder(images)
            std = torch.exp(0.5 * logvar)
            z = torch.randn_like(std) * std + mu
            recon_images = decoder(z)
            show_images(vutils.make_grid(images.cpu()[:8]), vutils.make_grid(recon_images.cpu()[:8]))
            break


train()
display(train_loader)
display(val_loader)

