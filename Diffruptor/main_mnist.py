import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from models_mnist import DiffAttack
import numpy as np
from PIL import Image
import random

def produce_graph(a, b, c, d, index=0):
    a = a.squeeze(0).squeeze(0).cpu()
    b = b.squeeze(0).squeeze(0).cpu()
    c = c.squeeze(0).squeeze(0).cpu()
    d = d.squeeze(0).squeeze(0).cpu()
    a = np.uint8(a * 255)
    b = np.uint8(b * 255)
    c = np.uint8(c * 255)
    d = np.uint8(d * 255)
    a = Image.fromarray(a)
    b = Image.fromarray(b)
    c = Image.fromarray(c)
    d = Image.fromarray(d)
    grid_image = Image.new('L', (56, 56))
    # Paste the images into the grid
    grid_image.paste(a, (0, 0, 28, 28))
    grid_image.paste(b, (28, 0, 56, 28))
    grid_image.paste(c, (0, 28, 28, 56))
    grid_image.paste(d, (28, 28, 56, 56))

    # Save the grid image as a PNG file
    grid_image.save(f'result_imgs/result_{index}.png')


def main(img, attacker):
    print("Testing attack...")
    test_image, _ = next(iter(mnist_testloader))
    test_image = test_image[img].unsqueeze(0).cuda()

    num_inference_steps = 20
    start_step = 14
    attack_iterations = 100
    lr = 2e-3
    adv_image, adv_latent = attacker.attack_vae(test_image,
                                        num_inference_steps=num_inference_steps,
                                        start_step=start_step,
                                        attack_iterations=attack_iterations,
                                        lr=lr,
                                        content_loss_alpha=10)


    test_image = test_image * 0.5 + 0.5
    adv_image = adv_image * 0.5 + 0.5

    with torch.no_grad():
        recon_images, _, _ = attacker.vae_pretrained(test_image)
        recon_images = recon_images.reshape(1, 1, 28, 28)
    MSE = torch.nn.MSELoss(reduction='sum')(recon_images, test_image)  # reconstruction loss

    with torch.no_grad():
        recon_adv_images, _, _ = attacker.vae_pretrained(adv_image)
        recon_adv_images = recon_adv_images.reshape(1, 1, 28, 28)
    MSE_adv = torch.nn.MSELoss(reduction='sum')(recon_adv_images, adv_image)  # reconstruction loss

    print(f"Original VAE reconstruction loss: {MSE.item():.4f}")
    print(f"Adversarial VAE reconstruction loss: {MSE_adv.item():.4f}")
    print(f"Diff: {((MSE_adv.item() - MSE.item())/MSE.item()) * 100:.2f}%")
    produce_graph(test_image, adv_image, recon_images, recon_adv_images, img)
    return round((MSE_adv.item() - MSE.item())/MSE.item(), 5)


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_trainloader = DataLoader(mnist_train, batch_size=128, shuffle=True)
    mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    mnist_testloader = DataLoader(mnist_test, batch_size=128, shuffle=False)
    test_image, _ = next(iter(mnist_testloader))
    attacker = DiffAttack(image_size=28)

    res = []
    for i in range(50):
        # r = random.randint(0, 79)
        res.append(main(i, attacker))
        # print(res)
