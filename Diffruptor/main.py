import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import DiffAttack
import numpy as np
from PIL import Image
import random
import lpips

def produce_graph(a, b, c, d, index=0):
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
    grid_image.save(f'result_imgs/result_{index}.png')
    a.save("temp/a.png")
    b.save("temp/b.png")
    c.save("temp/c.png")
    d.save("temp/d.png")



def main(img=0):
    attacker = DiffAttack(image_size=32)

    print("Testing attack...")
    test_image, _ = next(iter(cifar_valloader))
    test_image = test_image[img].unsqueeze(0).cuda()

    num_inference_steps = 20
    start_step = 15
    attack_iterations = 60
    lr = 2e-3
    adv_image, adv_latent = attacker.attack_vae(test_image,
                                        num_inference_steps=num_inference_steps,
                                        start_step=start_step,
                                        attack_iterations=attack_iterations,
                                        lr=lr)
    adv_image, _ = attacker.attack_vae(test_image,
                                        num_inference_steps=num_inference_steps,
                                        start_step=start_step + 3,
                                        attack_iterations=attack_iterations,
                                        lr=lr/2,
                                        content_loss_alpha=1,
                                        latent=(adv_latent, start_step))

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
    MSE_adv = torch.nn.MSELoss(reduction='sum')(recon_adv_images, adv_image)  # reconstruction loss

    print(f"Original VAE reconstruction loss: {MSE.item():.4f}")
    print(f"Adversarial VAE reconstruction loss: {MSE_adv.item():.4f}")
    print(f"MSE increase is: {((MSE_adv.item() - MSE.item())/MSE.item()) * 100:.2f}%")
    produce_graph(test_image, adv_image, recon_images, recon_adv_images, img)
    ploss_clean = ploss(test_image, recon_images).item()
    print(f'LPIPS score reconstruction (original): {ploss_clean}')
    ploss_adv = ploss(adv_image, recon_adv_images).item()
    print(f'LPIPS score reconstruction (adversarial): {ploss_adv}')
    print(f"LPIPS increase is: {((ploss_adv - ploss_clean) / ploss_clean) * 100:.2f}%")
    ploss_clean_adv = ploss(test_image, adv_image).item()
    print(f'LPIPS score original vs adversarial: {ploss_clean_adv}')

    return ((MSE_adv.item() - MSE.item())/MSE.item()), ((ploss_adv - ploss_clean) / ploss_clean), MSE_adv.item(), MSE.item(), ploss_clean, ploss_adv, ploss_clean_adv

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    cifar_train = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    cifar_trainloader = DataLoader(
        cifar_train,
        batch_size=80,
        shuffle=True,
        num_workers=2
    )

    cifar_val = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    cifar_valloader = DataLoader(
        cifar_val,
        batch_size=80,
        shuffle=False,
        num_workers=2
    )
    ploss = lpips.LPIPS(net='alex').cuda()
    res_mse = []
    res_lpips = []
    raw_mse = []
    raw_mse_adv = []
    raw_lpips_clean = []
    raw_lpips_adv = []
    raw_lpips_clean_adv = []
    for i in range(50):
        r = random.randint(0, 79)
        print(f"============ IMG {i} =============")
        ret = main(i)
        res_mse.append(round(ret[0], 5))
        res_lpips.append(round(ret[1], 5))
        raw_mse_adv.append(round(ret[2], 5))
        raw_mse.append(round(ret[3], 5))
        raw_lpips_clean.append(round(ret[4], 5))
        raw_lpips_adv.append(round(ret[5], 5))
        raw_lpips_clean_adv.append(round(ret[6], 5))

        # print("MSE Loss increase:")
        # print(res_mse)
        # print("LPIPS Loss increase:")
        # print(res_lpips)
        # print("Raw MSE Loss for clean image:")
        # print(raw_mse)
        # print("Raw MSE Loss for adversarial image:")
        # print(raw_mse_adv)
        # print("Raw LPIPS Loss for clean image:")
        # print(raw_lpips_clean)
        # print("Raw LPIPS Loss for adversarial image:")
        # print(raw_lpips_adv)
        # print("Raw LPIPS Loss for clean_vs_adversarial:")
        # print(raw_lpips_clean_adv)
        print("===================================")
        print("")
