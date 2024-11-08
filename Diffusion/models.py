import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import DDIMScheduler, UNet2DModel, AutoencoderKL
from tqdm import tqdm
from VAE_Model import Encoder, Decoder
import lpips

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x.view(-1, self.latent_dim, 1, 1))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)


class AttentionStore:
    def __init__(self, res):
        self.res = res
        self.store = {}
        self.attention_maps = []

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if attn.shape[1] <= (self.res ** 2):
            self.attention_maps.append(attn)
        return attn

    def get_average_attention(self):
        return torch.stack(self.attention_maps).mean(0)

    def reset(self):
        self.attention_maps = []


def contrastive_loss(reconstructed, original):
    return torch.mean((reconstructed - original) ** 2 / (torch.std(original) + 1e-5))


class DiffAttackGAN:
    def __init__(self, image_size=32):
        self.unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").cuda().eval()

        self.scheduler = DDIMScheduler.from_pretrained("google/ddpm-cifar10-32")

        self.attention_store = AttentionStore(image_size)
        self.image_size = image_size

        self.victim_encoder = Encoder().cuda()
        self.victim_encoder.load_state_dict(torch.load("encoder_state.pth", weights_only=True))
        self.victim_decoder = Decoder().cuda()
        self.victim_decoder.load_state_dict(torch.load("decoder_state.pth", weights_only=True))
        self.victim_encoder.eval()
        self.victim_decoder.eval()

        self.ploss = lpips.LPIPS(net='alex').cuda()


    def ddim_inversion(self, image, num_inference_steps=20):
        with torch.no_grad():
            self.scheduler.set_timesteps(num_inference_steps)
            # latent = self.encode_image(image)
            latent = image
            all_latents = [latent]
            # Inversion process
            for t in tqdm(self.scheduler.timesteps.flip(0)[:-1], desc="DDIM Inversion"):
                noise_pred = self.unet(latent, t).sample
                alpha_bar = self.scheduler.alphas_cumprod[t]
                next_timestep = t + self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                alpha_bar_next = self.scheduler.alphas_cumprod[next_timestep] \
                    if next_timestep <= self.scheduler.config.num_train_timesteps else torch.tensor(0.0)
                rev_x0 = (latent - noise_pred * torch.sqrt(1 - alpha_bar)) / torch.sqrt(alpha_bar)
                latent = rev_x0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * noise_pred
                all_latents.append(latent.cuda())
        return latent, all_latents

    def vae_loss(self, img):
        mu, logvar = self.victim_encoder(img)
        std = torch.exp(0.5 * logvar)
        z = torch.randn_like(std) * std + mu
        recon_images = self.victim_decoder(z)
        BCE = torch.nn.BCELoss(reduction='sum')(recon_images, img)  # reconstruction loss
        return BCE

    def vae_latent_loss(self, img1, img2):
        mu1, logvar1 = self.victim_encoder(img1)
        mu2, logvar2 = self.victim_encoder(img2)

        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)

        # KL divergence term-by-term calculation
        term1 = torch.log(var2.sqrt() / var1.sqrt())
        term2 = (var1 + (mu1 - mu2).pow(2)) / (2 * var2)
        kl_div = torch.sum(term1 + term2 - 0.5)
        return kl_div


    def attack_vae(
        self,
        image,
        num_inference_steps=20,
        start_step=15,
        attack_iterations=80,
        lr=2e-3,
    ):
        _, latents = self.ddim_inversion(image, num_inference_steps)
        latent = latents[start_step - 1]
        latent.requires_grad_(True)

        optimizer = optim.AdamW([latent], lr=lr)
        vae_image = image * 0.5 + 0.5

        with tqdm(range(attack_iterations), desc="Attacking VAE") as iterator:
            for _ in iterator:
                optimizer.zero_grad()
                # self.attention_store.reset()

                current_latents = latent
                for t in self.scheduler.timesteps[1+start_step-1:]:
                    noise_pred = self.unet(current_latents, t).sample
                    current_latents = self.scheduler.step(
                        noise_pred,
                        t,
                        current_latents
                    ).prev_sample

                # adv_image = self.decode_latent(current_latents)
                adv_image = current_latents
                vae_adv_image = adv_image * 0.5 + 0.5

                # 1. VAE Fooling Loss - try to make VAE fail to reconstruct
                # vae_bce_loss = self.vae_loss(vae_adv_image)
                # fooling_loss = - vae_bce_loss  # ~0.0005

                # Add a different loss that is the latent space loss in vae.
                vae_latent_loss = self.vae_latent_loss(vae_image, vae_adv_image)
                fooling_loss = -vae_latent_loss

                # 2. Content Preservation Loss
                content_loss = torch.nn.functional.mse_loss(adv_image, image)  # ~0.1

                # 3.Perceptual loss
                perceptual_loss = self.ploss(adv_image, image)

                total_loss = (
                    fooling_loss +
                    320000 * content_loss +
                    100000 * perceptual_loss
                )
                # 2500, 8
                total_loss.backward()
                optimizer.step()

                iterator.set_postfix(ordered_dict={
                    "L_VAE (big): ": vae_latent_loss.item(),
                    "L_Content (small): ": content_loss.item(),
                    "L_Perceptual(small): ": perceptual_loss.item()
                })

        with torch.no_grad():
            adv_latents = latent.clone()
            for t in self.scheduler.timesteps[1+start_step-1:]:
                noise_pred = self.unet(adv_latents, t).sample
                adv_latents = self.scheduler.step(
                    noise_pred,
                    t,
                    adv_latents
                ).prev_sample

            adv_image = adv_latents

        return adv_image
