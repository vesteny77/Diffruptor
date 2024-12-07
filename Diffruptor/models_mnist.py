import torch
import torch.optim as optim
from diffusers import DDIMScheduler, UNet2DModel, AutoencoderKL
from tqdm import tqdm
from VAE_Model import VAE_Pretrained

class DiffAttack:
    def __init__(self, image_size=32):
        self.unet = UNet2DModel.from_pretrained('1aurent/ddpm-mnist').cuda().eval()

        self.scheduler = DDIMScheduler.from_pretrained('1aurent/ddpm-mnist')

        self.image_size = image_size

        self.vae_pretrained = VAE_Pretrained().cuda().eval()
        self.vae_pretrained.load_state_dict(torch.load("vae_epoch_25.pth", weights_only=True))

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
        recon_images, _, _ = self.vae_pretrained(img)
        BCE = torch.nn.BCELoss(reduction='sum')(recon_images.reshape(1, 1, 28, 28), img)  # reconstruction loss
        return BCE

    def vae_latent_loss(self, img1, img2):
        mu1, logvar1 = self.vae_pretrained.encode(img1)
        mu2, logvar2 = self.vae_pretrained.encode(img2)

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
        attack_iterations=60,
        lr=2e-3,
        content_loss_alpha=1,
        latent=None
    ):
        if latent is None:
            _, latents = self.ddim_inversion(image, num_inference_steps)
            latent = latents[start_step - 1]
        else:
            latent, old_start_step = latent
            adv_latents = latent.clone()
            for t in self.scheduler.timesteps[1 + old_start_step - 1:start_step]:
                noise_pred = self.unet(adv_latents, t).sample
                adv_latents = self.scheduler.step(
                    noise_pred,
                    t,
                    adv_latents
                ).prev_sample
            latent = adv_latents.detach().cuda()

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


                # 1. VAE Fooling Loss
                vae_latent_loss = self.vae_latent_loss(vae_image, vae_adv_image)
                fooling_loss = -vae_latent_loss

                # 2. Content Preservation Loss
                content_loss = torch.nn.functional.mse_loss(adv_image, image)  # ~0.1


                total_loss = (
                    fooling_loss +
                    content_loss_alpha * 32 * 10e4 * content_loss
                )
                # 2500, 8
                total_loss.backward()
                optimizer.step()

                iterator.set_postfix(ordered_dict={
                    "L_VAE: ": fooling_loss.item(),
                    "L_Content: ": content_loss.item(),
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

        return adv_image, latent.clone()

