import torch
import torch.nn as nn
import torch.nn.functional as F
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


class AdvDiff:
    def __init__(self, image_size=32, device="cuda"):
        self.unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(device)
        self.scheduler = DDIMScheduler.from_pretrained("google/ddpm-cifar10-32")
        self.device = device
        self.image_size = image_size
        
    def classifier_free_guidance(self, x_t, t, y, w=0.3):
        """Classifier-free guidance for conditional generation"""
        eps_t_uncond = self.unet(x_t, t).sample
        eps_t_cond = self.unet(x_t, t, y).sample
        eps_t = (1 + w) * eps_t_cond - w * eps_t_uncond
        return eps_t

    def adversarial_guidance(self, x_t, t, target_model, y_target, s=0.5):
        """Adversarial guidance to fool target model"""
        x_t.requires_grad_(True)
        
        pred = target_model(x_t)
        
        loss = F.cross_entropy(pred, y_target)
        grad = torch.autograd.grad(loss, x_t)[0]
        
        # This part is the adversarial guidance
        x_t_adv = x_t + s * self.scheduler.sigmas[t].item() ** 2 * grad
        x_t.requires_grad_(False)
        
        return x_t_adv

    def noise_sampling_guidance(self, x_T, x_0, target_model, y_target, a=1.0):
        x_0.requires_grad_(True)
        
        pred = target_model(x_0)
        loss = F.cross_entropy(pred, y_target)
        grad = torch.autograd.grad(loss, x_0)[0]
        
        sigma_T = self.scheduler.sigmas[-1].item()
        x_T_adv = x_T + a * sigma_T ** 2 * grad
        x_0.requires_grad_(False)
        
        return x_T_adv

    def generate_adversarial(
        self,
        target_model,
        y_target,
        y_true,
        num_inference_steps=50,
        guidance_scale=0.5,
        adv_scale=0.5,
        noise_guidance_scale=1.0,
        num_noise_steps=10,
    ):
        self.scheduler.set_timesteps(num_inference_steps)
        
        x_t = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
        
        # adversarial examples
        x_adv = None
        
        for _ in range(num_noise_steps):
            # Reverse diffusion process
            for t in tqdm(self.scheduler.timesteps):
                eps_t = self.classifier_free_guidance(x_t, t, y_true, guidance_scale)
                
                x_t = self.scheduler.step(eps_t, t, x_t).prev_sample
                
                x_t = self.adversarial_guidance(x_t, t, target_model, y_target, adv_scale)
            
            x_0 = x_t
            
            with torch.no_grad():
                pred = target_model(x_0)
                if pred.argmax() == y_target:
                    x_adv = x_0
                    break
            
            x_t = self.noise_sampling_guidance(x_t, x_0, target_model, y_target, noise_guidance_scale)
        
        return x_adv

    def attack_vae_gan(
        self,
        vae_encoder,
        vae_decoder,
        discriminator,
        image,
        num_inference_steps=50,
        guidance_scale=0.3,
        adv_scale=0.5,
        noise_guidance_scale=1.0,
    ):
        class VAEGANTarget(nn.Module):
            def __init__(self, encoder, decoder, discriminator):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
                self.discriminator = discriminator
                
            def forward(self, x):
                mu, logvar = self.encoder(x)
                z = self.reparameterize(mu, logvar)
                recon = self.decoder(z)
                return self.discriminator(recon)
                
            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std
        
        target_model = VAEGANTarget(vae_encoder, vae_decoder, discriminator).to(self.device)
        target_model.eval()
        
        y_target = torch.zeros(1).to(self.device) if image.mean() > 0.5 else torch.ones(1).to(self.device)
        y_true = torch.ones(1).to(self.device) if image.mean() > 0.5 else torch.zeros(1).to(self.device)
        
        x_adv = self.generate_adversarial(
            target_model,
            y_target,
            y_true,
            num_inference_steps,
            guidance_scale,
            adv_scale,
            noise_guidance_scale
        )
        
        return x_adv


# def attack_advdiff():
#     advdiff = AdvDiff(image_size=32)

#     vae_encoder = Encoder().cuda()
#     vae_decoder = Decoder().cuda()
#     discriminator = Discriminator().cuda()

#     x_adv = advdiff.attack_vae_gan(
#         vae_encoder,
#         vae_decoder,
#         discriminator,
#         image,
#         num_inference_steps=50,
#         guidance_scale=0.3,
#         adv_scale=0.5,
#         noise_guidance_scale=1.0
#     )

#     return x_adv


class DiffAttack2:
    def __init__(self, image_size=32, device="cuda"):
        self.device = device
        self.image_size = image_size
        self.unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(device)
        self.scheduler = DDIMScheduler.from_pretrained("google/ddpm-cifar10-32")
        self.perceptual_loss = lpips.LPIPS(net='alex').to(device)
    
    def ddim_forward_to_timestep(self, image, target_timestep, num_inference_steps=20):
        """forward diffusion up to a specific timestep"""
        self.scheduler.set_timesteps(num_inference_steps)
        
        latent = image.to(self.device)
        timesteps = self.scheduler.timesteps
        
        for t in timesteps[:target_timestep]:
            with torch.no_grad():
                noise_pred = self.unet(latent, t).sample
                latent = self.scheduler.step(noise_pred, t, latent).prev_sample
        
        return latent, timesteps[target_timestep:]

    def sample_uniform_timesteps(self, total_steps, num_samples):
        """sample timesteps uniformly"""
        indices = torch.linspace(0, total_steps - 1, num_samples)
        indices = indices.long()
        return indices

    def segment_wise_backward(self, loss_fn, x_t, timesteps, start_idx, end_idx):
        grad = None
        for t in reversed(timesteps[start_idx:end_idx]):
            # store intermediate values
            x_t.requires_grad_(True)
            noise_pred = self.unet(x_t, t).sample
            next_x = self.scheduler.step(noise_pred, t, x_t).prev_sample
            
            if t == timesteps[end_idx - 1]:
                loss = loss_fn(next_x)
                grad = torch.autograd.grad(loss, x_t)[0]
            else:
                grad = torch.autograd.grad((next_x * grad).sum(), x_t)[0]
            
            x_t = x_t.detach()
            x_t = x_t - grad
            
        return x_t, grad

    def deviated_reconstruction_loss(self, x_t, x_t_orig, t):
        diff = x_t - x_t_orig
        loss = torch.mean(diff ** 2)
        alpha = 1.0 / (1.0 + t.item())
        return alpha * loss

    def attack(
        self,
        image,
        vae_encoder,
        vae_decoder,
        discriminator,
        num_inference_steps=20,
        start_timestep=10,
        num_sampling_steps=5,  # uniform samples for reconstruction loss
        attack_steps=100,
        lr=0.01,
        segment_size=5,
    ):
        vae_encoder.eval()
        vae_decoder.eval()
        discriminator.eval()
        self.unet.eval()
        
        latent, remaining_timesteps = self.ddim_forward_to_timestep(
            image, 
            start_timestep, 
            num_inference_steps
        )
        latent = latent.detach().requires_grad_(True)
        
        sampling_timesteps = self.sample_uniform_timesteps(
            len(remaining_timesteps), 
            num_sampling_steps
        )
        
        optimizer = optim.AdamW([latent], lr=lr)
        
        for step in tqdm(range(attack_steps), desc="DiffAttack2 Running"):
            optimizer.zero_grad()
            
            current_latent = latent
            total_loss = 0
            
            for seg_start in range(0, len(remaining_timesteps), segment_size):
                seg_end = min(seg_start + segment_size, len(remaining_timesteps))
                
                def segment_loss(x):
                    mu, logvar = vae_encoder(x)
                    z = torch.randn_like(mu) * torch.exp(0.5 * logvar) + mu
                    recon = vae_decoder(z)
                    
                    disc_out = discriminator(recon)
                    target = torch.zeros_like(disc_out) if disc_out.mean() > 0.5 else torch.ones_like(disc_out)
                    adv_loss = F.binary_cross_entropy(disc_out, target)
                    
                    dev_loss = 0
                    if seg_end - 1 in sampling_timesteps:
                        dev_loss = self.deviated_reconstruction_loss(x, image, remaining_timesteps[seg_end-1])
                    
                    content_loss = F.mse_loss(x, image)
                    
                    perc_loss = self.perceptual_loss(x, image)
                    
                    return (
                        adv_loss + 
                        0.1 * dev_loss +
                        10.0 * content_loss +
                        5.0 * perc_loss
                    )
                
                current_latent, grad = self.segment_wise_backward(
                    segment_loss, 
                    current_latent,
                    remaining_timesteps,
                    seg_start,
                    seg_end
                )
                
                if grad is not None:
                    total_loss += grad.norm()
            
            total_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                delta = latent - image
                delta = torch.clamp(delta, -8/255, 8/255)
                latent.data = image + delta
        
        with torch.no_grad():
            adv_image = latent
            for t in remaining_timesteps:
                noise_pred = self.unet(adv_image, t).sample
                adv_image = self.scheduler.step(noise_pred, t, adv_image).prev_sample
        
        return adv_image
