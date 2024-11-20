# Diffusion-based Adversarial Attack Methods
1. [AdvDiff](#advdiff-attack-method)
1. [DiffAttack](#diffattack)
1. [DiffAttack2](#diffattack2-aaginst-diffusion-based-adversarial-purification)

## AdvDiff Attack Method
### Basic Algorithm
1. Start from random noise
2. Apply classifier-free guidance for high quality
3. Add adversarial guidance at each step
4. Use noise sampling guidance if needed
5. Return successful adversarial example

### Classifier-Free Guidance
```python
def classifier_free_guidance(self, x_t, t, y, w=0.3):
```
implements the classifier-free guidance that is used to generate high-quality conditional samples.
- Gets predictions with and without the condition
- Combines them using guidance weight w
- Returns the guided noise prediction

### Adversarial Guidance
```python
def adversarial_guidance(self, x_t, t, target_model, y_target, s=0.5):
```
implements the adversarial guidance:
- computing gradient towards target class
- scaling gradient by variance and guidance scale
- adding scaled gradient to current sample

### Noise Sampling Guidance
```python
def noise_sampling_guidance(self, x_T, x_0, target_model, y_target, a=1.0):
```
implements the noise sampling guidance:
- computing gradient from final sample
- applying guidance to initial noise
- scaling by noise level and guidance scale

### Main Generation Process
```python
def generate_adversarial(self, target_model, y_target, y_true, ...):
```
This functions combines all previous modules.


## DiffAttack
- TODO


## DiffAttack2 (aaginst diffusion-based adversarial purification)
### 1. DDIM Inversion:
```python
def ddim_inversion(self, image, num_inference_steps=20):
```
- Takes the original image and runs it through the diffusion process
- Stores intermediate latents at each timestep

### 2. Gradient computation:
```python
def segment_wise_backward(self, loss_fn, x_t, timesteps, start_idx, end_idx):
```
- processes gradients in segments to save memory
- uses segment-wise processing instead of full backpropagation
- maintains constant memory usage regardless of diffusion length
- updates latents incrementally

### 3. Loss function:

Deviated Reconstruction Loss:

```python
def deviated_reconstruction_loss(self, x_t, x_t_orig, t):
    diff = x_t - x_t_orig
    loss = torch.mean(diff ** 2)
    alpha = 1.0 / (1.0 + t.item())
    return alpha * loss
```
to corrupt the diffusion process

VAE Attack Loss:

```python
mu, logvar = vae_encoder(x)
z = torch.randn_like(mu) * torch.exp(0.5 * logvar) + mu
recon = vae_decoder(z)
```

Adversarial Discriminator Loss:

```python
disc_out = discriminator(recon)
target = torch.zeros_like(disc_out) if disc_out.mean() > 0.5 else torch.ones_like(disc_out)
adv_loss = F.binary_cross_entropy(disc_out, target)
```

Content Preservation & Perceptual Loss:

```python
content_loss = F.mse_loss(x, image)
perc_loss = self.perceptual_loss(x, image)
```

### 4. Main Attack Loop:
```python
def attack(self, image, vae_encoder, vae_decoder, discriminator, ...):
```
