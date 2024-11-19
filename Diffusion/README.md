# AdvDiff Attack Method
## Basic Algorithm
1. Start from random noise
2. Apply classifier-free guidance for high quality
3. Add adversarial guidance at each step
4. Use noise sampling guidance if needed
5. Return successful adversarial example

## Classifier-Free Guidance
```python
def classifier_free_guidance(self, x_t, t, y, w=0.3):
```
implements the classifier-free guidance that is used to generate high-quality conditional samples.
- Gets predictions with and without the condition
- Combines them using guidance weight w
- Returns the guided noise prediction

## Adversarial Guidance
```python
def adversarial_guidance(self, x_t, t, target_model, y_target, s=0.5):
```
implements the adversarial guidance:
- computing gradient towards target class
- scaling gradient by variance and guidance scale
- adding scaled gradient to current sample

## Noise Sampling Guidance
```python
def noise_sampling_guidance(self, x_T, x_0, target_model, y_target, a=1.0):
```
implements the noise sampling guidance:
- computing gradient from final sample
- applying guidance to initial noise
- scaling by noise level and guidance scale

## Main Generation Process
```python
def generate_adversarial(self, target_model, y_target, y_true, ...):
```
This functions combines all previous modules.