# Assignment 9 Report

The experiment used OpenFWI velocity model data from:

```text
model-20260504T122211Z-3-001/model
```

The notebook loaded all 60 `.npy` files, from `model1.npy` to `model60.npy`. Each file contains velocity models with shape `(500, 1, 70, 70)`, so the run used 30000 velocity models in total. The train/validation split was 27000 training samples and 3000 validation samples. The original velocity range was from 1500.0 to 4500.0, and the models were normalized to `[0, 1]` before training.

The experiment was run on CUDA. The default setup used a latent dimension of 64, batch size of 64, 10 VAE epochs, and 10 GAN epochs.

Four models were trained and compared:

- MLP-VAE
- CNN-VAE
- MLP-GAN
- CNN-GAN

For the VAE models, the metric recorded during training was validation MSE. The MLP-VAE validation MSE decreased from `0.008507` at epoch 1 to `0.003994` at epoch 10. The CNN-VAE performed better, decreasing from `0.005947` at epoch 1 to `0.001422` at epoch 10. This shows that the CNN-VAE reconstructed the velocity models more accurately than the MLP-VAE.

For the GAN models, discriminator loss and generator loss were recorded. The MLP-GAN ended with discriminator loss `0.668086` and generator loss `0.856578`. The CNN-GAN ended with discriminator loss `0.179292` and generator loss `5.443690`.

The final model comparison used `generation_score`, which combines value statistics, histogram difference, and spatial-gradient difference between generated and real velocity models. Lower score means the generated velocity models are closer to the real OpenFWI models.

| Model | Family | Generation Score | Stats Error | Hist Error | Grad Error | Val MSE | D Loss | G Loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CNN-VAE | VAE | 0.028680 | 0.011219 | 0.006491 | 0.010970 | 0.001415 | - | - |
| CNN-GAN | GAN | 0.031843 | 0.012995 | 0.007808 | 0.011040 | - | 0.179292 | 5.443690 |
| MLP-GAN | GAN | 0.048641 | 0.005522 | 0.005395 | 0.037724 | - | 0.668086 | 0.856578 |
| MLP-VAE | VAE | 0.108828 | 0.077877 | 0.016271 | 0.014680 | 0.003975 | - | - |

Based on the saved notebook output, `CNN-VAE` achieved the best generation result with the lowest generation score of `0.028680`. The CNN-based models also performed better than the MLP-based models overall, which is reasonable because velocity models are 2-D spatial images and convolutional layers can preserve local structure better than flattened MLP inputs.

The visual sample grids saved in the notebook also compare real velocity models with generated samples from all four models. The quantitative result selects `CNN-VAE` as the best model for this run.
