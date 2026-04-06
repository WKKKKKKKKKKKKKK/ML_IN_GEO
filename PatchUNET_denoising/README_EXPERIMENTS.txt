assignment5 experiment structure
================================

This folder keeps the final reproducible experiments used for the notebook-based denoising comparison.

Core files
----------
- `Week7DIP.ipynb`: original notebook reference
- `Marmousi2_5hz_Data.npy`: seismic dataset used by all experiments
- `unet_dip_notebook_repro.py`: UNET experiment script for both no-skip and with-skip modes
- `patchunet_notebook_repro.py`: PatchUNET experiment script
- `patchunet_cnn_repro.py`: CNN-based PatchUNET experiment script

Result folders
--------------
- `no_skip/`: Q1 main UNET result without skip connections
- `with_skip/`: UNET result with additive skip connections
- `patchunet/`: PatchUNET hyperparameter search, final run, and comparison to Q1
- `patchunet_cnn/`: CNN-based PatchUNET hyperparameter search, final run, and comparison to both Q1 and PatchUNET

Example commands
----------------
- Q1 UNET without skip:
  `D:\anaconda3\envs\torch_env\python.exe assignment5\unet_dip_notebook_repro.py --results-json assignment5\no_skip\results.json --summary-path assignment5\no_skip\summary.txt --output-dir assignment5\no_skip\plots`

- UNET with skip:
  `D:\anaconda3\envs\torch_env\python.exe assignment5\unet_dip_notebook_repro.py --use-skip-connections --results-json assignment5\with_skip\results.json --summary-path assignment5\with_skip\summary.txt --output-dir assignment5\with_skip\plots`

- PatchUNET:
  `D:\anaconda3\envs\torch_env\python.exe assignment5\patchunet_notebook_repro.py --results-json assignment5\patchunet\results.json --summary-path assignment5\patchunet\summary.txt --output-dir assignment5\patchunet\plots`

- CNN-PatchUNET:
  `D:\anaconda3\envs\torch_env\python.exe assignment5\patchunet_cnn_repro.py --results-json assignment5\patchunet_cnn\results.json --summary-path assignment5\patchunet_cnn\summary.txt --output-dir assignment5\patchunet_cnn\plots`
