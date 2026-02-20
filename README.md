# f-AnoGAN Anomaly Detection on MNIST (Digits 0, 1, 8)

This project implements **f-AnoGAN** for **one-class anomaly detection** on MNIST.
We train a GAN on one “normal” digit (e.g., 0), train an encoder to map images to latent space, and detect anomalies using a weighted residual + feature score.

## Project Structure
- `src/train_gan.py` : Train GAN on normal digit
- `src/train_encoder.py` : Train encoder (f-AnoGAN)
- `src/score.py` : Anomaly score (pixel + feature residual)
- `src/eval.py` : AUROC/AUPRC, ROC/PR curves, example reconstructions
- `src/summarize_results.py` : Run digits 0/1/8 and export summary CSV

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
