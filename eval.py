import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

from src.data import get_mnist_loaders
from src.models import Generator, Discriminator, Encoder
from src.score import anomaly_score
from src.utils import get_device, set_seed


def save_grid_comparison(x, x_hat, y_bin, scores, out_path, n_show=16):
    """
    Save a figure with rows: [original, reconstruction, residual]
    """
    x = x[:n_show].cpu()
    x_hat = x_hat[:n_show].cpu()
    y_bin = y_bin[:n_show].cpu()
    scores = scores[:n_show].cpu()

    # unnormalize for display
    x01 = x * 0.5 + 0.5
    xhat01 = x_hat * 0.5 + 0.5
    resid = torch.abs(x01 - xhat01)

    fig, axes = plt.subplots(3, n_show, figsize=(2*n_show, 6))
    for i in range(n_show):
        axes[0, i].imshow(x01[i, 0], cmap="gray")
        axes[0, i].set_title(f"y={int(y_bin[i])}\nS={scores[i]:.3f}")
        axes[0, i].axis("off")

        axes[1, i].imshow(xhat01[i, 0], cmap="gray")
        axes[1, i].set_title("recon")
        axes[1, i].axis("off")

        axes[2, i].imshow(resid[i, 0], cmap="gray")
        axes[2, i].set_title("resid")
        axes[2, i].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--digit", type=int, default=0)
    parser.add_argument("--gan_ckpt", type=str, required=True)
    parser.add_argument("--enc_ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--lambda_pix", type=float, default=1.0)
    parser.add_argument("--lambda_feat", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    loaders = get_mnist_loaders(
        normal_digit=args.digit,
        batch_size=args.batch_size,
        val_size=2000,
        anomaly_fraction_in_val=0.5,
        anomaly_fraction_in_test=0.5,
    )

    # Load models
    G = Generator(z_dim=args.z_dim).to(device)
    D = Discriminator().to(device)
    E = Encoder(z_dim=args.z_dim).to(device)

    gan = torch.load(args.gan_ckpt, map_location=device)
    enc = torch.load(args.enc_ckpt, map_location=device)

    G.load_state_dict(gan["G_state"])
    D.load_state_dict(gan["D_state"])
    E.load_state_dict(enc["E_state"])

    G.eval(); D.eval(); E.eval()

    # ---------- Evaluate on TEST ----------
    all_scores = []
    all_labels = []
    all_x = []
    all_xhat = []

    with torch.no_grad():
        for x, y_bin in loaders.test:
            x = x.to(device)
            y_bin = y_bin.to(device)

            score, x_hat, pix, feat = anomaly_score(
                x, E, G, D, lambda_pix=args.lambda_pix, lambda_feat=args.lambda_feat
            )

            all_scores.append(score.cpu())
            all_labels.append(y_bin.cpu())

            # store a small subset for visualization
            if len(all_x) < 2:
                all_x.append(x.cpu())
                all_xhat.append(x_hat.cpu())

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()

    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")

    out_dir = Path("outputs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC (AUROC={auroc:.4f})")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "roc.png", dpi=150)
    plt.close()

    # PR curve
    prec, rec, _ = precision_recall_curve(labels, scores)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR (AUPRC={auprc:.4f})")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "pr.png", dpi=150)
    plt.close()

    # Save example reconstructions
    x0 = all_x[0]
    xhat0 = all_xhat[0]
    # get scores for this batch for titles
    with torch.no_grad():
        s0, _, _, _ = anomaly_score(x0.to(device), E, G, D, args.lambda_pix, args.lambda_feat)
    # we need labels too: re-load first batch labels quickly
    x_batch, y_batch = next(iter(loaders.test))
    save_grid_comparison(x_batch, xhat0, y_batch, s0.cpu(), out_dir / "examples.png", n_show=12)

    print("Saved: outputs/eval/roc.png, pr.png, examples.png")


if __name__ == "__main__":
    main()
