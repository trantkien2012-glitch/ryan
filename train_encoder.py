import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.data import get_mnist_loaders
from src.models import Generator, Discriminator, Encoder
from src.utils import get_device, set_seed, save_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--digit", type=int, default=0)
    parser.add_argument("--gan_ckpt", type=str, default=None)  # path to GAN checkpoint
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
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
    )

    # Models
    G = Generator(z_dim=args.z_dim).to(device)
    D = Discriminator().to(device)
    E = Encoder(z_dim=args.z_dim).to(device)

    # ---- Load GAN checkpoint (G + D) ----
    if args.gan_ckpt is None:
        raise ValueError("Please provide --gan_ckpt path to a trained GAN checkpoint .pt file")

    ckpt = torch.load(args.gan_ckpt, map_location=device)
    G.load_state_dict(ckpt["G_state"])
    D.load_state_dict(ckpt["D_state"])

    # Freeze G and D
    G.eval()
    D.eval()
    for p in G.parameters():
        p.requires_grad = False
    for p in D.parameters():
        p.requires_grad = False

    # Train only Encoder
    E.train()
    optE = optim.Adam(E.parameters(), lr=args.lr, betas=(0.5, 0.999))
    l1 = nn.L1Loss()

    out_dir = Path("outputs/encoder_samples")
    ckpt_dir = Path("checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_pix = 0.0
        total_feat = 0.0
        n_batches = 0

        for x, _ in loaders.train:  # train has only normal digit
            x = x.to(device)

            # Encode -> reconstruct
            z = E(x)
            x_hat = G(z)

            # Pixel loss
            loss_pix = l1(x_hat, x)

            # Feature loss (use discriminator features)
            _, f_real = D(x)
            _, f_fake = D(x_hat)
            loss_feat = l1(f_fake, f_real)

            loss = args.lambda_pix * loss_pix + args.lambda_feat * loss_feat

            optE.zero_grad()
            loss.backward()
            optE.step()

            total_loss += loss.item()
            total_pix += loss_pix.item()
            total_feat += loss_feat.item()
            n_batches += 1

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Loss={total_loss/max(1,n_batches):.4f} | "
            f"Pix={total_pix/max(1,n_batches):.4f} | "
            f"Feat={total_feat/max(1,n_batches):.4f}"
        )

        # Save a reconstruction grid
        with torch.no_grad():
            E.eval()
            x_vis, _ = next(iter(loaders.val))
            x_vis = x_vis.to(device)[:64]
            x_hat_vis = G(E(x_vis))
            # concat [real; recon] for easy visual comparison
            both = torch.cat([x_vis, x_hat_vis], dim=0)
            save_samples(both, str(out_dir / f"digit{args.digit}_epoch{epoch:03d}.png"), nrow=8)
            E.train()

        torch.save(
            {"epoch": epoch, "E_state": E.state_dict(), "args": vars(args)},
            ckpt_dir / f"encoder_digit{args.digit}_epoch{epoch:03d}.pt",
        )

    print("Done. Encoder checkpoints saved to checkpoints/ and recon grids to outputs/encoder_samples/")


if __name__ == "__main__":
    main()
