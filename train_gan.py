import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.data import get_mnist_loaders
from src.models import Generator, Discriminator
from src.utils import get_device, save_samples, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--digit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    loaders = get_mnist_loaders(
        normal_digit=args.digit,
        batch_size=args.batch_size,
        val_size=2000,  # not used here, but OK
    )

    G = Generator(z_dim=args.z_dim).to(device)
    D = Discriminator().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optG = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optD = optim.Adam(D.parameters(), lr=args.lr * 0.5, betas=(args.beta1, 0.999))

    fixed_z = torch.randn(64, args.z_dim, 1, 1, device=device)

    out_dir = Path("outputs/samples")
    ckpt_dir = Path("checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()

        d_loss_running = 0.0
        g_loss_running = 0.0
        n_batches = 0

        for x_real, _ in loaders.train:
            x_real = x_real.to(device)
            bs = x_real.size(0)

            # -------------------
            # Train Discriminator
            # -------------------
            z = torch.randn(bs, args.z_dim, 1, 1, device=device)
            x_fake = G(z).detach()

            real_logits, _ = D(x_real)
            fake_logits, _ = D(x_fake)

            real_labels = torch.full((bs,), 0.9, device=device)  # real = 0.9 (label smoothing)
            fake_labels = torch.zeros(bs, device=device)

            d_loss = criterion(real_logits, real_labels) + criterion(fake_logits, fake_labels)

            optD.zero_grad()
            d_loss.backward()
            optD.step()

            # -------------
            # Train Generator
            # -------------
            z = torch.randn(bs, args.z_dim, 1, 1, device=device)
            x_fake = G(z)
            fake_logits, _ = D(x_fake)

            # Generator wants discriminator to predict "real" for fake images
            g_loss = criterion(fake_logits, torch.ones(bs, device=device))

            optG.zero_grad()
            g_loss.backward()
            optG.step()

            d_loss_running += d_loss.item()
            g_loss_running += g_loss.item()
            n_batches += 1

        d_loss_epoch = d_loss_running / max(1, n_batches)
        g_loss_epoch = g_loss_running / max(1, n_batches)
        print(f"Epoch {epoch:03d}/{args.epochs} | D_loss={d_loss_epoch:.4f} | G_loss={g_loss_epoch:.4f}")

        # Save sample grid
        with torch.no_grad():
            G.eval()
            x_sample = G(fixed_z)
            save_samples(x_sample, str(out_dir / f"digit{args.digit}_epoch{epoch:03d}.png"))

        # Save checkpoint every epoch
        torch.save(
            {
                "epoch": epoch,
                "G_state": G.state_dict(),
                "D_state": D.state_dict(),
                "args": vars(args),
            },
            ckpt_dir / f"gan_digit{args.digit}_epoch{epoch:03d}.pt",
        )

    print("Done. Samples saved to outputs/samples/ and checkpoints saved to checkpoints/")


if __name__ == "__main__":
    main()
