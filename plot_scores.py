import argparse
import matplotlib.pyplot as plt
import torch

from src.data import get_mnist_loaders
from src.models import Generator, Discriminator, Encoder
from src.score import anomaly_score
from src.utils import get_device, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--digit", type=int, default=0)
    p.add_argument("--gan_ckpt", type=str, required=True)
    p.add_argument("--enc_ckpt", type=str, required=True)
    args = p.parse_args()

    set_seed(42)
    device = get_device()

    loaders = get_mnist_loaders(normal_digit=args.digit, batch_size=256)

    G = Generator().to(device)
    D = Discriminator().to(device)
    E = Encoder().to(device)

    gan = torch.load(args.gan_ckpt, map_location=device)
    enc = torch.load(args.enc_ckpt, map_location=device)
    G.load_state_dict(gan["G_state"])
    D.load_state_dict(gan["D_state"])
    E.load_state_dict(enc["E_state"])
    G.eval(); D.eval(); E.eval()

    scores_all = []
    labels_all = []

    with torch.no_grad():
        for x, y in loaders.test:
            x = x.to(device)
            s, _, _, _ = anomaly_score(x, E, G, D)
            scores_all.append(s.cpu())
            labels_all.append(y)

    scores = torch.cat(scores_all).numpy()
    labels = torch.cat(labels_all).numpy()

    s_norm = scores[labels == 0]
    s_anom = scores[labels == 1]

    plt.figure()
    plt.hist(s_norm, bins=50, alpha=0.6, label="normal (0)")
    plt.hist(s_anom, bins=50, alpha=0.6, label="anomaly (non-0)")
    plt.legend()
    plt.xlabel("Anomaly score")
    plt.ylabel("Count")
    plt.title("Score distributions")
    plt.show()


if __name__ == "__main__":
    main()
