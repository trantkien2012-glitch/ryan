import torch
import torch.nn.functional as F


@torch.no_grad()
def anomaly_score(
    x: torch.Tensor,
    E,
    G,
    D,
    lambda_pix: float = 1.0,
    lambda_feat: float = 0.1,
) -> torch.Tensor:
    """
    x: (N,1,28,28) normalized to [-1,1]
    returns: (N,) anomaly scores (higher = more anomalous)
    """
    z = E(x)
    x_hat = G(z)

    # Pixel residual (L1)
    pix = torch.mean(torch.abs(x - x_hat), dim=(1, 2, 3))  # (N,)

    # Feature residual (L1) from discriminator features
    _, f_real = D(x)
    _, f_fake = D(x_hat)
    feat = torch.mean(torch.abs(f_real - f_fake), dim=(1, 2, 3))  # (N,)

    score = lambda_pix * pix + lambda_feat * feat
    return score, x_hat, pix, feat
