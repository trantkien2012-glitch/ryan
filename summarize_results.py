import argparse
import csv
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> str:
    """Run a command and return stdout as text (raises on error)."""
    print("\n>>>", " ".join(cmd))
    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if p.stdout:
        print(p.stdout.strip())
    if p.stderr:
        # Many libraries print warnings to stderr; show it but don't crash (already check=True)
        print(p.stderr.strip())
    return (p.stdout or "") + "\n" + (p.stderr or "")


def parse_metrics(output: str):
    """Extract AUROC and AUPRC from eval output."""
    auroc = None
    auprc = None
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("AUROC:"):
            auroc = float(line.split("AUROC:")[1].strip())
        if line.startswith("AUPRC:"):
            auprc = float(line.split("AUPRC:")[1].strip())
    return auroc, auprc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--digits", type=int, nargs="+", default=[0, 1, 8])
    parser.add_argument("--gan_epochs", type=int, default=20)
    parser.add_argument("--enc_epochs", type=int, default=10)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--batch_size_gan", type=int, default=128)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    # scoring weights (you can tune this)
    parser.add_argument("--lambda_pix", type=float, default=1.0)
    parser.add_argument("--lambda_feat", type=float, default=0.1)
    parser.add_argument("--lambda_feat_grid", type=float, nargs="*", default=None)

    args = parser.parse_args()

    ckpt_dir = Path("checkpoints")
    out_dir = Path("outputs/summary")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for d in args.digits:
        gan_ckpt = ckpt_dir / f"gan_digit{d}_epoch{args.gan_epochs:03d}.pt"
        enc_ckpt = ckpt_dir / f"encoder_digit{d}_epoch{args.enc_epochs:03d}.pt"

        # 1) Train GAN if needed
        if not gan_ckpt.exists():
            run([
                "python", "-m", "src.train_gan",
                "--digit", str(d),
                "--epochs", str(args.gan_epochs),
                "--batch_size", str(args.batch_size_gan),
                "--z_dim", str(args.z_dim),
                "--seed", str(args.seed),
            ])
        else:
            print(f"\n[skip] GAN checkpoint exists: {gan_ckpt}")

        # 2) Train Encoder if needed
        if not enc_ckpt.exists():
            run([
                "python", "-m", "src.train_encoder",
                "--digit", str(d),
                "--gan_ckpt", str(gan_ckpt),
                "--epochs", str(args.enc_epochs),
                "--batch_size", str(args.batch_size_gan),
                "--z_dim", str(args.z_dim),
                "--seed", str(args.seed),
            ])
        else:
            print(f"\n[skip] Encoder checkpoint exists: {enc_ckpt}")

        # 3) Evaluate
                # 3) Evaluate (optionally sweep lambda_feat)
        lambda_grid = args.lambda_feat_grid or [args.lambda_feat]

        best = None  # (auroc, auprc, lambda_feat)

        for lf in lambda_grid:
            eval_out = run([
                "python", "-m", "src.eval",
                "--digit", str(d),
                "--gan_ckpt", str(gan_ckpt),
                "--enc_ckpt", str(enc_ckpt),
                "--batch_size", str(args.batch_size_eval),
                "--z_dim", str(args.z_dim),
                "--lambda_pix", str(args.lambda_pix),
                "--lambda_feat", str(lf),
                "--seed", str(args.seed),
            ])

            auroc, auprc = parse_metrics(eval_out)
            if auroc is None or auprc is None:
                raise RuntimeError(f"Could not parse AUROC/AUPRC for digit {d}. Output:\n{eval_out}")

            if (best is None) or (auroc > best[0]):
                best = (auroc, auprc, lf)

        auroc, auprc, best_lf = best

        results.append({
            "digit": d,
            "gan_ckpt": str(gan_ckpt),
            "enc_ckpt": str(enc_ckpt),
            "lambda_pix": args.lambda_pix,
            "lambda_feat": best_lf,   # store BEST for this digit
            "AUROC": auroc,
            "AUPRC": auprc,
        })

    # Save CSV
    csv_path = out_dir / "results_0_1_8.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Print table
    print("\n=== Summary Results ===")
    print(f"Saved CSV: {csv_path}")
    print(f"{'digit':<6} {'AUROC':<10} {'AUPRC':<10} {'lambda_feat':<12} {'gan_epochs':<10} {'enc_epochs':<10}")
    for r in results:
        print(f"{r['digit']:<6} {r['AUROC']:<10.4f} {r['AUPRC']:<10.4f} {r['lambda_feat']:<12.3f} {args.gan_epochs:<10} {args.enc_epochs:<10}")


if __name__ == "__main__":
    main()
