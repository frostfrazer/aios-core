"""
AIOS-Core :: Offline Training Loop
-------------------------------------
Trains NeuralScheduler on a synthetic labeled dataset derived from
scheduling domain knowledge:

  Rule 1 — CPU-bound process (feat[0] high, feat[1] low):
            nice_delta should be NEGATIVE (boost priority)
            cpu_weight HIGH, io_weight LOW, preempt_ms LOW

  Rule 2 — IO-bound process (feat[0] low, feat[2]+feat[3] high):
            nice_delta should be POSITIVE (yield CPU)
            cpu_weight LOW, io_weight HIGH, preempt_ms HIGH

  Rule 3 — Mixed process (both cpu and io moderate):
            nice_delta near 0, balanced weights, medium preempt

  Rule 4 — Idle process (feat[0] near 0, feat[1] near 0):
            nice_delta POSITIVE (lowest priority), preempt HIGH

We generate N samples per class, compute target outputs, run full
backprop (numpy-native), and save the trained weights.

Usage:
    python -m models.train
    python -m models.train --epochs 1000 --samples 2000 --lr 0.001
"""

import sys
import argparse
import json
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.neural_scheduler import NeuralScheduler, relu, tanh, sigmoid


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset generation
# ─────────────────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(0)

def _noise(shape, scale=0.05):
    return RNG.normal(0, scale, shape).astype(np.float32)

def make_dataset(n_per_class: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (X, Y) where:
      X : (N, 12)  feature vectors
      Y : (N, 4)   target outputs in [-1, 1] before scaling
          [cpu_weight_raw, io_weight_raw, nice_raw, preempt_raw]
    """
    X_list, Y_list = [], []

    for _ in range(n_per_class):
        # ── Class 1: CPU-bound ──────────────────────────────────────────────
        cpu_pct      = RNG.uniform(0.6, 1.0)
        io_rate      = RNG.uniform(0.0, 0.05)
        ctx_vol      = RNG.uniform(0.0, 0.1)
        ctx_nvol     = RNG.uniform(0.3, 0.8)   # lots of involuntary switches
        threads      = RNG.uniform(0.02, 0.2)
        x = np.array([
            cpu_pct, RNG.uniform(0,0.3), io_rate, io_rate*0.5,
            ctx_vol, ctx_nvol, threads,
            RNG.uniform(0,0.1), 1.0,  # status=running
            RNG.uniform(0.1, 1.0), 0.5, RNG.uniform(0.1,0.9)
        ], dtype=np.float32) + _noise(12, 0.03)
        x = np.clip(x, -1, 1)
        # Target: boost it — high cpu_weight, low io_weight, negative nice, low preempt
        y = np.array([0.8, -0.8, -0.7, -0.6], dtype=np.float32) + _noise(4, 0.05)
        X_list.append(x); Y_list.append(np.clip(y, -1, 1))

        # ── Class 2: IO-bound ───────────────────────────────────────────────
        cpu_pct  = RNG.uniform(0.0, 0.15)
        io_read  = RNG.uniform(0.4, 0.9)
        io_write = RNG.uniform(0.2, 0.7)
        ctx_vol  = RNG.uniform(0.5, 1.0)   # lots of voluntary yields
        ctx_nvol = RNG.uniform(0.0, 0.1)
        x = np.array([
            cpu_pct, RNG.uniform(0.1,0.6), io_read, io_write,
            ctx_vol, ctx_nvol, RNG.uniform(0.02,0.3),
            RNG.uniform(0.1,0.5), 0.5,   # status=sleeping
            RNG.uniform(0.1,1.0), 0.5, RNG.uniform(0.1,0.9)
        ], dtype=np.float32) + _noise(12, 0.03)
        x = np.clip(x, -1, 1)
        # Target: yield CPU — low cpu_weight, high io_weight, positive nice, high preempt
        y = np.array([-0.7, 0.8, 0.6, 0.7], dtype=np.float32) + _noise(4, 0.05)
        X_list.append(x); Y_list.append(np.clip(y, -1, 1))

        # ── Class 3: Mixed ──────────────────────────────────────────────────
        cpu_pct  = RNG.uniform(0.2, 0.6)
        io_rate  = RNG.uniform(0.1, 0.5)
        x = np.array([
            cpu_pct, RNG.uniform(0.1,0.4), io_rate, io_rate*0.7,
            RNG.uniform(0.2,0.6), RNG.uniform(0.1,0.4), RNG.uniform(0.05,0.4),
            RNG.uniform(0.05,0.3), RNG.uniform(0.3,1.0),
            RNG.uniform(0.2,1.0), 0.5, RNG.uniform(0.1,0.9)
        ], dtype=np.float32) + _noise(12, 0.03)
        x = np.clip(x, -1, 1)
        # Target: balanced
        y = np.array([0.1, 0.1, 0.0, 0.0], dtype=np.float32) + _noise(4, 0.08)
        X_list.append(x); Y_list.append(np.clip(y, -1, 1))

        # ── Class 4: Idle ───────────────────────────────────────────────────
        x = np.array([
            RNG.uniform(0,0.05), RNG.uniform(0,0.1),
            RNG.uniform(0,0.02), RNG.uniform(0,0.02),
            RNG.uniform(0,0.05), RNG.uniform(0,0.05),
            RNG.uniform(0.01,0.1), RNG.uniform(0,0.05),
            0.5,   # sleeping
            RNG.uniform(0.1,1.0), RNG.uniform(0.3,0.8), RNG.uniform(0.1,0.9)
        ], dtype=np.float32) + _noise(12, 0.02)
        x = np.clip(x, -1, 1)
        # Target: deprioritise — positive nice, high preempt, low weights
        y = np.array([-0.5, -0.3, 0.8, 0.8], dtype=np.float32) + _noise(4, 0.05)
        X_list.append(x); Y_list.append(np.clip(y, -1, 1))

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)

    # Shuffle
    idx = RNG.permutation(len(X))
    return X[idx], Y[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  Numpy backprop trainer
# ─────────────────────────────────────────────────────────────────────────────

def _forward_with_cache(w: dict, x: np.ndarray):
    """Returns (output, cache) where cache holds intermediates for backprop."""
    z1  = w["W1"] @ x  + w["b1"]
    h1  = relu(z1)
    z2  = w["W2"] @ h1 + w["b2"]
    h2  = relu(z2)
    z3  = w["W3"] @ h2 + w["b3"]
    out = tanh(z3)
    return out, (x, z1, h1, z2, h2, z3, out)


def _backward(w: dict, cache: tuple, target: np.ndarray, lr: float):
    """MSE loss backprop, in-place weight update."""
    x, z1, h1, z2, h2, z3, out = cache

    # Output layer — tanh derivative
    d_out = 2 * (out - target) / len(target)          # dL/d_out  (MSE)
    d_z3  = d_out * (1 - out**2)                      # tanh'

    dW3 = np.outer(d_z3, h2)
    db3 = d_z3

    # Hidden layer 2 — ReLU
    d_h2 = w["W3"].T @ d_z3
    d_z2 = d_h2 * (z2 > 0).astype(np.float32)

    dW2 = np.outer(d_z2, h1)
    db2 = d_z2

    # Hidden layer 1 — ReLU
    d_h1 = w["W2"].T @ d_z2
    d_z1 = d_h1 * (z1 > 0).astype(np.float32)

    dW1 = np.outer(d_z1, x)
    db1 = d_z1

    # Gradient descent step
    w["W3"] -= lr * dW3;  w["b3"] -= lr * db3
    w["W2"] -= lr * dW2;  w["b2"] -= lr * db2
    w["W1"] -= lr * dW1;  w["b1"] -= lr * db1


def train(
    model:     NeuralScheduler,
    X:         np.ndarray,
    Y:         np.ndarray,
    epochs:    int   = 500,
    lr:        float = 5e-3,
    batch:     int   = 32,
    val_split: float = 0.1,
    verbose:   bool  = True,
) -> list[float]:
    """
    Mini-batch SGD with MSE loss. Returns list of per-epoch val losses.
    """
    n      = len(X)
    n_val  = max(1, int(n * val_split))
    X_val, Y_val = X[:n_val], Y[:n_val]
    X_tr,  Y_tr  = X[n_val:], Y[n_val:]

    val_losses = []
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        # Shuffle training data
        idx = RNG.permutation(len(X_tr))
        X_tr, Y_tr = X_tr[idx], Y_tr[idx]

        # Mini-batch
        epoch_loss = 0.0
        n_batches  = 0
        for start in range(0, len(X_tr), batch):
            xb = X_tr[start:start+batch]
            yb = Y_tr[start:start+batch]
            batch_loss = 0.0
            # Accumulate gradients across batch
            grad_acc = {k: np.zeros_like(v) for k, v in model.weights.items()}
            for xi, yi in zip(xb, yb):
                out, cache = _forward_with_cache(model.weights, xi)
                batch_loss += float(np.mean((out - yi)**2))
                _backward(model.weights, cache, yi, lr / len(xb))
            epoch_loss += batch_loss / len(xb)
            n_batches  += 1

        # Validation loss
        val_loss = float(np.mean([
            np.mean((_forward_with_cache(model.weights, xi)[0] - yi)**2)
            for xi, yi in zip(X_val, Y_val)
        ]))
        val_losses.append(val_loss)

        if verbose and (epoch % 50 == 0 or epoch == 1):
            elapsed = time.perf_counter() - t0
            print(f"  epoch {epoch:4d}/{epochs}  "
                  f"train_loss={epoch_loss/n_batches:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"elapsed={elapsed:.1f}s")

    return val_losses


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Train NeuralScheduler offline")
    p.add_argument("--epochs",  type=int,   default=500)
    p.add_argument("--samples", type=int,   default=500,  help="Samples per class")
    p.add_argument("--lr",      type=float, default=5e-3)
    p.add_argument("--batch",   type=int,   default=32)
    p.add_argument("--out",     type=str,   default="weights/trained.json")
    args = p.parse_args()

    print(f"\n[Train] Generating dataset ({args.samples} samples/class x 4 classes)...")
    X, Y = make_dataset(args.samples)
    print(f"[Train] Dataset: X={X.shape}  Y={Y.shape}")

    model = NeuralScheduler()
    print(f"[Train] Starting training: epochs={args.epochs}  lr={args.lr}  batch={args.batch}\n")

    val_losses = train(model, X, Y, epochs=args.epochs, lr=args.lr, batch=args.batch)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)

    final_val = val_losses[-1]
    best_val  = min(val_losses)
    print(f"\n[Train] Done.  final_val_loss={final_val:.4f}  best={best_val:.4f}")
    print(f"[Train] Weights -> {args.out}")

    # Quick sanity check: run a CPU-bound feature vector through trained model
    print("\n[Train] Sanity check (CPU-bound process):")
    cpu_feat = np.array([0.9,0.1,0.02,0.01,0.05,0.6,0.1,0.05,1.0,0.5,0.5,0.5],
                        dtype=np.float32)
    result = model.predict(cpu_feat)
    print(f"  -> {result}")
    print(f"  Expected: nice_delta<0 (got {result['nice_delta']}), "
          f"cpu_weight>0.5 (got {result['cpu_weight']:.2f})")

    print("\n[Train] Sanity check (IO-bound process):")
    io_feat = np.array([0.05,0.2,0.7,0.5,0.8,0.05,0.1,0.2,0.5,0.5,0.5,0.5],
                       dtype=np.float32)
    result = model.predict(io_feat)
    print(f"  -> {result}")
    print(f"  Expected: nice_delta>0 (got {result['nice_delta']}), "
          f"io_weight>0.5 (got {result['io_weight']:.2f})")


if __name__ == "__main__":
    main()
