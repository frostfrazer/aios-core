"""
AIOS-Core :: Neural Scheduler Model
------------------------------------
Pure-numpy feed-forward network for on-device scheduler inference.
Architecture: 12 → 32 → 16 → 4 output heads
Outputs: [cpu_weight, io_weight, nice_delta, preempt_threshold]

No external ML dependencies by design — this is meant to run in a
constrained kernel-adjacent environment (future: port to C/Rust).
"""

import numpy as np
import json
import os
from pathlib import Path


# ─────────────────────────────────────────────
#  Activations
# ─────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# ─────────────────────────────────────────────
#  Model definition
# ─────────────────────────────────────────────

class NeuralScheduler:
    """
    Lightweight inference-only neural network.
    Input: 12-dim feature vector from telemetry window
    Output: 4-dim tuning vector (all values in [-1, 1] before scaling)

    Layer layout:
      dense(12→32, relu) → dense(32→16, relu) → dense(16→4, tanh)
    """

    INPUT_DIM  = 12
    HIDDEN1    = 32
    HIDDEN2    = 16
    OUTPUT_DIM = 4   # [cpu_weight, io_weight, nice_delta, preempt_ms]

    def __init__(self):
        self.weights = self._init_weights()

    def _init_weights(self) -> dict:
        """Xavier init — good enough for pre-training baseline."""
        rng = np.random.default_rng(42)

        def xavier(fan_in, fan_out):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, (fan_out, fan_in)).astype(np.float32)

        return {
            "W1": xavier(self.INPUT_DIM, self.HIDDEN1),
            "b1": np.zeros(self.HIDDEN1, dtype=np.float32),
            "W2": xavier(self.HIDDEN1, self.HIDDEN2),
            "b2": np.zeros(self.HIDDEN2, dtype=np.float32),
            "W3": xavier(self.HIDDEN2, self.OUTPUT_DIM),
            "b3": np.zeros(self.OUTPUT_DIM, dtype=np.float32),
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: shape (INPUT_DIM,) — already normalised by telemetry pipeline.
        Returns: shape (OUTPUT_DIM,) in [-1, 1].
        """
        assert x.shape == (self.INPUT_DIM,), f"Expected ({self.INPUT_DIM},), got {x.shape}"
        h1  = relu(self.weights["W1"] @ x  + self.weights["b1"])
        h2  = relu(self.weights["W2"] @ h1 + self.weights["b2"])
        out = tanh(self.weights["W3"] @ h2 + self.weights["b3"])
        return out

    def predict(self, features: np.ndarray) -> dict:
        """
        High-level API: raw features → named tuning parameters.
        Scales raw [-1,1] outputs to meaningful OS parameter ranges.
        """
        raw = self.forward(features.astype(np.float32))
        return {
            "cpu_weight":       float(np.clip((raw[0] + 1) / 2, 0.0, 1.0)),    # [0, 1]
            "io_weight":        float(np.clip((raw[1] + 1) / 2, 0.0, 1.0)),    # [0, 1]
            "nice_delta":       int(np.round(raw[2] * 10)),                     # [-10, +10]
            "preempt_ms":       float(np.clip((raw[3] + 1) * 50, 1.0, 100.0)), # [1ms, 100ms]
        }

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        """Serialise weights to JSON (portable, kernel-readable later)."""
        serialisable = {k: v.tolist() for k, v in self.weights.items()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2)
        print(f"[NeuralScheduler] Weights saved -> {path}")

    def load(self, path: str):
        """Load weights from JSON."""
        with open(path) as f:
            raw = json.load(f)
        self.weights = {k: np.array(v, dtype=np.float32) for k, v in raw.items()}
        print(f"[NeuralScheduler] Weights loaded <- {path}")

    # ── Naive online update (RL-lite, no backprop lib needed) ────────────────

    def apply_reward(self, features: np.ndarray, reward: float, lr: float = 1e-3):
        """
        Dead-simple policy-gradient-style weight nudge.
        reward > 0 → reinforce last action, reward < 0 → suppress.
        This is NOT full backprop — it's a placeholder for the real
        RL training loop we'll bolt on in Phase 2.
        """
        x   = features.astype(np.float32)
        h1  = relu(self.weights["W1"] @ x  + self.weights["b1"])
        h2  = relu(self.weights["W2"] @ h1 + self.weights["b2"])
        out = tanh(self.weights["W3"] @ h2 + self.weights["b3"])

        # Gradient of tanh output w.r.t. W3 (simplified, first-order only)
        grad_out  = reward * (1 - out**2)
        self.weights["W3"] += lr * np.outer(grad_out, h2)
        self.weights["b3"] += lr * grad_out
