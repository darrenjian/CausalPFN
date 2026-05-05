"""
lora.py — LoRA Adapter Injection for CausalPFN's Transformer Backbone

LoRA (Low-Rank Adaptation) inserts trainable low-rank matrices alongside
frozen attention projections. This lets the backbone adapt to your encoder's
output distribution without destroying its pre-trained causal reasoning.

How it works:
    Original forward:   output = W @ input          (W frozen, large)
    LoRA forward:       output = W @ input + B @ A @ input
                               (A: r x d_in, B: d_out x r, both trained)

    B @ A is initialized to zero, so training starts from pre-trained behavior.
    As training proceeds, B @ A learns a small correction — a dialect adapter
    that teaches the backbone to interpret your encoder's embeddings.

We inject LoRA into kv_proj, q_proj, and out_proj of all 20 transformer layers.

Trainable parameter count (rank=8):
    kv_proj (384->768):   per layer = 384*8 + 8*768  = 9,216  | total = 184,320
    q_proj  (384->1152):  per layer = 384*8 + 8*1152 = 12,288 | total = 245,760
    out_proj (384->384):  per layer = 384*8 + 8*384  = 6,144  | total = 122,880
    Grand total LoRA: ~552,960 parameters

Usage:
    from lora import inject_lora, print_parameter_summary
    inject_lora(icl_model, rank=8)
    icl_model = icl_model.to(device)  # MUST call .to(device) AFTER inject_lora
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds a trainable low-rank adapter.

    The base layer is frozen. Only lora_A and lora_B are trained.
    B @ A is initialized to zero so the adapter starts as a no-op.

    Parameters
    ----------
    original_layer : nn.Linear
        The pre-trained linear layer to wrap. Its weights are frozen.
    rank : int
        Rank of the low-rank decomposition. r=8 is standard.
    alpha : float
        LoRA scaling factor. Output is scaled by alpha/rank.
        Setting alpha=rank gives scaling=1.0.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 8.0,
    ):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank

        # Keep the entire original layer as a submodule.
        # This means .to(device) will correctly move it along with lora_A/B.
        self.base_layer = original_layer

        # Freeze the base layer weights
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # LoRA matrices — these are the only trainable parameters
        # A: projects input down to rank, B: projects back up to output dim
        # Initialized so B @ A = 0 at start of training
        self.lora_A = nn.Parameter(
            torch.empty(rank, original_layer.in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(original_layer.out_features, rank)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        output = base_layer(x) + scaling * B @ A @ x

        The base layer handles pre-trained behavior (frozen).
        The LoRA term provides the adaptation (trained).
        """
        # Frozen base computation
        result = self.base_layer(x)

        # LoRA adaptation — lora_A and lora_B will be on the correct device
        # as long as icl_model.to(device) is called after inject_lora()
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return result + self.scaling * lora_out


def inject_lora(
    icl_model: nn.Module,
    rank: int = 8,
    alpha: float = 8.0,
    target_layers: List[str] = None,
) -> None:
    """
    Inject LoRA adapters into CausalPFN's transformer layers IN PLACE.

    Steps:
    1. Walk all named modules in icl_model
    2. Replace target nn.Linear layers with LoRALinear wrappers
    3. Freeze all non-LoRA parameters

    IMPORTANT: Call icl_model.to(device) AFTER this function.
    The LoRA parameter tensors are created on CPU — .to(device) moves
    everything (base weights + LoRA adapters) to the correct device.

    Parameters
    ----------
    icl_model : nn.Module
        The loaded CausalPFN InContextModel (est.icl_model).
    rank : int
        LoRA rank. Default 8.
    alpha : float
        LoRA scaling. Default 8.0.
    target_layers : list of str
        Layer name substrings to target. Default: kv_proj, q_proj, out_proj.
    """
    if target_layers is None:
        target_layers = ['kv_proj', 'out_proj', 'q_proj']

    # Walk the module tree and replace target layers
    n_injected = 0
    for module_name, module in list(icl_model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        # Check if this layer's name matches any target
        layer_attr = module_name.split(".")[-1]
        if not any(t == layer_attr for t in target_layers):
            continue

        # Navigate to the parent module
        parts = module_name.split(".")
        parent = icl_model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        # Replace with LoRA wrapper
        setattr(parent, parts[-1], LoRALinear(module, rank=rank, alpha=alpha))
        n_injected += 1

    if n_injected == 0:
        raise RuntimeError(
            "No layers were injected. Check that target_layers match "
            f"actual layer names in the model. Got: {target_layers}"
        )

    # Freeze all non-LoRA parameters
    for name, param in icl_model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            param.requires_grad = False

    print(f"LoRA injection complete: {n_injected} layers adapted "
          f"(rank={rank}, alpha={alpha})")
    print("NOTE: Call icl_model.to(device) after this to move LoRA params to GPU.")


def print_parameter_summary(icl_model: nn.Module, encoder: nn.Module) -> None:
    """Print a summary of trainable vs frozen parameters."""

    def count_trainable(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    def count_frozen(m):
        return sum(p.numel() for p in m.parameters() if not p.requires_grad)

    lora_trainable = count_trainable(icl_model)
    lora_frozen = count_frozen(icl_model)
    enc_trainable = count_trainable(encoder)

    total_trainable = lora_trainable + enc_trainable
    total_all = lora_frozen + total_trainable

    print("=" * 55)
    print("Parameter Summary")
    print("=" * 55)
    print(f"{'CausalPFN backbone (frozen):':<35} {lora_frozen:>10,}")
    print(f"{'CausalPFN LoRA adapters (trainable):':<35} {lora_trainable:>10,}")
    print(f"{'Temporal encoder (trainable):':<35} {enc_trainable:>10,}")
    print("-" * 55)
    print(f"{'Total trainable:':<35} {total_trainable:>10,}")
    print(f"{'Total parameters:':<35} {total_all:>10,}")
    print(f"{'Trainable fraction:':<35} {100 * total_trainable / total_all:>9.1f}%")
    print("=" * 55)
