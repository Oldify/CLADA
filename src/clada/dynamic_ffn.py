import torch
from typing import Set

class DynamicFFNModule(torch.nn.Module):
    """easy switching of active neurons."""

    def __init__(self, original_module, module_type, device, dtype):

        super().__init__()
        self.original_module = original_module
        self.module_type = module_type
        self.device = device
        self.dtype = dtype
        self.active_indices = None
        self.reduced_module = None
        self.using_reduced = False

        # Cache all potentially used neurons (for quick switching).
        self.cached_modules = {}

    def update_active_neurons(self, active_indices: Set[int]):
        """Dynamically update active neurons."""

        # Convert the set to a sorted array and generate a unique key.
        sorted_indices = sorted(list(active_indices))
        key = hash(tuple(sorted_indices))

        # Check if it already exists in the cache.
        if key in self.cached_modules:
            self.reduced_module = self.cached_modules[key]
        else:
            # Create a new reduced module.
            self.reduced_module = self._create_reduced_module(sorted_indices)
            # Add to the cache.
            self.cached_modules[key] = self.reduced_module

        self.active_indices = active_indices
        self.using_reduced = True

    def use_original(self):
        """Go back to original module"""
        self.using_reduced = False

    def forward(self, x):
        """During forward propagation, use the original or reduced module based on the settings."""
        if self.using_reduced and self.reduced_module is not None:
            return self.reduced_module(x)
        else:
            return self.original_module(x)

    def _create_reduced_module(self, sorted_indices):
        """
        Create reduced FFN module.
        You can change these code based on your own model structure.
        """
        if self.module_type == "gpt2_mlp":
            # GPT-2 MLP module
            original_fc1 = self.original_module.fc1
            original_fc2 = self.original_module.fc2

            intermediate_size = len(sorted_indices)

            # Create reduced fc1 (in_features -> reduced_features)
            reduced_fc1 = torch.nn.Linear(
                original_fc1.in_features,
                intermediate_size,
                bias=original_fc1.bias is not None
            ).to(device=self.device, dtype=self.dtype)

            # Copy weights and bias
            reduced_fc1.weight.data = original_fc1.weight.data[sorted_indices, :]
            if original_fc1.bias is not None:
                reduced_fc1.bias.data = original_fc1.bias.data[sorted_indices]

            # Create reduced fc2 (reduced_features -> out_features)
            reduced_fc2 = torch.nn.Linear(
                intermediate_size,
                original_fc2.out_features,
                bias=original_fc2.bias is not None
            ).to(device=self.device, dtype=self.dtype)

            # Copy
            reduced_fc2.weight.data = original_fc2.weight.data[:, sorted_indices]
            if original_fc2.bias is not None:
                reduced_fc2.bias.data = original_fc2.bias.data.clone()

            # New MLP
            class ReducedMLP(torch.nn.Module):
                def __init__(self, fc1, fc2, act):
                    super().__init__()
                    self.fc1 = fc1
                    self.fc2 = fc2
                    self.act = act

                def forward(self, x):
                    return self.fc2(self.act(self.fc1(x)))

            # Use same activation function
            activation_fn = self.original_module.act
            return ReducedMLP(reduced_fc1, reduced_fc2, activation_fn)

        elif self.module_type == "llama_mlp":
            # LLaMA MLP module.
            original_gate_proj = self.original_module.gate_proj
            original_up_proj = self.original_module.up_proj
            original_down_proj = self.original_module.down_proj

            intermediate_size = len(sorted_indices)

            # Create reduced projector
            reduced_gate_proj = torch.nn.Linear(
                original_gate_proj.in_features,
                intermediate_size,
                bias=original_gate_proj.bias is not None
            ).to(device=self.device, dtype=self.dtype)

            reduced_up_proj = torch.nn.Linear(
                original_up_proj.in_features,
                intermediate_size,
                bias=original_up_proj.bias is not None
            ).to(device=self.device, dtype=self.dtype)

            reduced_down_proj = torch.nn.Linear(
                intermediate_size,
                original_down_proj.out_features,
                bias=original_down_proj.bias is not None
            ).to(device=self.device, dtype=self.dtype)

            # Copy weights
            reduced_gate_proj.weight.data = original_gate_proj.weight.data[sorted_indices, :]
            reduced_up_proj.weight.data = original_up_proj.weight.data[sorted_indices, :]
            reduced_down_proj.weight.data = original_down_proj.weight.data[:, sorted_indices]

            # Copy bias if we had one
            if original_gate_proj.bias is not None:
                reduced_gate_proj.bias.data = original_gate_proj.bias.data[sorted_indices]
            if original_up_proj.bias is not None:
                reduced_up_proj.bias.data = original_up_proj.bias.data[sorted_indices]
            if original_down_proj.bias is not None:
                reduced_down_proj.bias.data = original_down_proj.bias.data.clone()

            # New reduced MLP
            class ReducedLLaMaMLP(torch.nn.Module):
                def __init__(self, gate_proj, up_proj, down_proj, act_fn):
                    super().__init__()
                    self.gate_proj = gate_proj
                    self.up_proj = up_proj
                    self.down_proj = down_proj
                    self.act_fn = act_fn

                def forward(self, x):
                    # SwiGLU
                    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

            # Same activation function
            act_fn = self.original_module.act_fn if hasattr(self.original_module, 'act_fn') else torch.nn.functional.silu

            return ReducedLLaMaMLP(reduced_gate_proj, reduced_up_proj, reduced_down_proj, act_fn)

        # Default to return a copy of original module
        return self.original_module
