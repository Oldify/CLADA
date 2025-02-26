import os
import gc
import time
import h5py
import json
import random
import logging
import numpy as np
from functools import lru_cache
from tqdm import tqdm
from typing import List, Dict, Optional, Set

import matplotlib.pyplot as plt

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from .config import ConditionalComputationConfig
from .data_loader import PromptDataLoader
from .dynamic_ffn import DynamicFFNModule

logger = logging.getLogger(__name__)

class ConditionalComputationModel:
    """Yep...the computation"""

    def __init__(
            self,
            config: ConditionalComputationConfig,
            device: Optional[str] = None
    ):

        self.config = config

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Use device: {self.device}")

        # Output dir
        os.makedirs(config.output_dir, exist_ok=True)

        # type
        self.dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64
        }
        self.dtype = self.dtype_map.get(config.precision, torch.float32)

        # Initialize model and tokenizer
        self._initialize_model()

        # activation matrices and cache
        self.ffn_activations = {}
        self.active_neurons = {}
        self.dynamic_ffn_modules = {}
        self.original_ffn_modules = {}
        self.activation_hooks = []

        # metrics
        self.surprisal_history = []
        self.entropy_history = []
        self.surprisal_threshold_value = 0.0
        self.entropy_threshold_value = 0.0

        # Dataloader
        self.data_loader = PromptDataLoader(config, self.tokenizer)

        # register to get FFN
        self._register_hooks()

        logger.info("Complete conditional computation model initialization.")

    def _initialize_model(self):
        """init model and tokenizer"""
        model_path = self.config.model_path

        # check
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        logger.info(f"Loading model and tokenizer: {model_path}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Read config to get model type
            model_config = AutoConfig.from_pretrained(model_path)
            self.model_type = getattr(model_config, "model_type", "unknown")
            logger.info(f"Found model type: {self.model_type}")

            # Load model according to memory mode
            if self.config.low_memory_mode:
                try:
                    logger.info("Loading model in 8-bit quantization mode...")
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=0.0
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map=self.device,
                        quantization_config=quantization_config,
                        torch_dtype=self.dtype
                    )
                except ImportError:
                    logger.warning("Bitsandbytes is unavailable. Loading model in standard mode.")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=self.dtype
                    ).to(self.device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=self.dtype
                ).to(self.device)

            self.model.eval()
            self.vocab_size = len(self.tokenizer)

            logger.info(f"Model loaded. Vocab size is: {self.vocab_size}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _analyze_model_structure(self):
        """In order to locate FFN"""

        self.ffn_paths = self._get_ffn_paths()
        logger.info(f"Found {len(self.ffn_paths)} FFN modules.")

        # Save original FFN
        self.ffn_info = {}
        for path in self.ffn_paths:
            ffn_module = self._get_module_by_path(path)

            if self.model_type == "gpt2" and hasattr(ffn_module, 'fc1') and hasattr(ffn_module, 'fc2'):
                # GPT-2 MLP
                self.ffn_info[path] = {
                    'type': 'gpt2_mlp',
                    'fc1_in': ffn_module.fc1.in_features,
                    'fc1_out': ffn_module.fc1.out_features,
                    'fc2_in': ffn_module.fc2.in_features,
                    'fc2_out': ffn_module.fc2.out_features
                }
            elif self.model_type == "llama" and hasattr(ffn_module, 'gate_proj') and hasattr(ffn_module,
                                                                                             'down_proj'):
                # LLaMA MLP
                self.ffn_info[path] = {
                    'type': 'llama_mlp',
                    'gate_in': ffn_module.gate_proj.in_features,
                    'gate_out': ffn_module.gate_proj.out_features,
                    'up_in': ffn_module.up_proj.in_features,
                    'up_out': ffn_module.up_proj.out_features,
                    'down_in': ffn_module.down_proj.in_features,
                    'down_out': ffn_module.down_proj.out_features
                }
            else:
                logger.warning(f"Unknown FFN structure: {path}")
                # Attempting to find a structure similar to FFN.
                for name, submodule in ffn_module.named_modules():
                    if isinstance(submodule, torch.nn.Linear):
                        logger.info(f"Submodule: {name}, in={submodule.in_features}, out={submodule.out_features}")

    def _get_ffn_paths(self) -> List[str]:
        """Obtain the path to the FFN layer based on the model type."""
        ffn_paths = []

        if self.model_type == "gpt2":
            # GPT-2
            num_layers = self.model.config.n_layer
            for i in range(num_layers):
                ffn_paths.append(f"transformer.h.{i}.mlp")

        elif self.model_type == "llama":
            # LLaMA
            num_layers = self.model.config.num_hidden_layers
            for i in range(num_layers):
                ffn_paths.append(f"model.layers.{i}.mlp")

        elif self.model_type == "opt":
            # OPT
            num_layers = self.model.config.num_hidden_layers
            for i in range(num_layers):
                ffn_paths.append(f"model.decoder.layers.{i}.fc1")
                ffn_paths.append(f"model.decoder.layers.{i}.fc2")

        elif self.model_type == "gpt_neox":
            # GPT-NeoX
            num_layers = self.model.config.num_hidden_layers
            for i in range(num_layers):
                ffn_paths.append(f"gpt_neox.layers.{i}.mlp")

        elif self.model_type == "falcon":
            # Falcon
            num_layers = self.model.config.num_hidden_layers
            for i in range(num_layers):
                ffn_paths.append(f"transformer.h.{i}.mlp")

        else:
            # Find a general FFN module
            logger.warning(f"Unknown model type: {self.model_type}, try to find one FFN-like module.")

            def search_modules(module, path=""):
                found_paths = []
                for name, child in module.named_children():
                    child_path = f"{path}.{name}" if path else name

                    # check the name
                    if name in ["mlp", "feed_forward", "ffn", "MLP"]:
                        found_paths.append(child_path)

                    found_paths.extend(search_modules(child, child_path))

                return found_paths

            ffn_paths = search_modules(self.model)

            if not ffn_paths:
                logger.warning("Unable to automatically identify the FFN layer; conditional computation may not function properly.")

        return ffn_paths

    def _get_module_by_path(self, path: str):

        parts = path.split('.')
        module = self.model
        for part in parts:
            module = getattr(module, part)
        return module

    def _set_module_by_path(self, path: str, new_module):

        parts = path.split('.')
        parent_path = '.'.join(parts[:-1])
        last_part = parts[-1]

        if parent_path:
            parent = self._get_module_by_path(parent_path)
            setattr(parent, last_part, new_module)
        else:
            setattr(self.model, last_part, new_module)

    def _register_hooks(self):
        """Get FFN activation matrix"""

        self._analyze_model_structure()

        self.activation_hooks = []

        for path, info in self.ffn_info.items():
            module = self._get_module_by_path(path)

            # Save original module
            self.original_ffn_modules[path] = module

            # Create dynamicFFN
            dynamic_module = DynamicFFNModule(
                original_module=module,
                module_type=info['type'],
                device=self.device,
                dtype=self.dtype
            )

            # Replace
            self._set_module_by_path(path, dynamic_module)
            self.dynamic_ffn_modules[path] = dynamic_module

            # Hooks
            def get_activation_hook(layer_path):
                def hook(module, input, output):
                    # Save FFN activations
                    self.ffn_activations[layer_path] = input[0].detach()

                return hook

            # 注册钩子
            if info['type'] == "gpt2_mlp":
                # For GPT-2's MLP，hook fc1
                hook = module.fc1.register_forward_hook(get_activation_hook(path))
            elif info['type'] == "llama_mlp":
                # For LLaMA MLP，hook gate_proj
                hook = module.gate_proj.register_forward_hook(get_activation_hook(path))
            else:
                # For others, hook whatever provided
                hook = module.register_forward_hook(get_activation_hook(path))

            self.activation_hooks.append(hook)

        logger.info(f"Registered {len(self.activation_hooks)} activation hooks.")

    @lru_cache(maxsize=128)
    def _compute_token_metrics(self, logits: torch.Tensor, token_id: int) -> Dict:
        """Calculate the surprisal and entropy for a single token.

        Args:
            logits: logits from model output
            token_id: token id of current token

        Returns:
            a dict
        """
        # Convert logits to a probability distribution.
        probs = torch.softmax(logits, dim=-1).cpu()

        # Obtain the probability of the current token.
        token_prob = probs[token_id].item()

        # Calculate surprisal: \(-\log_2(p)\).
        surprisal = -np.log2(max(token_prob, 1e-10))

        # Calculate entropy: \(-\sum (p \cdot \log_2(p))\).
        valid_probs = probs[probs > 0]
        entropy = -torch.sum(valid_probs * torch.log2(valid_probs)).item()

        # Calculate normalization metrics relative to the vocabulary.
        # Different from paper here.
        normalized_surprisal = surprisal / np.log2(self.vocab_size)
        normalized_entropy = entropy / np.log2(self.vocab_size)

        return {
            "token_id": token_id,
            "token": self.tokenizer.decode([token_id]),
            "probability": token_prob,
            "surprisal": surprisal,
            "entropy": entropy,
            "normalized_surprisal": normalized_surprisal,
            "normalized_entropy": normalized_entropy
        }

    def _select_top_k_neurons(self, activations: Dict[str, torch.Tensor]) -> Dict[str, Set[int]]:
        """
        Select the top-k neurons based on FFN activation values.
        Here, we draw on the implementation from Griffin(Dong2024).
        """
        active_neurons = {}

        for path, activation in activations.items():
            # Make sure there's a activation
            if activation is None or activation.size(0) == 0:
                logger.warning(f"{path} does not has activation.")
                continue

            # Get FFN type
            ffn_type = self.ffn_info[path]['type']

            # Get activation dimension
            if ffn_type == "gpt2_mlp":
                intermediate_dim = self.ffn_info[path]['fc1_out']
            elif ffn_type == "llama_mlp":
                intermediate_dim = self.ffn_info[path]['gate_out']
            else:
                logger.warning(f"Unknown {ffn_type}, use activation size for output.")
                intermediate_dim = activation.size(-1)

            # Importance for each neuron.
            neuron_importance = activation.abs().mean(dim=(0, 1))

            # Decide the k of Top-K
            k = max(
                int(intermediate_dim * self.config.top_k_ratio),
                self.config.min_active_neurons
            )
            k = min(k, intermediate_dim)

            # Select
            _, top_indices = torch.topk(neuron_importance, k)
            active_neurons[path] = set(top_indices.cpu().numpy().tolist())

            if self.config.verbose:
                logger.info(f"Layer {path}: {k}/{intermediate_dim} neurons are selected. ({k / intermediate_dim:.1%})")

        return active_neurons

    def _update_ffn_layers(self, active_neurons: Dict[str, Set[int]]):
        """Update FFN"""
        for path, indices in active_neurons.items():
            if path in self.dynamic_ffn_modules:
                self.dynamic_ffn_modules[path].update_active_neurons(indices)

    def _use_original_ffn_layers(self):

        for module in self.dynamic_ffn_modules.values():
            module.use_original()

    def _update_token_thresholds(self):
        """
        Update surprisal and entropy thresholds for the token.
        Slightly different from the implementation in the paper.
        """
        window_size = min(len(self.surprisal_history), self.config.update_window)

        if window_size > 5:  # More prompts
            # Use only the most recent window of data.
            recent_surprisals = self.surprisal_history[-window_size:]
            recent_entropies = self.entropy_history[-window_size:]

            # Calculate threshold using percentile.
            self.surprisal_threshold_value = np.percentile(
                recent_surprisals,
                self.config.surprisal_threshold * 100
            )
            self.entropy_threshold_value = np.percentile(
                recent_entropies,
                self.config.entropy_threshold * 100
            )

            if self.config.verbose:
                logger.info(f"Update threshold - Surprisal: {self.surprisal_threshold_value:.4f}, "
                            f"Entropy: {self.entropy_threshold_value:.4f}")

    def _initialize_active_neurons(self, input_ids):
        """init methods"""
        # Get activations
        with torch.no_grad():
            _ = self.model(input_ids)

        if self.config.smart_neuron_init:
            initial_active_neurons = self._select_top_k_neurons(self.ffn_activations)
            self._update_ffn_layers(initial_active_neurons)
        else:
            self._use_original_ffn_layers()

    def _clear_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_with_conditional_computation(
            self,
            prompt: str,
            output_file_prefix: str,
            max_new_tokens: Optional[int] = None,
            **generation_kwargs
    ) -> Dict:
        """Generate.

        Args:
            output_file_prefix: Prefix for easier file saving

        """
        logger.info(f"Start generating, prompt length: {len(prompt.split())}")

        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens

        # Update generation params
        gen_kwargs = {
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p
        }
        gen_kwargs.update(generation_kwargs)


        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_length = input_ids.shape[1]

        logger.info(f"The prompt contains {input_length} tokens.")

        generated_ids = input_ids.clone()
        token_metrics = []

        self.surprisal_history = []
        self.entropy_history = []

        # Create metric file
        timestamp = int(time.time())
        metrics_file = f"{self.config.output_dir}/{output_file_prefix}_metrics_{timestamp}.h5"

        if self.config.save_metrics:
            with h5py.File(metrics_file, 'w') as f:
                f.create_dataset('prompt', data=prompt)
                f.create_dataset('input_ids', data=input_ids.cpu().numpy())
                token_metrics_group = f.create_group('token_metrics')
                f.attrs['conditional_computation'] = not self.config.baseline_run

        progress_bar = tqdm(total=max_new_tokens, desc="生成Token")

        if not self.config.baseline_run:
            self._initialize_active_neurons(input_ids)

        # Trace the change of activation!
        activation_changes = 0

        start_time = time.time()

        # Begin autoregressive generation.
        current_length = input_length
        for i in range(max_new_tokens):
            # Current input
            current_input = generated_ids[:, -min(1024, current_length):]

            # Get next token
            with torch.no_grad():
                outputs = self.model(current_input)

            # Get logits of next token
            next_token_logits = outputs.logits[:, -1, :]

            if gen_kwargs.get("do_sample", False):
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Add new token
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            current_length += 1

            # Metrics of next tokens
            token_id = next_token.item()
            metrics = self._compute_token_metrics(next_token_logits, token_id)
            token_metrics.append(metrics)

            # Update history metrics
            self.surprisal_history.append(metrics["normalized_surprisal"])
            self.entropy_history.append(metrics["normalized_entropy"])

            # Update threshold every update_window token
            if (i % self.config.update_window == 0 and
                    i > 0 and
                    self.config.adaptive_thresholds):
                self._update_token_thresholds()

            # Save
            if self.config.save_metrics:
                with h5py.File(metrics_file, 'a') as f:
                    token_group = f['token_metrics'].create_group(f'token_{i}')
                    for k, v in metrics.items():
                        if not isinstance(v, (str, bytes)):
                            token_group.create_dataset(k, data=v)
                        else:
                            token_group.create_dataset(
                                k,
                                data=np.array([v.encode('utf-8')], dtype='S100')
                            )

            # Check whether to trigger the FFN update condition.
            if (not self.config.baseline_run and
                    len(self.surprisal_history) > self.config.update_window and
                    metrics["normalized_surprisal"] > self.surprisal_threshold_value and
                    metrics["normalized_entropy"] > self.entropy_threshold_value):

                # Obtain the latest FFN activations and update active neurons.
                updated_active_neurons = self._select_top_k_neurons(self.ffn_activations)
                self._update_ffn_layers(updated_active_neurons)
                activation_changes += 1

                if (self.config.verbose and
                        i % self.config.log_frequency == 0):
                    logger.info(f"\nToken {i}: 检测到高不确定性，更新FFN层")
                    logger.info(
                        f"  Surprisal: {metrics['normalized_surprisal']:.4f} (阈值: {self.surprisal_threshold_value:.4f})")
                    logger.info(
                        f"  Entropy: {metrics['normalized_entropy']:.4f} (阈值: {self.entropy_threshold_value:.4f})")

            if i % 10 == 0:
                progress_bar.set_postfix({
                    "surprisal": f"{metrics['normalized_surprisal']:.3f}",
                    "entropy": f"{metrics['normalized_entropy']:.3f}",
                    "changes": activation_changes
                })
            progress_bar.update(1)

            if i % 50 == 0:
                self._clear_memory()

        progress_bar.close()

        # Retain original FFN
        self._use_original_ffn_layers()

        generation_time = time.time() - start_time
        tokens_per_second = max_new_tokens / generation_time

        # Decode
        generated_text = self.tokenizer.decode(generated_ids[0, input_length:], skip_special_tokens=True)

        # Whole metrics
        avg_surprisal = np.mean([m["normalized_surprisal"] for m in token_metrics])
        avg_entropy = np.mean([m["normalized_entropy"] for m in token_metrics])

        # Statistics
        if self.config.save_metrics:
            self._save_generation_stats(
                metrics_file,
                token_metrics,
                activation_changes,
                generation_time,
                tokens_per_second
            )

        logger.info(f"Generation complete, produced {len(token_metrics)} tokens.")
        logger.info(f"Average surprisal: {avg_surprisal:.4f}, average entropy: {avg_entropy:.4f}.")
        logger.info(f"Activation scheme changed: {activation_changes} times.")
        logger.info(f"Generation speed: {tokens_per_second:.2f} tokens per second.")

        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "token_metrics": token_metrics,
            "avg_surprisal": avg_surprisal,
            "avg_entropy": avg_entropy,
            "activation_changes": activation_changes,
            "metrics_file": metrics_file if self.config.save_metrics else None,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second
        }

    def _save_generation_stats(
            self,
            metrics_file: str,
            token_metrics: List[Dict],
            activation_changes: int,
            generation_time: float,
            tokens_per_second: float
    ):
        """Save generated statistics and visualizations."""
        # Get metrics
        surprisals = [m["normalized_surprisal"] for m in token_metrics]
        entropies = [m["normalized_entropy"] for m in token_metrics]

        # Save metrics to HDF5.
        with h5py.File(metrics_file, 'a') as f:
            stats = f.create_group('statistics')
            stats.create_dataset('surprisals', data=surprisals)
            stats.create_dataset('entropies', data=entropies)
            stats.create_dataset('activation_changes', data=activation_changes)
            stats.create_dataset('avg_surprisal', data=np.mean(surprisals))
            stats.create_dataset('avg_entropy', data=np.mean(entropies))
            stats.create_dataset('generation_time', data=generation_time)
            stats.create_dataset('tokens_per_second', data=tokens_per_second)

        # Visualization
        plt.figure(figsize=(12, 8))

        # Surprisal changes with time
        plt.subplot(2, 1, 1)
        plt.plot(surprisals, label='Normalized Surprisal')
        if self.surprisal_threshold_value > 0:
            plt.axhline(y=self.surprisal_threshold_value, color='r', linestyle='--', label='Thres')
        plt.xlabel('Token Idx')
        plt.ylabel('Normalized Surprisal')
        plt.title('Token Surprisal Durning Generation')
        plt.legend()

        # Entropy changes with time
        plt.subplot(2, 1, 2)
        plt.plot(entropies, label='Normalized Entropy')
        if self.entropy_threshold_value > 0:
            plt.axhline(y=self.entropy_threshold_value, color='r', linestyle='--', label='阈值')
        plt.xlabel('Token Idx')
        plt.ylabel('Normalized Entropy')
        plt.title('Token Entropy Dyring Generation')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{metrics_file.replace('.h5', '_plot.png')}")
        plt.close()

    def run_experiment(self):
        """run as a whole"""

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        experiment_dir = f"{self.config.output_dir}/experiment_{timestamp}"
        os.makedirs(experiment_dir, exist_ok=True)

        # Save configurations
        config_file = f"{experiment_dir}/config.json"
        with open(config_file, 'w') as f:
            # Change dataclass to dict
            config_dict = {k: v for k, v in self.config.__dict__.items()}
            json.dump(config_dict, f, indent=2)

        logger.info(f"Experimental configuration saved to:{config_file}")

        # Save all results
        all_results = []

        # Get prompts
        prompts = list(self.data_loader.get_prompts())
        logger.info(f"There are {len(prompts)} prompts to process.")

        # Handle each prompt
        for i, prompt in enumerate(prompts):
            logger.info(f"\nHandle Prompts {i + 1}/{len(prompts)}")

            # Prefix for output files.
            prefix = f"prompt_{i + 1}"

            # Run the baseline version first if a comparison is needed.
            if self.config.compare_with_baseline:
                logger.info("Run the baseline version (without conditional computation)...")

                # Set baseline_run=True here.
                original_baseline = self.config.baseline_run
                self.config.baseline_run = True

                baseline_result = self.generate_with_conditional_computation(
                    prompt=prompt,
                    output_file_prefix=f"{prefix}_baseline"
                )

                # Go back to original config
                self.config.baseline_run = original_baseline

                baseline_result["experiment_type"] = "baseline"
                all_results.append(baseline_result)

                self._clear_memory()

            # Run the conditional version
            if not self.config.baseline_run:
                logger.info("Run the conditional computation version...")

                conditional_result = self.generate_with_conditional_computation(
                    prompt=prompt,
                    output_file_prefix=f"{prefix}_conditional"
                )

                conditional_result["experiment_type"] = "conditional"
                all_results.append(conditional_result)
            else:
                # If only baseline is needed
                logger.info("OK...good to know you only choose to run the baseline version...")

                baseline_result = self.generate_with_conditional_computation(
                    prompt=prompt,
                    output_file_prefix=f"{prefix}_baseline"
                )

                baseline_result["experiment_type"] = "baseline"
                all_results.append(baseline_result)

            self._clear_memory()

            # Deliberately print a separator line after each prompt.
            logger.info("-" * 60)

        # Sav the summary
        summary_file = f"{experiment_dir}/summary.json"

        # Change to json
        serializable_results = []
        for result in all_results:
            serializable_result = {
                "prompt": result["prompt"][:100] + "..." if len(result["prompt"]) > 100 else result["prompt"],
                "generated_text_length": len(result["generated_text"]),
                "avg_surprisal": result["avg_surprisal"],
                "avg_entropy": result["avg_entropy"],
                "activation_changes": result.get("activation_changes", 0),
                "experiment_type": result["experiment_type"],
                "generation_time": result.get("generation_time", 0),
                "tokens_per_second": result.get("tokens_per_second", 0)
            }
            serializable_results.append(serializable_result)

        with open(summary_file, 'w') as f:
            json.dump({
                "experiment_config": config_dict,
                "results": serializable_results,
                "timestamp": timestamp
            }, f, indent=2)

        logger.info(f"Experiments save to: {summary_file}")

        # If a baseline comparison was conducted, print the comparison results.
        if self.config.compare_with_baseline:
            conditional_results = [r for r in all_results if r["experiment_type"] == "conditional"]
            baseline_results = [r for r in all_results if r["experiment_type"] == "baseline"]

            if conditional_results and baseline_results:
                logger.info("\nBaseline vs. CLADA: ")

                # Meow
                avg_conditional_surprisal = np.mean([r["avg_surprisal"] for r in conditional_results])
                avg_baseline_surprisal = np.mean([r["avg_surprisal"] for r in baseline_results])

                avg_conditional_entropy = np.mean([r["avg_entropy"] for r in conditional_results])
                avg_baseline_entropy = np.mean([r["avg_entropy"] for r in baseline_results])

                avg_conditional_speed = np.mean([r.get("tokens_per_second", 0) for r in conditional_results])
                avg_baseline_speed = np.mean([r.get("tokens_per_second", 0) for r in baseline_results])

                logger.info(
                    f"Average surprisal: Baseline={avg_baseline_surprisal:.4f}, CLADA={avg_conditional_surprisal:.4f}")
                logger.info(f"Average entropy: Baseline={avg_baseline_entropy:.4f}, CLADA={avg_conditional_entropy:.4f}")
                logger.info(
                    f"Generation latency(tokens/秒): Baseline={avg_baseline_speed:.2f}, CLADA={avg_conditional_speed:.2f}")

                # Speedup
                if avg_baseline_speed > 0:
                    speedup = avg_conditional_speed / avg_baseline_speed
                    logger.info(f"Speekup: {speedup:.2f}x")

        return all_results
