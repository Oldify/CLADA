from dataclasses import dataclass


@dataclass
class ConditionalComputationConfig:
    """Conditional computation configuration parameters."""
    # Model and data configuration.
    model_path: str = "./pretrained_model"
    data_path: str = "./data/xsum/train.jsonl"
    output_dir: str = "./outputs"
    prompt_field: str = "article"

    # FFN configuration
    ffn_reduction_ratio: float = 0.3  # Reduce the FFN layer to specific proportion.
    top_k_ratio: float = 0.3  # Select the top k% of neurons.
    min_active_neurons: int = 100  # Minimum number of activated neurons.
    smart_neuron_init: bool = True  # Whether to use special methods for neuron initialization.

    # Dynamic conditional configuration.
    surprisal_threshold: float = 0.7  # High surprisal threshold (percentile).
    entropy_threshold: float = 0.7  # High entropy threshold (percentile).
    update_window: int = 10  # Update statistics after generating specific tokens.
    adaptive_thresholds: bool = True  # Whether to use adaptive threshold.

    # Generation configuration
    max_new_tokens: int = 1024
    max_prompt_length: int = 512
    batch_size: int = 1
    num_prompts: int = 5  # Number of prompts to process; -1 indicates all.

    # Sampling configuration
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9  # Cumulative probability threshold.

    # Memory management
    low_memory_mode: bool = True
    precision: str = "float16"
    # gpu_batch_size: int = 1
    # max_memory_usage: float = 0.9

    # Experimental control
    baseline_run: bool = False
    compare_with_baseline: bool = False
    seed: int = 42
    verbose: bool = False