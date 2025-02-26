import logging
import argparse
from .config import ConditionalComputationConfig
from .clada import ConditionalComputationModel



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)



def parse_args():
    parser = argparse.ArgumentParser(description="CLADA")

    # Model and data
    parser.add_argument("--model_path", type=str, default="./pretrained_model",
                        help="Path to pretrained model weights")
    parser.add_argument("--data_path", type=str, default="./data/xsum/train.jsonl",
                        help="Path to prompt file")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--prompt_field", type=str, default="Article",
                        help="Field name for the prompt in the jsonl file")

    # FFN
    parser.add_argument("--ffn_reduction_ratio", type=float, default=0.3)
    parser.add_argument("--top_k_ratio", type=float, default=0.3)
    parser.add_argument("--min_active_neurons", type=int, default=100)

    # DynamicFFN
    parser.add_argument("--surprisal_threshold", type=float, default=0.7,
                        help="High surprisal threshold (percentile).")
    parser.add_argument("--entropy_threshold", type=float, default=0.7,
                        help="High entropy threshold (percentile).")
    parser.add_argument("--update_window", type=int, default=10,
                        help="Update statistics after generating every update_window tokens.")

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Max generation tokens")
    parser.add_argument("--num_prompts", type=int, default=5,
                        help="Number of prompts to process; -1 indicates all.")

    # Sample
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    # Memory
    parser.add_argument("--low_memory_mode", action="store_true")
    parser.add_argument("--precision", type=str, default="float16",
                        choices=["float16", "float32", "float64"])

    # Experiments
    parser.add_argument("--baseline_run", action="store_true",
                        help="Whether run baseline(no CLADA)")
    parser.add_argument("--compare_with_baseline", action="store_true",
                        help="Whether compare with baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main():

    args = parse_args()

    config = ConditionalComputationConfig(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        prompt_field=args.prompt_field,
        ffn_reduction_ratio=args.ffn_reduction_ratio,
        top_k_ratio=args.top_k_ratio,
        min_active_neurons=args.min_active_neurons,
        surprisal_threshold=args.surprisal_threshold,
        entropy_threshold=args.entropy_threshold,
        update_window=args.update_window,
        max_new_tokens=args.max_new_tokens,
        num_prompts=args.num_prompts,
        temperature=args.temperature,
        top_p=args.top_p,
        low_memory_mode=args.low_memory_mode,
        precision=args.precision,
        baseline_run=args.baseline_run,
        compare_with_baseline=args.compare_with_baseline,
        seed=args.seed,
        verbose=args.verbose
    )

    model = ConditionalComputationModel(config=config)

    results = model.run_experiment()

    logger.info("实验完成")


if __name__ == "__main__":
    main()