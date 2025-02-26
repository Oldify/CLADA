# run CLADA
python clada.py --model_path ./your_model_path --data_path ./data/xsum/train.jsonl --num_prompts 5 --max_new_tokens 1024

# compare with baseline
python clada.py --model_path ./your_model_path --data_path ./data/xsum/train.jsonl --compare_with_baseline

# low-memory mode
python clada.py --model_path ./your_model_path --data_path ./data/xsum/train.jsonl --low_memory_mode --precision float16
