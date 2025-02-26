import json
import logging
import random
from typing import Iterator

logger = logging.getLogger(__name__)

class PromptDataLoader:
    """Load and process prompt data."""
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.data_path = config.data_path
        self.prompt_field = config.prompt_field
        self._load_data()

    def _load_data(self):
        """Load jsonl"""
        self.data = []
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        if self.prompt_field in item:
                            self.data.append(item)
                        else:
                            logger.warning(f"Field '{self.prompt_field}' not found in data item.")
                    except json.JSONDecodeError:
                        logger.warning(f"Error parsing JSON line: {line[:50]}...")

            logger.info(f"Loaded {len(self.data)} data entries.")

            # Perform sampling if `config.num_prompts` is not -1.
            if self.config.num_prompts != -1 and self.config.num_prompts < len(self.data):
                random.seed(self.config.seed)
                self.data = random.sample(self.data, self.config.num_prompts)
                logger.info(f"Randomly sample {self.config.num_prompts} data entries.")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def get_prompts(self) -> Iterator[str]:
        """Return prompt iterator."""
        for item in self.data:
            prompt = item[self.prompt_field]

            # Ensure the prompt does not exceed the maximum length.
            if self.config.max_prompt_length > 0:
                prompt_tokens = self.tokenizer.encode(prompt)
                if len(prompt_tokens) > self.config.max_prompt_length:
                    # Preserve the initial content when truncating.
                    prompt_tokens = prompt_tokens[:self.config.max_prompt_length]
                    prompt = self.tokenizer.decode(prompt_tokens)
                    if self.config.verbose:
                        logger.info(f"Prompt truncated to {self.config.max_prompt_length} tokens.")

            yield prompt
