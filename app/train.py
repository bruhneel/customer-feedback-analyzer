"""Fine-tune the pretrained GPT-2 category-token tagger on a CPU."""
import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import Trainer, TrainingArguments, set_seed

from app.core import FeedbackCollator, extend_categories, split_feedback, training_example
from app.llm import DEFAULT_BASE, load_base
from app.storage import load_feedback, save_report

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def make_trainer(
    model: "PreTrainedModel", tokenizer: "PreTrainedTokenizerBase",
    train: list[dict], validation: list[dict], output: Path, *,
    epochs: float = 4, batch_size: int = 8, learning_rate: float = 0.0001,
    seed: int = 42, max_steps: int = -1, quiet: bool = False,
) -> Trainer:
    """Configure Trainer without periodic checkpoints, remote logging, or accelerators."""
    args = TrainingArguments(
        output_dir=str(output), use_cpu=True, num_train_epochs=epochs, max_steps=max_steps,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate, lr_scheduler_type="linear", warmup_ratio=0.05,
        weight_decay=0.01, max_grad_norm=1.0, optim="adamw_torch",
        eval_strategy="epoch", logging_strategy="steps", logging_steps=10,
        logging_first_step=True, save_strategy="no", report_to="none",
        prediction_loss_only=True, disable_tqdm=quiet, seed=seed, data_seed=seed,
    )
    return Trainer(
        model=model, args=args, processing_class=tokenizer,
        train_dataset=train, eval_dataset=validation,
        data_collator=FeedbackCollator(tokenizer.pad_token_id),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--dataset", type=Path, default=Path("data/feedback.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--log", type=Path, default=Path("results/training.json"))
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.learning_rate, args.threads, args.max_length) <= 0:
        parser.error("Epochs, batch size, learning rate, threads and context must be positive.")
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    model, tokenizer = load_base(args.base_model)
    if args.max_length > model.config.max_position_embeddings:
        parser.error("Maximum length exceeds the model context limit.")
    ids = extend_categories(model, tokenizer)
    model.config.feedback_max_length = args.max_length
    model.config.feedback_split_seed = args.seed
    splits = split_feedback(load_feedback(args.dataset), args.seed)
    train, val = ([training_example(tokenizer, row, ids, args.max_length)
                   for row in splits[name]] for name in ("train", "validation"))
    trainer = make_trainer(
        model, tokenizer, train, val, args.output, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed,
    )
    initial = trainer.evaluate()["eval_loss"]
    trainer.train()
    final = trainer.evaluate()["eval_loss"]
    trainer.save_model()
    tokenizer.save_pretrained(args.output)
    save_report(args.log, {
        "base_model": args.base_model, "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "max_length": args.max_length, "category_tokens": ids,
        "split_sizes": {key: len(part) for key, part in splits.items()},
        "initial_validation_loss": initial, "final_validation_loss": final,
        "history": trainer.state.log_history,
    })


if __name__ == "__main__":
    main()
