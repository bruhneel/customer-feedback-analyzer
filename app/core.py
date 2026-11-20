"""Synthetic retail dataset and category-token classification logic."""
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

CATEGORIES = {
    "product_issue": "A product is defective, damaged, incomplete, or performs poorly.",
    "staff_praise": "Positive feedback about an employee or service interaction.",
    "pricing": "Prices, discounts, fees, or billing discrepancies.",
    "delivery_or_stock": "Shipping, collection, availability, or stock problems.",
    "store_experience": "The in-store environment, layout, access, or wait.",
}


@dataclass(frozen=True)
class Feedback:
    comment: str
    label: str
    group: str


def category_token(label: str) -> str:
    """Return the reserved token for a known category."""
    if label not in CATEGORIES:
        raise ValueError(f"Unknown category: {label}")
    return f"<|{label}|>"


def format_prompt(comment: str) -> str:
    """Keep inputs on one line and prevent literal special-token injection."""
    cleaned = " ".join(comment.split()).translate(str.maketrans("<>", "‹›"))
    if not cleaned:
        raise ValueError("Enter a nonempty feedback comment.")
    return f"Feedback: {cleaned}\nCategory:"


def format_record(row: Feedback, end_token: str = "<|endoftext|>") -> str:
    """Build the full causal target sequence for a labeled comment."""
    return f"{format_prompt(row.comment)}{category_token(row.label)}{end_token}"


def split_feedback(rows: Sequence[Feedback], seed: int = 42) -> dict[str, list[Feedback]]:
    """Stratify by label while keeping all phrasings of each scenario together."""
    groups: dict[str, dict[str, list[Feedback]]] = defaultdict(lambda: defaultdict(list))
    seen = set()
    group_labels: dict[str, str] = {}
    for row in rows:
        prompt = format_prompt(row.comment).casefold()
        category_token(row.label)
        if not row.group or prompt in seen:
            raise ValueError("Each comment must be unique and have a group.")
        seen.add(prompt)
        if group_labels.setdefault(row.group, row.label) != row.label:
            raise ValueError(f"Scenario group {row.group} has inconsistent labels.")
        groups[row.label][row.group].append(row)
    rng = random.Random(seed)
    splits: dict[str, list[Feedback]] = {"train": [], "validation": [], "test": []}
    for label in CATEGORIES:
        scenarios = sorted(groups[label])
        if len(scenarios) < 3:
            raise ValueError(f"At least three groups are required for {label}.")
        rng.shuffle(scenarios)
        holdout = max(1, len(scenarios) // 6)
        for name, selected in (
            ("validation", scenarios[:holdout]),
            ("test", scenarios[holdout:2 * holdout]),
            ("train", scenarios[2 * holdout:]),
        ):
            for group in selected:
                splits[name].extend(groups[label][group])
    for records in splits.values():
        rng.shuffle(records)
    return splits


def category_token_ids(tokenizer: "PreTrainedTokenizerBase") -> dict[str, int]:
    """Resolve each category to a distinct, single special-token ID."""
    result = {}
    for label in CATEGORIES:
        token = category_token(label)
        ids = tokenizer.encode(token, add_special_tokens=False)
        if token not in tokenizer.all_special_tokens or len(ids) != 1:
            raise ValueError(f"Tokenizer is missing the special category token {token}.")
        result[label] = ids[0]
    if len(set(result.values())) != len(result):
        raise ValueError("Category tokens must have distinct IDs.")
    return result


def extend_categories(model: "PreTrainedModel", tokenizer: "PreTrainedTokenizerBase") -> dict[str, int]:
    """Register category tokens and initialize new tied embeddings at the old mean."""
    import torch

    initial_size = model.get_input_embeddings().num_embeddings
    if initial_size != len(tokenizer):
        raise ValueError("The base model and tokenizer vocabularies must match.")
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [category_token(label) for label in CATEGORIES]},
        replace_additional_special_tokens=False,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("A GPT-2 tokenizer must define an end token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if len(tokenizer) != initial_size:
        with torch.no_grad():
            initial_mean = model.get_input_embeddings().weight.mean(dim=0)
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            model.get_input_embeddings().weight[initial_size:].copy_(initial_mean)
            model.tie_weights()
    model.config.pad_token_id = tokenizer.pad_token_id
    return category_token_ids(tokenizer)


def prompt_ids(tokenizer: "PreTrainedTokenizerBase", comment: str, max_length: int) -> list[int]:
    """Truncate the comment while preserving the category cue and two target slots."""
    prompt = format_prompt(comment)
    prefix = tokenizer.encode("Feedback:", add_special_tokens=False)
    suffix = tokenizer.encode("\nCategory:", add_special_tokens=False)
    budget = max_length - len(prefix) - len(suffix) - 2
    if budget < 1:
        raise ValueError("Context limit is too short for the feedback template.")
    text = prompt[len("Feedback:"):-len("\nCategory:")]
    if len(text) > 12000:
        text = text[:9000] + " " + text[-3000:]
    body = tokenizer.encode(text, add_special_tokens=False)
    if len(body) > budget:
        tail = budget // 4
        body = body[:budget - tail] + (body[-tail:] if tail else [])
    return [*prefix, *body, *suffix]


def training_example(
    tokenizer: "PreTrainedTokenizerBase", row: Feedback, ids: dict[str, int], max_length: int
) -> dict[str, list[int]]:
    """Apply cross-entropy only to the category and EOS, including on padded batches."""
    prefix = prompt_ids(tokenizer, row.comment, max_length)
    target = [ids[row.label], tokenizer.eos_token_id]
    return {"input_ids": prefix + target, "labels": [-100] * len(prefix) + target}


@dataclass(frozen=True)
class FeedbackCollator:
    """Right-pad a batch without supervising pad positions."""

    pad_id: int

    def __call__(self, rows: Sequence[dict[str, list[int]]]) -> dict:
        import torch

        length = max(len(row["input_ids"]) for row in rows)
        ids, labels, masks = [], [], []
        for row in rows:
            count = len(row["input_ids"])
            padding = length - count
            ids.append(row["input_ids"] + [self.pad_id] * padding)
            labels.append(row["labels"] + [-100] * padding)
            masks.append([1] * count + [0] * padding)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(masks)}


# --- Milestone 4: Implement an inference function to classify new feedback.
# Hint: Adapt the inference pipeline from the drill to generate category tokens.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Build a simple Streamlit app to input comments and show classifications.
# Hint: Integrate your inference function into a basic Streamlit UI.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
