"""Evaluate the local feedback tagger against a training-majority baseline."""
import argparse
from collections import Counter
from pathlib import Path
from typing import Sequence

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from app.core import CATEGORIES, split_feedback
from app.llm import load_classifier
from app.storage import load_feedback, save_report

EXAMPLES = [
    "The strap split the first time I put anything in the bag.",
    "The cashier was brilliant and went back to find the right size for me.",
    "I was charged twice even though I only bought one bowl.",
    "The tracking says delivered and there is no box anywhere.",
    "The shop was so crowded I couldn't get to the tills.",
]


def summarize(actual: Sequence[str], predicted: Sequence[str]) -> dict:
    """Return accuracy, stable per-class metrics, and true-by-predicted counts."""
    labels = list(CATEGORIES)
    report = classification_report(
        actual, predicted, labels=labels, output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(actual, predicted, labels=labels)
    return {"accuracy": float(accuracy_score(actual, predicted)),
            "per_class": {label: report[label] for label in labels},
            "confusion_matrix": matrix.tolist()}


def _table(report: dict) -> str:
    labels = report["label_order"]
    models = report["models"]
    lines = [
        "# Held-out feedback evaluation", "",
        f'{report["test_count"]} synthetic comments from {report["test_group_count"]} entirely held-out scenarios. '
        "Test metrics use a forced category to compare all models on the same rows.", "",
        "| Model | Accuracy |", "| --- | ---: |",
    ]
    lines += [f'| {name} | {metrics["accuracy"]:.1%} |' for name, metrics in models.items()]
    for name, metrics in models.items():
        lines += ["", f"## {name}", "", "| Category | Precision | Recall | F1 | Support |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for label in labels:
            row = metrics["per_class"][label]
            lines.append(
                f'| {label} | {row["precision"]:.3f} | {row["recall"]:.3f} | '
                f'{row["f1-score"]:.3f} | {int(row["support"])} |'
            )
        lines += ["", "Confusion matrix, rows are actual and columns are predicted.", "",
                  "| Actual / predicted | " + " | ".join(labels) + " |",
                  "| --- |" + " ---: |" * len(labels)]
        for label, counts in zip(labels, metrics["confusion_matrix"], strict=True):
            lines.append("| " + " | ".join([label, *map(str, counts)]) + " |")
    lines += ["", "## Novel demonstration inputs", "", "| Comment | Category | Score |",
              "| --- | --- | ---: |"]
    for row in report["examples"]:
        lines.append(f'| {row["comment"]} | {row["category"]} | {row["confidence"]:.1%} |')
    return "\n".join(lines) + "\n"


def evaluate(model_path: Path, dataset: Path) -> dict:
    """Use the checkpoint's split seed and record both metrics and individual outputs."""
    torch.set_num_threads(4)
    classifier = load_classifier(model_path)
    seed = getattr(classifier.model.config, "feedback_split_seed", 42)
    splits = split_feedback(load_feedback(dataset), seed)
    train, test = splits["train"], splits["test"]
    counts = Counter(row.label for row in train)
    majority = max(CATEGORIES, key=counts.__getitem__)
    outputs = classifier.classify_batch([row.comment for row in test])
    actual = [row.label for row in test]
    predicted = [row["category"] for row in outputs]
    examples = [dict(comment=comment, **prediction)
                for comment, prediction in zip(EXAMPLES, classifier.classify_batch(EXAMPLES), strict=True)]
    return {
        "seed": seed, "label_order": list(CATEGORIES), "test_count": len(test),
        "test_group_count": len({row.group for row in test}), "training_majority": majority,
        "models": {"DistilGPT2": summarize(actual, predicted),
                   "Training majority": summarize(actual, [majority] * len(test))},
        "examples": examples,
        "test_predictions": [
            {"comment": row.comment, "group": row.group, "actual": row.label, **prediction}
            for row, prediction in zip(test, outputs, strict=True)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--dataset", type=Path, default=Path("data/feedback.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = evaluate(args.model, args.dataset)
    save_report(args.output / "evaluation.json", report)
    (args.output / "evaluation.md").write_text(_table(report), encoding="utf-8")
    for model, results in report["models"].items():
        print(f'{model}: {results["accuracy"]:.1%}')


if __name__ == "__main__":
    main()
