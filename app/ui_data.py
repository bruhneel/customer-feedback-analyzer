"""Format text submissions and export classification results."""
import csv
from io import StringIO
from typing import Sequence

from app.core import FeedbackClassifier


def parse_comments(text: str) -> list[str]:
    """Return nonempty pasted lines without changing their wording."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def classify_comments(
    classifier: FeedbackClassifier, comments: Sequence[str], threshold: float
) -> list[dict[str, str | float]]:
    """Pair the input rows with the model's labels and top scores."""
    predictions = classifier.classify_batch(comments, threshold=threshold)
    return [{"Comment": comment, "Category": result["category"], "Confidence": result["confidence"]}
            for comment, result in zip(comments, predictions, strict=True)]


def results_csv(rows: list[dict[str, str | float]]) -> str:
    """Export classifications and escape spreadsheet formula prefixes."""
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["Comment", "Category", "Confidence"])
    writer.writeheader()
    for row in rows:
        comment = str(row["Comment"])
        if comment.lstrip().startswith(("=", "+", "-", "@")):
            comment = "'" + comment
        writer.writerow({**row, "Comment": comment})
    return buffer.getvalue()
