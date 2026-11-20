"""Format text submissions and export classification results."""
from collections import Counter
import csv
from datetime import date, timedelta
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
    fields = ["Comment", "Category", "Confidence"]
    if any("Date" in row for row in rows):
        fields.append("Date")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        comment = str(row["Comment"])
        if comment.lstrip().startswith(("=", "+", "-", "@")):
            comment = "'" + comment
        writer.writerow({**row, "Comment": comment})
    return buffer.getvalue()


def parse_upload(content: bytes, today: date) -> list[tuple[str, date]]:
    """Read UTF-8 CSV with a comment and an optional ISO calendar date."""
    try:
        reader = csv.DictReader(StringIO(content.decode("utf-8-sig")), strict=True)
        headers = reader.fieldnames or []
        normalized = {field.strip().casefold(): field for field in headers}
        if "comment" not in normalized or len(normalized) != len(headers):
            raise ValueError("CSV needs a single 'comment' column and optional 'date' column.")
        rows = []
        for number, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(f"CSV row {number} has more values than headers.")
            comment = (row.get(normalized["comment"]) or "").strip()
            if not comment:
                continue
            value = (row.get(normalized.get("date", "")) or "").strip()
            try:
                received = date.fromisoformat(value) if value else today
                if value and received.isoformat() != value:
                    raise ValueError("Expected YYYY-MM-DD.")
            except ValueError as exc:
                raise ValueError(f"CSV row {number} date must be YYYY-MM-DD.") from exc
            rows.append((comment, received))
        return rows
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError("Upload a valid UTF-8 comma-separated file.") from exc


def weekly_issues(rows: list[dict[str, str | float]], today: date) -> list[tuple[str, int]]:
    """Rank non-praise, non-uncertain categories from Monday through today."""
    monday = today - timedelta(days=today.weekday())
    counts = Counter(
        str(row["Category"]) for row in rows
        if row["Category"] not in {"staff_praise", "uncertain"}
        and monday <= date.fromisoformat(str(row["Date"])) <= today
    )
    return sorted(counts.items(), key=lambda entry: (-entry[1], entry[0]))
