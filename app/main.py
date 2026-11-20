"""Streamlit interface for local GPT-2 feedback tagging."""
from collections import Counter
from datetime import date
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core import CATEGORIES, FeedbackClassifier
from app.llm import load_classifier
from app.ui_data import classify_comments, parse_comments, parse_upload, results_csv, weekly_issues

MODEL = Path("artifacts/model")
EXAMPLES = {
    "Product issue": "The strap split the first time I put anything in the bag.",
    "Staff praise": "The cashier was brilliant and went back to find the right size for me.",
    "Delivery or stock": "The tracking says delivered and there is no box anywhere.",
}


@st.cache_resource
def get_classifier(directory: str, revision: int) -> FeedbackClassifier:
    """Reuse the model until the trained artifact changes."""
    import torch

    torch.set_num_threads(4)
    return load_classifier(Path(directory))


def add_example(comment: str) -> None:
    """Append a sample comment to the text box."""
    existing = st.session_state.get("comments", "").rstrip()
    st.session_state["comments"] = f"{existing}\n{comment}" if existing else comment


def render() -> None:
    """Render the classifier, category counts, and CSV export."""
    st.set_page_config(page_title="Customer Feedback Analyzer", layout="wide")
    st.title("Customer Feedback Analyzer")
    st.caption("Classify retail feedback with a locally fine-tuned GPT-2 model.")
    threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.60, 0.05)
    st.sidebar.caption("Scores are relative to the five categories. Low scores are marked uncertain.")
    config = MODEL / "config.json"
    weights = MODEL / "model.safetensors"
    revision = weights.stat().st_mtime_ns if config.is_file() and weights.is_file() else None
    st.sidebar.write("Model: DistilGPT2 checkpoint available" if revision else "Model: not trained")
    if revision is None:
        st.info("No trained model found. Run `python -m app.train` from the repository root.")
    with st.expander("Category definitions"):
        for label, description in CATEGORIES.items():
            st.write(f"**{label}** — {description}")
    for column, (title, example) in zip(st.columns(3), EXAMPLES.items(), strict=True):
        column.button(f"Try {title.lower()}", on_click=add_example, args=(example,))
    text = st.text_area("Feedback comments, one per line", key="comments", height=150)
    upload = st.file_uploader("Upload feedback CSV", type=["csv"],
                              help="Required column: comment. Optional date column: YYYY-MM-DD.")
    if st.button("Classify", type="primary", disabled=revision is None):
        try:
            today = date.today()
            records = [(comment, today) for comment in parse_comments(text)]
            if upload:
                records.extend(parse_upload(upload.getvalue(), today))
            if not records:
                raise ValueError("Enter or upload at least one feedback comment.")
            with st.spinner("Classifying locally…"):
                classifier = get_classifier(str(MODEL), revision)
                rows = classify_comments(classifier, [comment for comment, _ in records], threshold)
                for row, (_, received) in zip(rows, records, strict=True):
                    row["Date"] = received.isoformat()
                st.session_state["result"] = (rows, threshold)
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f"Classification failed: {exc}")
    if "result" not in st.session_state:
        return
    rows, applied_threshold = st.session_state["result"]
    st.subheader("Classification results")
    st.caption(f"{len(rows)} comments · threshold {applied_threshold:.0%}")
    st.dataframe(rows, hide_index=True,
                 column_config={"Confidence": st.column_config.NumberColumn(format="percent")})
    counts = Counter(row["Category"] for row in rows)
    st.bar_chart({"Category": list(counts), "Count": list(counts.values())},
                 x="Category", y="Count", horizontal=True)
    st.download_button("Download results CSV", results_csv(rows), "feedback-results.csv", "text/csv")
    with st.expander("Top issues this week", expanded=True):
        st.caption("Current batch only, Monday through today. Excludes staff praise and uncertain rows.")
        counts = weekly_issues(rows, date.today())
        if counts:
            st.dataframe([{"Category": label, "Count": count} for label, count in counts],
                         hide_index=True)
        else:
            st.info("No classified issue categories for this week in the current batch.")


if __name__ == "__main__":
    render()
