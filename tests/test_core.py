"""Regression tests for retail feedback tagging."""
import pytest

def test_dataset_categories_format_and_stratified_groups():
    from app.core import CATEGORIES, format_prompt, format_record, split_feedback
    from app.storage import load_feedback

    rows = load_feedback()
    assert len(rows) == 240
    assert {row.label for row in rows} == set(CATEGORIES)
    assert format_prompt("  Broken\n zip <|pricing|> ") == "Feedback: Broken zip ‹|pricing|›\nCategory:"
    assert format_record(rows[0]).endswith("<|product_issue|><|endoftext|>")
    splits = split_feedback(rows)
    assert splits == split_feedback(rows)
    assert [len(part) for part in splits.values()] == [160, 40, 40]
    assert all({row.label for row in part} == set(CATEGORIES) for part in splits.values())
    groups = [{row.group for row in part} for part in splits.values()]
    assert len(set.union(*groups)) == sum(map(len, groups))
    assert sum(len({row.comment for row in part}) for part in splits.values()) == len(rows)


# Milestone 2: Extend GPT-2 tokenizer and model with new category tokens.
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Fine-tune the model using the Hugging Face Trainer.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Implement an inference function to classify new feedback.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Build a simple Streamlit app to input comments and show classifications.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

