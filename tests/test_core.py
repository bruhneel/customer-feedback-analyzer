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


def test_category_tokens_resize_and_preserve_inputs(tiny_pair):
    import torch

    from app.core import CATEGORIES, category_token, extend_categories

    model, tokenizer = tiny_pair
    old_vocab = tokenizer.get_vocab()
    old_weights = model.get_input_embeddings().weight.detach().clone()
    ids = extend_categories(model, tokenizer)
    assert list(ids) == list(CATEGORIES) and len(set(ids.values())) == len(CATEGORIES)
    assert all(tokenizer.encode(category_token(label), add_special_tokens=False) == [index]
               for label, index in ids.items())
    assert all(tokenizer.convert_tokens_to_ids(word) == index for word, index in old_vocab.items())
    weight = model.get_input_embeddings().weight
    assert weight.shape[0] == len(tokenizer) == len(old_vocab) + len(CATEGORIES)
    torch.testing.assert_close(weight[:len(old_vocab)], old_weights, rtol=0, atol=0)
    torch.testing.assert_close(weight[len(old_vocab):], old_weights.mean(0).expand(len(ids), -1))
    assert weight is model.get_output_embeddings().weight
    assert tokenizer.pad_token_id == model.config.pad_token_id == tokenizer.eos_token_id


def test_trainer_overfits_tiny_model(tiny_pair, tmp_path):
    import math

    from app.core import Feedback, FeedbackCollator, extend_categories, training_example
    from app.train import make_trainer

    model, tokenizer = tiny_pair
    ids = extend_categories(model, tokenizer)
    rows = [
        training_example(tokenizer, Feedback("Broken zipper", "product_issue", "a"), ids, 24),
        training_example(tokenizer, Feedback("Staff helped me", "staff_praise", "b"), ids, 24),
    ]
    for row in rows:
        assert row["labels"][:-2] == [-100] * (len(row["input_ids"]) - 2)
        assert row["labels"][-2:] == row["input_ids"][-2:]
    batch = FeedbackCollator(tokenizer.pad_token_id)(rows)
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    trainer = make_trainer(model, tokenizer, rows * 4, rows, tmp_path, max_steps=32,
                           batch_size=4, learning_rate=0.02, quiet=True)
    initial = trainer.evaluate()["eval_loss"]
    train_loss = trainer.train().training_loss
    final = trainer.evaluate()["eval_loss"]
    assert math.isfinite(train_loss) and math.isfinite(final) and final < initial * 0.6


# Milestone 4: Implement an inference function to classify new feedback.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Build a simple Streamlit app to input comments and show classifications.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

