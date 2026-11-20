"""Regression tests for retail feedback tagging."""
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


def test_constrained_inference_batch_threshold_and_truncation(tiny_pair):
    import pytest

    from app.core import CATEGORIES, FeedbackClassifier, extend_categories, prompt_ids

    model, tokenizer = tiny_pair
    extend_categories(model, tokenizer)
    classifier = FeedbackClassifier(model, tokenizer)
    texts = ["Broken zipper", "Staff helped me " * 5000]
    outputs = classifier.classify_batch(texts)
    for text, result in zip(texts, outputs, strict=True):
        single = classifier.classify(text)
        assert result["category"] == single["category"] and result["category"] in CATEGORIES
        assert result["scores"] == pytest.approx(single["scores"], abs=1e-6)
        assert sum(result["scores"].values()) == pytest.approx(1.0)
        assert result["confidence"] == max(result["scores"].values())
    assert classifier.classify(texts[0], threshold=1)["category"] == "uncertain"
    ids = prompt_ids(tokenizer, texts[1], classifier.max_length)
    tail = tokenizer.encode("\nCategory:", add_special_tokens=False)
    assert len(ids) == classifier.max_length - 2 and ids[-len(tail):] == tail
    with pytest.raises(ValueError, match="nonempty"):
        classifier.classify("   ")
    assert classifier.classify_batch([]) == []


def test_evaluation_reports_zero_support_labels():
    from app.evaluate import summarize

    result = summarize(["product_issue", "staff_praise"], ["product_issue", "product_issue"])
    assert result["accuracy"] == 0.5
    assert result["per_class"]["product_issue"]["f1-score"] == 2 / 3
    assert result["per_class"]["pricing"]["support"] == 0
    assert result["confusion_matrix"][1][0] == 1


def test_streamlit_missing_model_and_clickable_examples(tmp_path, monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    entry = Path("app/main.py").resolve()
    monkeypatch.chdir(tmp_path)
    page = AppTest.from_file(str(entry)).run()
    assert not page.exception
    assert page.title[0].value == "Customer Feedback Analyzer"
    assert "python -m app.train" in page.info[0].value
    assert page.button[-1].disabled
    page.button[0].click().run()
    assert page.text_area[0].value


def test_ui_classification_and_csv(tiny_pair):
    from app.core import CATEGORIES, FeedbackClassifier, extend_categories
    from app.ui_data import classify_comments, parse_comments, results_csv

    model, tokenizer = tiny_pair
    extend_categories(model, tokenizer)
    inputs = parse_comments(" Broken zipper\n\n =1+2\n")
    assert inputs == ["Broken zipper", "=1+2"]
    rows = classify_comments(FeedbackClassifier(model, tokenizer), inputs, 0)
    assert len(rows) == 2 and all(row["Category"] in CATEGORIES for row in rows)
    exported = results_csv(rows)
    assert "Comment,Category,Confidence" in exported and "'=1+2" in exported
