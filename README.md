# Customer Feedback Analyzer

[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/4r7npt8fa6/badge.svg)](https://msrit.ask.thili.ai/b/4r7npt8fa6)

A local Streamlit app that labels retail feedback with a fine-tuned GPT-2 model and summarizes category counts. **Sample data only.** The dataset contains 240 original synthetic comments with no real customer records.

## Install and run

From the repository root with Python 3.12 and Git:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# On Linux, install the CPU PyTorch wheel first to avoid CUDA packages:
# pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env
python -m app.train
python -m app.evaluate
streamlit run app/main.py
```

The first training run downloads the public [DistilGPT2 checkpoint](https://huggingface.co/distilbert/distilgpt2) and tokenizer into the Hugging Face cache. After that, `HF_HUB_OFFLINE=1` enables offline retraining. Inference reads `artifacts/model/` locally. The workflow needs no API key. The starter's separate Ollama, Groq, or Gemini helper is optional. Keep `.env` local.

Use `HF_HUB_OFFLINE=1 pytest -q` for download-free tests with a tiny random GPT-2 and a locally trained tokenizer. The Dockerfile serves Streamlit on port 8080 by default. Mount a trained `artifacts/` directory at `/srv/artifacts` to run inference in the container.

## Milestones

1. **Data.** The five [category definitions](app/core.py) cover product issues, staff praise, pricing, delivery or stock, and store experience. Regenerate the checked-in synthetic data with `python data/build_feedback.py`. Seed 42 assigns 160 training, 40 validation and 40 test comments. All phrasings of the same scenario stay in one split.
2. **Tokenizer.** Add one reserved category token per label. Resizing adds one embedding row per new token to the tied input/output weight matrix. Original token IDs and rows are preserved. The extension initializes new rows at the old mean, and the training command seeds them from existing words like “price” and “delivery” for a semantic starting point. GPT-2 uses EOS as its pad ID, with padding masked.
3. **Training.** Hugging Face Trainer runs on CPU with a fixed seed and saves a final local checkpoint. It masks the prompt and padding to `-100`. The objective combines category-only cross-entropy with a small full-vocabulary category/EOS term. Early runs with plain full-vocabulary loss collapsed to a single category, so subsequent development looked at grouped validation predictions. The final run used 10 epochs and reduced validation objective from 2.5038 to 0.4242. [Training log](results/training.json)
4. **Inference.** Build the same prompt, preserve the `Category:` suffix when truncating, and take next-token logits at the last real prompt token. Softmax is applied only over the five category IDs. The top score falls back to `uncertain` below the chosen threshold.
5. **Streamlit.** Paste one comment per line or click a sample. Classify to see a table, count chart and CSV download. The sidebar controls the confidence threshold and shows whether the checkpoint exists.

## CSV and weekly summary

Upload a UTF-8 CSV with a `comment` column and optionally a `date` column using `YYYY-MM-DD`. Pasted comments and undated CSV rows use today's local date. The weekly view counts the current batch from Monday through today, excluding `staff_praise` and `uncertain`. It counts categories that may contain issues and does not infer whether a comment is positive or negative. The result CSV includes the date used for each row.

## Held-out results

DistilGPT2 achieved **67.5%** accuracy against **20.0%** for the training-majority baseline. In a balanced training set, the tie goes to the first configured label, `product_issue`.

| Category | DistilGPT2 F1 | Majority F1 |
| --- | ---: | ---: |
| product_issue | 0.800 | 0.333 |
| staff_praise | 0.706 | 0.000 |
| pricing | 0.875 | 0.000 |
| delivery_or_stock | 0.667 | 0.000 |
| store_experience | 0.267 | 0.000 |

The most common test confusions were `store_experience` labeled `staff_praise` (3), `store_experience` labeled `product_issue` (3), `delivery_or_stock` labeled `store_experience` (3), `staff_praise` labeled `store_experience` (2). The [evaluation report](results/evaluation.md) includes precision, recall, confusion matrices and the individual predictions in [JSON](results/evaluation.json).

A word unigram/bigram TF-IDF model with logistic regression trained on exactly the same training split achieved **25.0%**. Its per-class metrics and predictions are in the same evaluation reports. This comparison is specific to the small grouped synthetic split.

### Novel example comments

| Comment | Predicted category | Score |
| --- | --- | ---: |
| The strap split the first time I put anything in the bag. | `product_issue` | 100.0% |
| The cashier was brilliant and went back to find the right size for me. | `staff_praise` | 100.0% |
| I was charged twice even though I only bought one bowl. | `pricing` | 90.7% |
| The tracking says delivered and there is no box anywhere. | `delivery_or_stock` | 100.0% |
| The shop was so crowded I couldn't get to the tills. | `store_experience` | 100.0% |

## Limitations

The test set contains 40 synthetic comments from 10 held-out scenarios. The results are a small pipeline check, and the split was also inspected during the initial failed run. Scores are relative across the allowed categories and have not been calibrated. The app assigns one label to mixed-topic text and can miss uncommon phrasing, class overlap, or issues outside the five categories.
