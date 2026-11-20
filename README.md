# Customer Feedback Analyzer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/4r7npt8fa6/badge.svg)](https://ask.thili.ai/b/4r7npt8fa6)

Retail managers need to quickly understand customer feedback. This project builds a Customer Feedback Analyzer using a fine-tuned GPT-2 model. It automatically categorizes comments into relevant tags like `<product_issue>` or `<staff_praise>`, helping managers identify trends and prioritize improvements efficiently.

**SAMPLE DATA:** The corpus is synthetic and contains no real customer records.

## What you are building

**Goal.** You will have built a tool that automatically categorizes customer feedback comments using a fine-tuned GPT-2 model, enabling retail managers to quickly identify trends and prioritize improvements.

**Done when.** You will have a Streamlit application where a user can input customer feedback comments and see them classified into predefined categories by your fine-tuned GPT-2 model. The classification logic and inference pipeline will be proven to work.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Define 3-5 feedback categories and create a small dataset of example comments with labels.**  
      Hint: Use techniques for dataset creation and labeling from the drill.
- [ ] **2. Extend GPT-2 tokenizer and model with new category tokens.**  
      Hint: Apply the tokenizer extension methods shown in the drill.
- [ ] **3. Fine-tune the model using the Hugging Face Trainer.**  
      Hint: Utilize the Hugging Face Trainer class as demonstrated in the drill.
- [ ] **4. Implement an inference function to classify new feedback.**  
      Hint: Adapt the inference pipeline from the drill to generate category tokens.
- [ ] **5. Build a simple Streamlit app to input comments and show classifications.**  
      Hint: Integrate your inference function into a basic Streamlit UI.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Display simple statistics (e.g., counts per category) of the classified comments within the Streamlit app.
- Add a feature to allow users to provide feedback on classification accuracy for individual comments.
- Enhance the Streamlit UI with more interactive elements or a better layout for improved user experience.
