"""Tiny deterministic model and tokenizer shared by download-free tests."""
import pytest


@pytest.fixture
def tiny_pair():
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    torch.set_num_threads(1)
    torch.manual_seed(42)
    backend = Tokenizer(WordLevel(unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.train_from_iterator(
        ["Feedback: Broken zipper\nCategory:", "Feedback: Staff helped me\nCategory:"],
        WordLevelTrainer(special_tokens=["[UNK]", "[EOS]"]),
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]")
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(tokenizer), n_positions=48, n_embd=24, n_layer=1, n_head=4,
        eos_token_id=tokenizer.eos_token_id, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
    ))
    return model, tokenizer
