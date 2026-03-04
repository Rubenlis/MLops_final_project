import torch
from qa.model import QAModel

# Test to check if model return the adequate dimensions
def test_model_output_shapes():
    batch_size = 4
    seq_len = 20
    model = QAModel(vocab_size=100, embedding_dim=32, hidden_dim=16)
    
    # Fake batch d'input_ids
    dummy_input = torch.randint(low=1, high=100, size=(batch_size, seq_len))
    start_logits, end_logits = model(dummy_input)
    
    assert start_logits.shape == (batch_size, seq_len), "Error of dimension for start_logits"
    assert end_logits.shape == (batch_size, seq_len), "Error of dimension for end_logits"


# Test to check if the mask updates the padding logits to -1e9 
def test_model_padding_mask():
    pad_idx = 0
    model = QAModel(vocab_size=100, embedding_dim=32, hidden_dim=16, pad_idx=pad_idx)
    
    dummy_input = torch.tensor([[12, 45, 2, 8, 0, 0, 0]])
    start_logits, end_logits = model(dummy_input)
    
    assert (start_logits[0, 4:] == -1e9).all(), "The mask start_logits did not work"
    assert (end_logits[0, 4:] == -1e9).all(), "The mask end_logits did not work"
