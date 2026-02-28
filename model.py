import torch
import torch.nn as nn

class QAModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pad_idx=0, dropout_rate=0.2):
        super(QAModel, self).__init__()
        self.pad_idx = pad_idx # Store padding index
        
        # 1. Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # 2. BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            bidirectional=True, 
            batch_first=True,
            dropout=dropout_rate if dropout_rate > 0 else 0
        )
        
        # 3. Output layers
        self.start_linear = nn.Linear(hidden_dim * 2, 1)
        self.end_linear = nn.Linear(hidden_dim * 2, 1)


    def forward(self, input_ids):
        # Create mask with shape: [batch_size, seq_len]
        mask = (input_ids == self.pad_idx)
        
        # 1. Get embeddings
        embeds = self.embedding(input_ids)
        
        # 2. Pass through BiLSTM
        lstm_out, _ = self.lstm(embeds)
        
        # 3. Get raw logits
        start_logits = self.start_linear(lstm_out).squeeze(-1)
        end_logits = self.end_linear(lstm_out).squeeze(-1)
        
        # 4. Apply mask: replace padding logits with -1e9  (value close to negative infinite)
        # softmax will turn later these -1e9 into 0 probability
        start_logits.masked_fill_(mask, -1e9)
        end_logits.masked_fill_(mask, -1e9)
        
        return start_logits, end_logits