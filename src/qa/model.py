import torch
import torch.nn as nn


class QAModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pad_idx=0, dropout_rate=0.2):
        super(QAModel, self).__init__()
        self.pad_idx = pad_idx

        # 1) Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # NOTE: LSTM dropout is only applied if num_layers > 1.
        # We keep num_layers=1 for simplicity, and apply dropout manually.
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True,
            num_layers=1
        )

        self.dropout = nn.Dropout(dropout_rate)

        # 3) Output layers (start/end)
        self.start_linear = nn.Linear(hidden_dim * 2, 1)
        self.end_linear = nn.Linear(hidden_dim * 2, 1)

    def forward(self, input_ids: torch.Tensor):
        # mask padding tokens
        mask = (input_ids == self.pad_idx)

        # embeddings + dropout
        embeds = self.dropout(self.embedding(input_ids))

        # BiLSTM
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)

        # logits
        start_logits = self.start_linear(lstm_out).squeeze(-1)  # [B, L]
        end_logits = self.end_linear(lstm_out).squeeze(-1)      # [B, L]

        # mask padding logits
        start_logits = start_logits.masked_fill(mask, -1e9)
        end_logits = end_logits.masked_fill(mask, -1e9)

        return start_logits, end_logits