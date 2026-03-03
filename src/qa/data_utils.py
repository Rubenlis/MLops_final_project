import re
import torch
from torch.utils.data import Dataset


class SquadPreprocessor:
    """
    Handles data cleaning, tokenization with offset mapping,
    and vocabulary building for SQuAD 2.0 (answerable examples).
    """
    def __init__(self):
        # Special tokens
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<SEP>": 2}

    def ultimate_tokenize(self, text: str):
        """
        Fine-grained tokenization while preserving character positions (offsets).
        Matches alphanumeric words or any non-whitespace punctuation.
        """
        tokens = []
        for match in re.finditer(r"\w+|[^\w\s]", text):
            tokens.append({
                "text": match.group().lower(),
                "start": match.start(),
                "end": match.end()
            })
        return tokens

    def process_raw_data(self, raw_data):
        """
        raw_data must be a LIST of dict with keys:
          context, question, answer_text, answer_start
        Aligns answer spans with token indices.
        Returns a list of processed dictionaries.
        """
        cleaned = []
        for item in raw_data:
            context_raw = item["context"]
            question_raw = item["question"]
            answer_text = item["answer_text"]
            ans_start = item["answer_start"]

            # skip impossible / invalid
            if ans_start is None or ans_start < 0 or not isinstance(answer_text, str) or len(answer_text) == 0:
                continue

            c_tokens_meta = self.ultimate_tokenize(context_raw)
            q_tokens_meta = self.ultimate_tokenize(question_raw)

            ans_end = ans_start + len(answer_text)
            ans_last_char = ans_end - 1

            start_token_idx = -1
            end_token_idx = -1

            # Map character-level answer positions to token-level indices (robust end mapping)
            for i, t in enumerate(c_tokens_meta):
                if start_token_idx == -1 and (t["start"] <= ans_start < t["end"]):
                    start_token_idx = i
                if (t["start"] <= ans_last_char < t["end"]):
                    end_token_idx = i
                    break

            # Only include examples where the answer was successfully mapped
            if start_token_idx != -1 and end_token_idx != -1 and end_token_idx >= start_token_idx:
                cleaned.append({
                    "context_tokens": [t["text"] for t in c_tokens_meta],
                    "question_tokens": [t["text"] for t in q_tokens_meta],
                    "start_token_idx": start_token_idx,
                    "end_token_idx": end_token_idx
                })

        return cleaned

    def build_vocab(self, cleaned_data):
        """
        Generates the word-to-index mapping from the processed dataset.
        """
        for item in cleaned_data:
            for word in item["question_tokens"] + item["context_tokens"]:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
        return self.vocab


class SquadDataset(Dataset):
    """
    PyTorch Dataset class for SQuAD.
    Prepares input tensors: [Question] + [<SEP>] + [Context]
    """
    INVALID_LABEL = -100  # for CrossEntropyLoss(ignore_index=-100)

    def __init__(self, data, word2idx, max_len=512):
        self.data = data
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Concatenate Question and Context with a separator
        tokens = item["question_tokens"] + ["<SEP>"] + item["context_tokens"]
        input_ids = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in tokens]

        # Calculate offset for start/end positions due to the question and <SEP> prefix
        token_offset = len(item["question_tokens"]) + 1
        start_pos = item["start_token_idx"] + token_offset
        end_pos = item["end_token_idx"] + token_offset

        # Truncation first
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]

        # If the answer span is outside max_len, ignore it in the loss
        if start_pos >= self.max_len or end_pos >= self.max_len or start_pos < 0 or end_pos < 0:
            start_pos = self.INVALID_LABEL
            end_pos = self.INVALID_LABEL

        # Padding
        if len(input_ids) < self.max_len:
            input_ids += [self.word2idx["<PAD>"]] * (self.max_len - len(input_ids))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "start_position": torch.tensor(start_pos, dtype=torch.long),
            "end_position": torch.tensor(end_pos, dtype=torch.long),
        }