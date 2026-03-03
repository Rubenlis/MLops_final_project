import json
import re
import torch
from torch.utils.data import Dataset

class SquadPreprocessor:
    """
    Handles data cleaning, tokenization with offset mapping, 
    and vocabulary building for SQuAD 2.0.
    """
    def __init__(self):
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<SEP>": 2}
        
    def ultimate_tokenize(self, text):
        """
        Fine-grained tokenization while preserving character positions (offsets).
        This ensures high accuracy when mapping answers back to the original text.
        """
        tokens = []
        # Matches alphanumeric words or any non-whitespace punctuation
        for match in re.finditer(r'\w+|[^\w\s]', text):
            tokens.append({
                'text': match.group().lower(),
                'start': match.start(),
                'end': match.end()
            })
        return tokens

    def process_raw_data(self, raw_data):
        """
        Cleans raw SQuAD JSON data and aligns answer spans with token indices.
        Returns a list of processed dictionaries.
        """
        cleaned = []
        for item in raw_data:
            context_raw = item['context']
            c_tokens_meta = self.ultimate_tokenize(context_raw)
            q_tokens_meta = self.ultimate_tokenize(item['question'])
            
            ans_start = item['answer_start']
            ans_end = ans_start + len(item['answer_text'])
            
            start_token_idx = -1
            end_token_idx = -1
            
            # Map character-level answer positions to token-level indices
            for i, t in enumerate(c_tokens_meta):
                if t['start'] <= ans_start < t['end']:
                    start_token_idx = i
                if t['start'] < ans_end <= t['end']:
                    end_token_idx = i
                    break
            
            # Only include examples where the answer was successfully mapped
            if start_token_idx != -1 and end_token_idx != -1:
                cleaned.append({
                    'context_tokens': [t['text'] for t in c_tokens_meta],
                    'question_tokens': [t['text'] for t in q_tokens_meta],
                    'start_token_idx': start_token_idx,
                    'end_token_idx': end_token_idx
                })
        return cleaned

    def build_vocab(self, cleaned_data):
        """
        Generates the word-to-index mapping from the processed dataset.
        """
        for item in cleaned_data:
            for word in item['question_tokens'] + item['context_tokens']:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
        return self.vocab

class SquadDataset(Dataset):
    """
    PyTorch Dataset class for SQuAD. 
    Prepares input tensors: [Question] + [<SEP>] + [Context]
    """
    def __init__(self, data, word2idx, max_len=512):
        self.data = data
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Concatenate Question and Context with a separator
        tokens = item['question_tokens'] + ["<SEP>"] + item['context_tokens']
        input_ids = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in tokens]
        
        # Calculate offset for start/end positions due to the question and <SEP> prefix
        token_offset = len(item['question_tokens']) + 1
        start_pos = item['start_token_idx'] + token_offset
        end_pos = item['end_token_idx'] + token_offset

        # Handle Padding and Truncation
        if len(input_ids) < self.max_len:
            input_ids += [self.word2idx["<PAD>"]] * (self.max_len - len(input_ids))
        else:
            input_ids = input_ids[:self.max_len]
            # If the answer span is outside max_len, reset positions to 0 (PAD)
            if start_pos >= self.max_len: start_pos = 0
            if end_pos >= self.max_len: end_pos = 0

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'start_position': torch.tensor(start_pos, dtype=torch.long),
            'end_position': torch.tensor(end_pos, dtype=torch.long)
        }