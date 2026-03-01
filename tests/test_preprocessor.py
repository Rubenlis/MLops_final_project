import unittest
from data_utils import SquadPreprocessor

class TestSquadPreprocessor(unittest.TestCase):
    def setUp(self):
        """Initialize the preprocessor before each test."""
        self.preprocessor = SquadPreprocessor()

    def test_tokenization_offsets(self):
        """Test if tokens correctly map to their character positions in the text."""
        text = "MLOps is great!"
        tokens = self.preprocessor.ultimate_tokenize(text)
        
        # Check first token 'mlops'
        self.assertEqual(tokens[0]['text'], "mlops")
        self.assertEqual(tokens[0]['start'], 0)
        self.assertEqual(tokens[0]['end'], 5)
        
        # Verify that the slice of the original text matches the token text
        for token in tokens:
            original_slice = text[token['start']:token['end']].lower()
            self.assertEqual(token['text'], original_slice)

    def test_process_raw_data_alignment(self):
        """Test if the answer span is correctly mapped to token indices."""
        raw_data = [{
            'context': "The capital of France is Paris.",
            'question': "What is the capital of France?",
            'answer_text': "Paris",
            'answer_start': 25
        }]
        
        processed = self.preprocessor.process_raw_data(raw_data)
        
        self.assertEqual(len(processed), 1)
        item = processed[0]
        
        # 'Paris' should be the last token before the period
        start_idx = item['start_token_idx']
        end_idx = item['end_token_idx']
        
        reconstructed_answer = item['context_tokens'][start_idx:end_idx+1]
        self.assertIn("paris", reconstructed_answer)

    def test_vocabulary_building(self):
        """Test if the vocabulary correctly includes special tokens and words."""
        cleaned_data = [{
            'context_tokens': ['the', 'model', 'works'],
            'question_tokens': ['how', 'is']
        }]
        
        vocab = self.preprocessor.build_vocab(cleaned_data)
        
        # Special tokens must be present
        self.assertIn("<PAD>", vocab)
        self.assertIn("<UNK>", vocab)
        self.assertIn("<SEP>", vocab)
        # Data tokens must be present
        self.assertIn("model", vocab)
        self.assertEqual(vocab["<PAD>"], 0)

if __name__ == '__main__':
    unittest.main()