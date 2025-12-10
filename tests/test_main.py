import unittest
import torch
import os
import json
from PIL import Image
import numpy as np
from omegaconf import OmegaConf

# --- Mock Transformers Objects ---
# We create minimal, mock versions of Hugging Face classes to avoid downloading real models.

class MockTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0, "<eos>": 1, "<image>": 2, "hello": 3, "world": 4}
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, return_tensors="pt", **kwargs):
        tokens = [self.vocab.get(t, 0) for t in text.split()]
        return {"input_ids": torch.tensor([tokens])}

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token)

    def apply_chat_template(self, messages, **kwargs):
        # Simulate chat template by just concatenating content
        text = " ".join([m['content'] for m in messages])
        return self("hello world") # Return fixed tokens for simplicity

class MockProcessor:
    def __call__(self, images, return_tensors="pt"):
        # Return a dummy tensor of the expected shape
        return {"pixel_values": torch.randn(len(images), 3, 224, 224)}

# --- Project Modules (Import after mocks are defined) ---
from dataset import CCDataset, create_dataloader
from modeling.model import FireboltLMForCausalLM, FireboltLMConfig
from train import train, eval_model

class TestDataset(unittest.TestCase):
    def setUp(self):
        """Set up a dummy dataset for testing."""
        self.test_dir = "tests/temp_data"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create dummy image file
        self.image_path = os.path.join(self.test_dir, "dummy.jpg")
        Image.new('RGB', (10, 10)).save(self.image_path)

        # Create dummy annotation file
        self.json_path = os.path.join(self.test_dir, "annotations.json")
        self.dummy_data = [
            {"image": "dummy.jpg", "conversations": [{"from": "human", "value": "hello"}, {"from": "assistant", "value": "world"}]},
            {"image": "dummy.jpg", "conversations": [{"from": "human", "value": "hello"}, {"from": "assistant", "value": "world"}]}
        ]
        with open(self.json_path, 'w') as f:
            json.dump(self.dummy_data, f)

        self.tokenizer = MockTokenizer()
        self.processor = MockProcessor()

    def test_dataset_loading(self):
        """Test if the CCDataset loads items correctly."""
        dataset = CCDataset(self.test_dir, self.json_path, self.tokenizer, self.processor)
        self.assertEqual(len(dataset), 2)
        
        item = dataset[0]
        self.assertIn("input_ids", item)
        self.assertIn("pixel_values", item)
        self.assertEqual(item["pixel_values"].shape, (3, 224, 224))

    def test_dataloader_creation(self):
        """Test if the create_dataloader function works and produces batches."""
        dataloaders = create_dataloader(
            image_path=self.test_dir,
            json_path=self.json_path,
            tokenizer=self.tokenizer,
            processor=self.processor,
            batch_size=2,
            train_val_split=[1.0, 0.0] # Use all data for training
        )
        train_loader = dataloaders["train_dataloader"]
        batch = next(iter(train_loader))

        self.assertIn("input_ids", batch)
        self.assertIn("pixel_values", batch)
        self.assertEqual(batch["input_ids"].shape[0], 2)
        self.assertEqual(batch["pixel_values"].shape, (2, 3, 224, 224))

    def tearDown(self):
        """Clean up temporary files."""
        os.remove(self.image_path)
        os.remove(self.json_path)
        os.rmdir(self.test_dir)

class TestModel(unittest.TestCase):
    def setUp(self):
        """Set up a small, random model for testing."""
        self.config = FireboltLMConfig(
            lm_name_or_path="mock", # Prevent download
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            vision_hidden_size=48, # Dummy size
            vision_ckpt_path=None # No real vision encoder
        )
        # Monkey-patch AutoModelForCausalLM to return a small random model
        from transformers import AutoModelForCausalLM
        AutoModelForCausalLM.from_pretrained = self.mock_from_pretrained
        self.model = FireboltLMForCausalLM(self.config)

    def mock_from_pretrained(self, path, **kwargs):
        from transformers.models.gpt2.modeling_gpt2 import GPT2Model
        from transformers.models.gpt2.configuration_gpt2 import GPT2Config
        # Return a tiny, random GPT2Model as the base LM
        return GPT2Model(GPT2Config(vocab_size=self.config.vocab_size, n_embd=self.config.hidden_size, n_layer=2, n_head=2))

    def test_forward_pass_text_only(self):
        """Test the model's forward pass with only text."""
        input_ids = torch.randint(0, self.config.vocab_size, (2, 10))
        outputs = self.model(input_ids=input_ids, labels=input_ids)
        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.logits.shape, (2, 10, self.config.vocab_size))

    def test_forward_pass_multimodal(self):
        """Test the model's forward pass with text and images."""
        input_ids = torch.randint(0, self.config.vocab_size, (2, 10))
        # The model's vision_encoder is None, so we pass pre-computed embeds
        vision_embeds = torch.randn(2, self.config.vision_output_tokens, self.config.hidden_size)
        
        outputs = self.model(input_ids=input_ids, vision_embeds=vision_embeds, labels=input_ids)
        self.assertIsNotNone(outputs.loss)
        # Note: Shape might change depending on fusion strategy (prepend)
        expected_seq_len = 10 + self.config.vision_output_tokens
        self.assertEqual(outputs.logits.shape, (2, expected_seq_len, self.config.vocab_size))

class TestTrainingPipeline(unittest.TestCase):
    def setUp(self):
        """Set up a minimal environment for testing the training pipeline."""
        self.model_tester = TestModel()
        self.model_tester.setUp()
        self.model = self.model_tester.model

        self.dataset_tester = TestDataset()
        self.dataset_tester.setUp()

        self.cfg = OmegaConf.create({
            'training': {
                'num_epochs': 1,
                'device': 'cpu',
                'results_dir': 'tests/temp_results',
                'optimizer': {'lr': 1e-4},
                'scheduler': {'T_max': 1, 'eta_min': 1e-6}
            }
        })
        os.makedirs(self.cfg.training.results_dir, exist_ok=True)

    def test_training_loop(self):
        """Test that the main training function runs for one epoch without errors."""
        dataloaders = create_dataloader(
            image_path=self.dataset_tester.test_dir,
            json_path=self.dataset_tester.json_path,
            tokenizer=self.dataset_tester.tokenizer,
            processor=self.dataset_tester.processor,
            batch_size=1,
            train_val_split=[0.5, 0.5]
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.training.optimizer.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1, eta_min=1e-6)

        try:
            train(
                self.cfg, self.model, dataloaders['train_dataloader'], 
                dataloaders['val_dataloader'], optimizer, scheduler, self.cfg.training.device
            )
        except Exception as e:
            self.fail(f"Training loop failed with an exception: {e}")

    def tearDown(self):
        self.dataset_tester.tearDown()
        # Clean up results directory
        for f in os.listdir(self.cfg.training.results_dir):
            os.remove(os.path.join(self.cfg.training.results_dir, f))
        os.rmdir(self.cfg.training.results_dir)

if __name__ == "__main__":
    unittest.main()
