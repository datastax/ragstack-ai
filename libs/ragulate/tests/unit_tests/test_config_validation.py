import unittest
import os

from ragstack_ragulate.config.config_parser import ConfigParser
from ragstack_ragulate.config.config_schema_0_1 import ConfigSchema_0_1


class TestConfigValidation(unittest.TestCase):

    def test_full_config(self):
        config = {
            "version": 0.1,
            "steps": {
                "ingest": [
                    {
                        "name": "chunk_size_ingest",
                        "script": "chunk_size_experiment.py",
                        "method": "ingest",
                    }
                ],
                "query": [
                    {
                        "name": "chunk_size_query",
                        "script": "chunk_size_experiment.py",
                        "method": "query",
                    }
                ],
                "cleanup": [
                    {
                        "name": "chunk_size_cleanup",
                        "script": "chunk_size_experiment.py",
                        "method": "cleanup",
                    }
                ],
            },
            "recipes": [
                {
                    "name": "chunk_size_500",
                    "ingest": "chunk_size_ingest",
                    "query": "chunk_size_query",
                    "cleanup": "chunk_size_cleanup",
                    "ingredients": [{"chunk_size": 500}],
                },
                {
                    "name": "chunk_size_1000",
                    "ingest": "chunk_size_ingest",
                    "query": "chunk_size_query",
                    "cleanup": "chunk_size_cleanup",
                    "ingredients": [{"chunk_size": 1000}],
                },
            ],
            "datasets": [
                {"name": "blockchain_solana", "kind": "llama"},
                {"name": "braintrust_coda_help_desk", "kind": "llama"},
            ],
            "eval_llms": [
                {
                    "vendor": "open_ai",
                    "model": "gpt3.5-turbo",
                    "name": "gpt3.5",
                    "default": True,
                },
                {"name": "llama3", "vendor": "huggingface", "model": "llama3"},
            ],
            "metrics": {
                "groundedness": {"enabled": True},
                "answer_correctness": {"enabled": True, "eval_llm": "llama3"},
            },
        }
        os.makedirs(os.path.join("datasets", "llama", "blockchain_solana"), exist_ok=True)
        os.makedirs(os.path.join("datasets", "llama", "braintrust_coda_help_desk"), exist_ok=True)
        parser = ConfigParser(config_schema=ConfigSchema_0_1(), config=config)

        for field, errors in parser.errors.items():
            print(f"{field}: {errors}")

        self.assertTrue(parser.is_valid)

        config = parser.get_config()

        self.assertIn("chunk_size_500", config.recipes)
        chunk_size_500 = config.recipes["chunk_size_500"]
        self.assertEqual(chunk_size_500.cleanup.method, "cleanup")
        self.assertEqual(chunk_size_500.query.script, "chunk_size_experiment.py")
        self.assertEqual(chunk_size_500.name, "chunk_size_500")
        self.assertIn("chunk_size", chunk_size_500.ingredients)
        self.assertEqual(chunk_size_500.ingredients["chunk_size"], 500)

        self.assertIn("chunk_size_1000", config.recipes)
        chunk_size_1000 = config.recipes["chunk_size_1000"]
        self.assertEqual(chunk_size_1000.ingest.method, "ingest")
        self.assertEqual(chunk_size_1000.query.script, "chunk_size_experiment.py")
        self.assertEqual(chunk_size_1000.name, "chunk_size_1000")
        self.assertIn("chunk_size", chunk_size_1000.ingredients)
        self.assertEqual(chunk_size_1000.ingredients["chunk_size"], 1000)

    def test_minimal_config(self):
        config = {
            "version": 0.1,
            "steps": {
                "query": [
                    {"name": "minimal", "script": "minimal.py", "method": "query"}
                ],
            },
            "recipes": [
                {"query": "minimal", "ingredients": [{"chunk_size": 500}]},
                {"query": "minimal", "ingredients": [{"chunk_size": 1000}]},
            ],
            "datasets": ["blockchain_solana", "other_dataset"],
            "eval_llms": ["gpt3.5-turbo"],
            "metrics": [
                "groundedness",
                "answer_correctness",
            ],
        }
        os.makedirs(os.path.join("datasets", "llama", "blockchain_solana"), exist_ok=True)
        os.makedirs(os.path.join("datasets", "llama", "other_dataset"), exist_ok=True)
        parser = ConfigParser(config_schema=ConfigSchema_0_1(), config=config)

        for field, errors in parser.errors.items():
            print(f"{field}: {errors}")

        self.assertTrue(parser.is_valid)

        config = parser.get_config()
        self.assertIn("chunk_size_500", config.recipes)
        chunk_size_500 = config.recipes["chunk_size_500"]
        self.assertIsNone(chunk_size_500.cleanup)
        self.assertEqual(chunk_size_500.query.script, "minimal.py")
        self.assertEqual(chunk_size_500.name, "chunk_size_500")
        self.assertIn("chunk_size", chunk_size_500.ingredients)
        self.assertEqual(chunk_size_500.ingredients["chunk_size"], 500)

        self.assertIn("chunk_size_1000", config.recipes)
        chunk_size_1000 = config.recipes["chunk_size_1000"]
        self.assertIsNone(chunk_size_500.ingest)
        self.assertEqual(chunk_size_1000.query.script, "minimal.py")
        self.assertEqual(chunk_size_1000.name, "chunk_size_1000")
        self.assertIn("chunk_size", chunk_size_1000.ingredients)
        self.assertEqual(chunk_size_1000.ingredients["chunk_size"], 1000)
