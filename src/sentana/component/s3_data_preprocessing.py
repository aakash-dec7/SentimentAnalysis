import os
import torch
import pandas as pd
from transformers import AutoTokenizer
from src.sentana.logger import logger
from src.sentana.utils.utils import update_yaml_file
from src.sentana.entity.entity import DataPreprocessingConfig
from src.sentana.config.configuration import ConfigurationManager


class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        """Initialize tokenizer and configuration."""
        self.config: DataPreprocessingConfig = config
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name
        )
        self.tokenizer.add_special_tokens({"bos_token": "[BOS]", "eos_token": "[EOS]"})
        self.data: pd.DataFrame | None = None

    def _load_data(self) -> None:
        """Load dataset from the specified path."""
        logger.info(f"Loading dataset from: {self.config.data_path}")
        try:
            self.data = pd.read_csv(self.config.data_path, nrows=10)
        except FileNotFoundError:
            logger.exception(f"File not found: {self.config.data_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.exception("The provided CSV file is empty.")
            raise
        except Exception as e:
            logger.exception(f"Error loading dataset: {str(e)}")
            raise

    def _remove_missing_values(self) -> None:
        """Remove rows with missing values in required columns."""
        if self.data is None:
            logger.error("Data not loaded. Skipping missing value removal.")
            return

        required_columns = set(self.config.schema.get("required_columns", []))
        initial_shape = self.data.shape
        self.data.dropna(subset=required_columns, inplace=True)
        logger.info(
            f"Removed missing values. Shape changed from {initial_shape} to {self.data.shape}."
        )

    def _convert_labels(self) -> None:
        """Convert labels from categorical to numerical values."""
        logger.info("Converting labels.")
        try:
            self.data[self.config.schema.target_column] = self.data[
                self.config.schema.target_column
            ].map({"positive": 1, "negative": 0})
        except KeyError as e:
            logger.exception(f"Missing required column: {str(e)}")
            raise

    def _tokenize_texts(self) -> dict:
        """Tokenize text data using a pretrained tokenizer."""
        logger.info("Tokenizing text data.")
        try:
            texts = self.data[self.config.schema.input_column].tolist()
            return self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.params.max_length,
                return_tensors="pt",
            )
        except Exception as e:
            logger.exception(f"Error during tokenization: {str(e)}")
            raise

    def _save_tokenizer(self) -> None:
        """Save tokenizer to disk for later use."""
        logger.info("Saving tokenizer.")
        try:
            os.makedirs(self.config.tokenizer_path, exist_ok=True)
            tokenizer_path = os.path.join(self.config.tokenizer_path, "tokenizer")
            self.tokenizer.save_pretrained(tokenizer_path)
            logger.info(f"Tokenizer saved at: {tokenizer_path}.")
        except Exception as e:
            logger.exception(f"Error saving tokenizer: {str(e)}")
            raise

    def _update_vocab_size(self) -> None:
        """Update vocabulary size in the configuration file."""
        logger.info("Updating vocabulary size.")
        try:
            vocab_size = len(self.tokenizer)
            update_yaml_file(
                "hyperparameters", "vocab_size", vocab_size, "config/params.yaml"
            )
            logger.info("Vocabulary size updated in params.yaml.")
        except Exception as e:
            logger.exception(f"Error updating vocabulary size: {str(e)}")
            raise

    def _save_preprocessed_data(self, tokenized_dataset: dict) -> None:
        """Save the tokenized dataset to a file."""
        logger.info("Saving preprocessed data.")
        try:
            os.makedirs(self.config.root_dir, exist_ok=True)
            data_path = os.path.join(self.config.root_dir, "data.pt")
            torch.save(tokenized_dataset, data_path)
            logger.info(f"Preprocessed data saved at: {data_path}")
        except Exception as e:
            logger.exception(f"Error saving preprocessed data: {str(e)}")
            raise

    def run(self) -> None:
        """Execute the complete data preprocessing pipeline."""
        self._load_data()
        self._remove_missing_values()
        self._convert_labels()
        tokenized_data = self._tokenize_texts()
        labels = torch.tensor(self.data[self.config.schema.target_column].tolist())
        tokenized_dataset = {"label": labels, **tokenized_data}
        self._save_tokenizer()
        self._update_vocab_size()
        self._save_preprocessed_data(tokenized_dataset)
        logger.info("Data preprocessing completed successfully.")


if __name__ == "__main__":
    try:
        data_preprocessing_config = (
            ConfigurationManager().get_data_preprocessing_config()
        )
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
        data_preprocessing.run()
    except Exception as e:
        logger.exception("Data preprocessing pipeline failed.")
        raise RuntimeError("Data preprocessing pipeline failed.") from e
