import os
import torch
import pickle
from datasets import Dataset, DatasetDict
from src.sentana.logger import logger
from src.sentana.entity.entity import DataTransformationConfig
from src.sentana.config.configuration import ConfigurationManager


class DataTransformation:
    def __init__(self, config: DataTransformationConfig) -> None:
        """Initialize DataTransformation with configuration settings."""
        self.config = config

    def _load_data(self) -> dict[str, torch.Tensor]:
        """Load dataset from the specified path."""
        try:
            logger.info(f"Loading data from {self.config.data_path}...")
            return torch.load(self.config.data_path)
        except FileNotFoundError as e:
            logger.exception(f"File not found: {e.filename}")
            raise
        except Exception as e:
            logger.exception(f"Error loading data: {e}")
            raise

    def _convert_to_dataset(self, data: dict[str, torch.Tensor]) -> Dataset:
        """Convert PyTorch tensors to a Hugging Face Dataset."""
        try:
            logger.info("Converting data to Hugging Face Dataset...")
            return Dataset.from_dict(
                {key: value.tolist() for key, value in data.items()}
            )
        except Exception as e:
            logger.exception(f"Error converting data: {e}")
            raise

    def _split_data(self, dataset: Dataset) -> DatasetDict:
        """Split the dataset into training and testing sets."""
        try:
            logger.info("Splitting dataset into train and test sets...")
            split_data = dataset.train_test_split(
                test_size=self.config.params.test_size
            )
            return DatasetDict(
                {"train": split_data["train"], "test": split_data["test"]}
            )
        except Exception as e:
            logger.exception(f"Error during train-test split: {e}")
            raise

    def _save_data(self, dataset_dict: DatasetDict) -> None:
        """Save the processed dataset as a pickle (.pkl) file."""
        try:
            logger.info("Saving processed dataset...")
            os.makedirs(self.config.root_dir, exist_ok=True)
            save_path = os.path.join(self.config.root_dir, "data.pt")
            with open(save_path, "wb") as f:
                pickle.dump(dataset_dict, f)
            logger.info("Dataset successfully saved.")
        except Exception as e:
            logger.exception(f"Error saving data: {e}")
            raise

    def run(self) -> None:
        """Execute the full data transformation pipeline."""
        try:
            logger.info("Starting data transformation pipeline...")
            data = self._load_data()
            dataset = self._convert_to_dataset(data)
            dataset_dict = self._split_data(dataset)
            self._save_data(dataset_dict)
            logger.info("Data transformation process completed successfully.")
        except Exception as e:
            logger.exception("Data transformation pipeline failed.")
            raise RuntimeError("Data transformation pipeline failed.") from e


if __name__ == "__main__":
    try:
        config = ConfigurationManager().get_data_transformation_config()
        data_transformation = DataTransformation(config=config)
        data_transformation.run()
    except Exception as e:
        logger.exception("Fatal error in data transformation pipeline.")
        raise RuntimeError("Data transformation pipeline terminated.") from e
