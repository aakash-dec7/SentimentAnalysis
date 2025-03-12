import os
import torch
import pickle
from transformers import (
    Trainer,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from src.sentana.logger import logger
from src.sentana.component.model import Model
from src.sentana.entity.entity import ModelTrainingConfig
from src.sentana.config.configuration import ConfigurationManager


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig) -> None:
        """Initialize model training setup."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._initialize_model()
        self.tokenizer = self._load_tokenizer()
        self.dataloader = self._load_data()
        logger.info("Model training initialized.")

    def _initialize_model(self) -> Model:
        """Initialize the model and move it to the appropriate device."""
        model = Model(config=ConfigurationManager().get_model_config()).to(self.device)
        return model

    def _load_tokenizer(self) -> AutoTokenizer:
        """Load tokenizer from the specified file."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
            logger.info("Tokenizer loaded successfully.")
            return tokenizer
        except Exception as e:
            logger.exception("Failed to load tokenizer: %s", str(e))
            raise

    def _load_data(self) -> dict:
        """Load the dataset from a pickle (.pkl) file."""
        try:
            with open(self.config.data_path, "rb") as f:
                data = pickle.load(f)
            logger.info("Data loaded successfully.")
            return {"train": data["train"], "test": data["test"]}
        except Exception as e:
            logger.exception("Failed to load data: %s", str(e))
            raise

    def _get_training_args(self) -> TrainingArguments:
        """Define and return training arguments."""
        output_dir = os.path.join(self.config.root_dir, "results")
        logging_dir = os.path.join(self.config.root_dir, "logs")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)

        return TrainingArguments(
            output_dir=output_dir,
            save_steps=25,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=logging_dir,
            logging_steps=10,
            learning_rate=5e-5,
            lr_scheduler_type="linear",
            fp16=True,
            gradient_accumulation_steps=2,
            report_to=[],
        )

    def _train(self) -> None:
        """Train the model using the Trainer API."""
        try:
            trainer = Trainer(
                model=self.model,
                args=self._get_training_args(),
                train_dataset=self.dataloader["train"],
                eval_dataset=self.dataloader["test"],
                data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            )
            trainer.train()
            logger.info("Training complete.")
        except Exception as e:
            logger.error("Training error: %s", str(e))
            raise

    def _save_model(self) -> None:
        """Save the trained model."""
        try:
            model_path = os.path.join(self.config.root_dir, "model.pth")
            torch.save(self.model.state_dict(), model_path)
            logger.info("Model saved successfully at: %s", model_path)
        except Exception as e:
            logger.error("Error saving model: %s", str(e))
            raise

    def run(self) -> None:
        """Execute the training pipeline."""
        self._train()
        self._save_model()


if __name__ == "__main__":
    try:
        config = ConfigurationManager().get_model_training_config()
        trainer = ModelTraining(config=config)
        trainer.run()
    except Exception as e:
        logger.exception("Model training pipeline failed.")
        raise
