import torch
from main import model, tokenizer
from src.sentana.logger import logger
from src.sentana.entity.entity import PredictionConfig


class Prediction:
    def __init__(self, config: PredictionConfig):
        """Initializes the Prediction class with configuration settings."""
        self.config: PredictionConfig = config
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model
        self.tokenizer = tokenizer

    def _preprocess_input(self, text: str) -> dict:
        """Tokenizes and prepares input text for model inference."""
        try:
            text = text.strip()
            if not text:
                logger.warning("Received empty input text.")
                return {
                    "input_ids": torch.zeros(
                        (1, self.config.params.max_length), dtype=torch.long
                    )
                }

            return self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.config.params.max_length,
                return_tensors="pt",
            )
        except Exception as e:
            logger.exception(f"Error during input preprocessing: {e}")
            raise

    def _predict(self, input_seq: dict) -> torch.Tensor:
        """Runs model inference and returns logits."""
        try:
            with torch.no_grad():
                output = self.model(
                    input_ids=input_seq["input_ids"],
                    attention_mask=input_seq.get("attention_mask"),
                )
            return output["logits"]
        except Exception as e:
            logger.exception(f"Error during model prediction: {e}")
            raise

    def _postprocess(self, logits: torch.Tensor) -> str:
        """Converts logits into a sentiment prediction."""
        try:
            sentiment: str = (
                "POSITIVE"
                if torch.argmax(torch.sigmoid(logits), dim=-1).item() == 1
                else "NEGATIVE"
            )
            logger.info(f"Prediction result: {sentiment}")
            return sentiment
        except Exception as e:
            logger.exception(f"Error during postprocessing: {e}")
            raise

    def predict(self, text: str) -> str:
        """Generates a sentiment prediction from input text."""
        try:
            logger.info("Starting prediction process...")
            preprocessed_text = self._preprocess_input(text)
            logits = self._predict(preprocessed_text)
            return self._postprocess(logits)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return ""
