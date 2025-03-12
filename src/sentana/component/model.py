import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig
from src.sentana.config.configuration import ConfigurationManager


class CustomBERT(nn.Module):
    def __init__(self, model_name, out_features, dropout):
        super().__init__()

        try:
            self.bert_config = AutoConfig.from_pretrained(model_name)
            self.bert = AutoModelForSequenceClassification.from_pretrained(
                model_name, config=self.bert_config
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load BERT model: {e}")

        self.custom_layer = nn.Sequential(
            nn.Linear(self.bert_config.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_features),
            nn.Dropout(dropout),
        )

        self.bert.classifier = nn.Linear(out_features, out_features)

    def forward(self, input_ids, attention_mask, labels=None):
        try:
            outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_representation = outputs.last_hidden_state[:, 0, :]
            custom_output = self.custom_layer(cls_representation)
            logits = self.bert.classifier(custom_output)

            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits, labels)

            return {"loss": loss, "logits": logits}
        except Exception as e:
            raise RuntimeError(f"Error during forward pass: {e}")


class Model(CustomBERT):
    def __init__(self, config):
        super().__init__(
            model_name=config.model_params.model_name,
            out_features=config.model_params.out_features,
            dropout=config.model_params.dropout,
        )


if __name__ == "__main__":
    try:
        config = ConfigurationManager().get_model_config()
        model = Model(config)
        print("Model initialized successfully.")
    except Exception as e:
        raise RuntimeError("Model initialization failed.") from e
