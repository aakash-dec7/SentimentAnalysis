from pathlib import Path
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    download_path: Path


@dataclass
class DataValidationConfig:
    data_path: Path
    schema: dict


@dataclass
class DataPreprocessingConfig:
    root_dir: Path
    data_path: Path
    model_name: str
    tokenizer_path: Path
    params: dict
    schema: dict


@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    params: dict


@dataclass
class ModelConfig:
    model_params: dict


@dataclass
class ModelTrainingConfig:
    root_dir: Path
    data_path: Path
    tokenizer_path: Path
    params: dict


@dataclass
class PredictionConfig:
    model_path: Path
    tokenizer_path: Path
    params: dict
