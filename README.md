# Sentiment Analysis

This repository contains a **BERT-based** sentiment analysis model trained on the **IMDB dataset**. The model is implemented using `PyTorch` and follows a modular architecture for enhanced readability, maintainability, and scalability.

## Features

**PyTorch Implementation**: Fully built using `PyTorch` for efficient deep learning workflows.

**Data Version Control (DVC)**: Manages training and evaluation pipelines effectively.

**Transformer AutoTokenizer**: Uses `bert-base-uncased` for robust text processing.

**Experiment Tracking & Model Management**: Integrated with `MLflow` and `DagsHub` for seamless tracking.

**Containerized Deployment**: Docker images stored in `Amazon Elastic Container Registry (ECR)`.

**Scalable Deployment**: Model deployed on `Amazon Elastic Kubernetes Service (EKS)` for production readiness.

**Automated CI/CD**: End-to-end deployment automation using AWS and GitHub Actions.

## Prerequisites

Ensure the following dependencies and services are installed and configured:

- Python 3.10
- AWS Account
- AWS CLI
- Docker Desktop (for local image testing)
- DagsHub Account (for experiment tracking)
- Git & GitHub (for version control)

## Dataset

**Source:** [IMDB Movies Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data)

**Description:**
The dataset consists of two columns:

- **Review**
- **Sentiment**

## Model Architecture

The **SentimentAnalysis** model consists of the following components:

### 1. Base Model (BERT)

- Uses a pre-trained **BERT model** from Hugging Face's `transformers` library.
- Dynamically loads configurations via `AutoConfig`.
- Extracts **contextualized word representations** from input sequences.

### 2. Custom Fully Connected Layers

- A feed-forward network is added after the BERT encoder:
  - **Linear Layer 1:** Projects BERT’s hidden size (768) to 1024 neurons.
  - **ReLU Activation:** Introduces non-linearity.
  - **Linear Layer 2:** Maps 1024 neurons to `out_features` (number of classes).
  - **Dropout Layer:** Prevents overfitting by randomly deactivating neurons.

### 3. Classifier Head

- Replaces the original BERT classification head with a custom linear layer:
  - **Linear Layer:** Ensures compatibility with the new architecture.

#### Model Summary

```text
Model(
  (bert): BertForSequenceClassification(
    (bert): BertModel(
      (embeddings): BertEmbeddings(
        (word_embeddings): Embedding(30522, 768, padding_idx=0)
        (position_embeddings): Embedding(512, 768)
        (token_type_embeddings): Embedding(2, 768)
        (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (encoder): BertEncoder(
        (layer): ModuleList(
          (0-11): 12 x BertLayer(
            (attention): BertAttention(
              (self): BertSdpaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
              (intermediate_act_fn): GELUActivation()
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
      )
      (pooler): BertPooler(
        (dense): Linear(in_features=768, out_features=768, bias=True)
        (activation): Tanh()
      )
    )
    (dropout): Dropout(p=0.1, inplace=False)
    (classifier): Linear(in_features=1024, out_features=1024, bias=True)
  )
  (custom_layer): Sequential(
    (0): Linear(in_features=768, out_features=1024, bias=True)
    (1): ReLU()
    (2): Linear(in_features=1024, out_features=1024, bias=True)
    (3): Dropout(p=0.2, inplace=False)
  )
)
```

## Installation

Clone the repository:

```sh
git clone https://github.com/aakash-dec7/SentimentAnalysis.git
cd SentimentAnalysis
```

### Install Dependencies

```sh
pip install -r requirements.txt
```

### Initialize DVC Pipeline

```sh
dvc init
```

## DVC Pipeline Stages

1. **Data Ingestion** - Fetches and stores the raw dataset.
2. **Data Validation** - Ensures data quality and integrity.
3. **Data Preprocessing** - Cleans, tokenizes, and prepares the dataset.
4. **Data Transformation** - Converts processed data into a model-friendly format.
5. **Model Definition** - Defines the model architecture.
6. **Model Training** - Trains the model using the **Hugging Face Trainer API**.

### Run Training and Evaluation

```sh
dvc repro
```

The trained model will be saved in: `artifacts/model/model.pth`

## Deployment

### 1. Create an Amazon ECR Repository

Ensure the ECR repository name matches the project name defined in `setup.py`:

```python
setup(
    name="sentana",
    version="1.0.0",
    packages=find_packages(),
)
```

### 2. Create an Amazon EKS Cluster

Use the following command to create an **Amazon EKS cluster**:

```sh
eksctl create cluster --name <cluster-name> \
    --region <region> \
    --nodegroup-name <nodegroup-name> \
    --nodes <number-of-nodes> \
    --nodes-min <nodes-min> \
    --nodes-max <nodes-max> \
    --node-type <node-type> \
    --managed
```

### 3. Push Code to GitHub

Before pushing, add **GitHub Actions secrets** in **Settings > Secrets and Variables > Actions**:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`
- `ECR_REGISTRY_URI`

Push the code to GitHub:

```sh
git add .
git commit -m "Initial commit"
git push origin main
```

### 4. CI/CD Automation

GitHub Actions will automate the CI/CD process, ensuring that the model is built, tested, and deployed to **Amazon EKS**.

## Accessing the Deployed Application

Once deployed, follow these steps:

1. Navigate to **EC2 Instances** in the **AWS Console**.
2. Go to **Security Groups** and update inbound rules to allow traffic.

Retrieve the external IP of the deployed service:

```sh
kubectl get svc
```

Copy the `EXTERNAL-IP` and append `:5000` to access the application:

```text
http://<EXTERNAL-IP>:5000
```

The SentimentAnalysis application is now deployed and accessible online.

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
