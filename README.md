# Sentiment Analysis

This repository contains a BERT-based model for sentiment analysis on the IMDB dataset. The model is implemented using PyTorch and follows a modular structure for improved readability, maintainability, and scalability.

## Features

- **PyTorch** implementation.
- **Data Version Control (DVC)** for managing training and evaluation pipelines.
- **BERT Tokenizer**: Utilizes the `bert-base-uncased` tokenizer for efficient text processing.
- **MLflow and DagsHub** for experiment tracking and model management.
- **Amazon Elastic Container Registry (ECR)** for storing Docker images.
- **Amazon Elastic Kubernetes Service (EKS)** for deploying the model as a containerized application.
- **Complete CI/CD Implementation** using AWS and GitHub Actions for automated deployment.

## Prerequisites

Ensure the following dependencies and services are installed and configured:

- Python 3.10
- AWS Account
- AWS CLI
- Docker Desktop (for local image testing)
- DagsHub Account (for experiment tracking)
- Git

## Dataset

**Source:** [IMDB Movies Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data)

**Description:**
The dataset comprises two columns:

- **Review**
- **Sentiment**

## Model Architecture

The architecture consists of the following components:

### 1. Base Model (BERT)

- Uses a pre-trained BERT model from Hugging Face’s `transformers` library.
- Loads the configuration dynamically via `AutoConfig`.
- Extracts contextualized word representations from the input sequence.

### 2. Custom Fully Connected Layers

- A feed-forward network is added after the BERT encoder output:
  - **Linear Layer 1:** Maps BERT’s hidden size to 1024 neurons.
  - **ReLU Activation:** Introduces non-linearity for better feature representation.
  - **Linear Layer 2:** Projects 1024 neurons to `out_features` (number of output classes).
  - **Dropout Layer:** Prevents overfitting by randomly deactivating neurons.

### 3. Classifier Head

- The original BERT classification head is replaced with a custom linear layer:
  - **Linear Layer:** Maps `out_features` to `out_features` (ensuring compatibility with the new architecture).

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
cd SequenceToSequence
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
2. **Data Validation** - Ensures data quality and integrity before processing.
3. **Data Preprocessing** - Cleans, tokenizes, and prepares the dataset for transformation.
4. **Data Transformation** - Converts processed data into a format suitable for model training.
5. **Model Definition** - Defines the model architecture.
6. **Model Training** - Trains the model using the tranformers trainer.

### Run Model Training and Evaluation

```sh
dvc repro
```

The trained model will be saved in the project directory: `artifacts/model/model.pth`

## Deployment

### Create an ECR Repository

Create an Amazon ECR repository with the same name as specified in `setup.py`:

```python
setup(
    name="sentana",
    version="1.0.0",
    author="Aakash Singh",
    author_email="aakash.dec7@gmail.com",
    packages=find_packages(),
)
```

### Create an EKS Cluster

Execute the following command to create an Amazon EKS cluster:

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

### Push Code to GitHub

Before pushing the code, ensure that the necessary GitHub Actions secrets are added under **Settings > Secrets and Variables > Actions**:

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

### CI/CD Automation

GitHub Actions will automate the CI/CD process, ensuring that the model is built, tested, and deployed on the EKS cluster.

## Accessing the Deployed Application

Once deployment is successful:

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

Your English-to-French translation application is now deployed and accessible online.
