import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict

# HF native
from functools import partial
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score
)

DATA_NAME = "alpaca"
NUM_BUCKETS = 20  # Define the number of buckets
MAX_OUTPUT_TOKENS = 512  # Define the maximum number of tokens in the output


# Predictor
actn2actfunc = {'relu': nn.ReLU(inplace=True), 'leakyrelu': nn.LeakyReLU(inplace=True), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 'selu': nn.SELU(inplace=True), 'softplus': nn.Softplus(), 'gelu': nn.GELU(), None: nn.Identity()}

class MLPAdaptor(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list, output_dim: int, p: float, norm: str, actn: str, order: str = 'nd'):
        super(MLPAdaptor, self).__init__()
        self.n_layer = len(hidden_dims) - 1
        self.in_dim = in_dim
        
        try:
            actn = actn2actfunc[actn]
        except:
            print(actn)
            raise NotImplementedError

        # input layer
        layers = [nn.Linear(self.in_dim, hidden_dims[0]), actn]
        # hidden layers
        for i in range(self.n_layer):
            layers += self.compose_layer(
                in_dim=hidden_dims[i], out_dim=hidden_dims[i+1], norm=norm, actn=actn, p=p, order=order
            )
        # output layers
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.fc = nn.Sequential(*layers)

    def compose_layer(
        self,
        in_dim: int,
        out_dim: int,
        norm: str,
        actn: nn.Module,
        p: float = 0.0,
        order: str = 'nd'
    ):
        norm2normlayer = {'bn': nn.BatchNorm1d(in_dim), 'ln': nn.LayerNorm(in_dim), None: None, 'None': None}  # because in_dim is only fixed here
        try:
            norm = norm2normlayer[norm]
        except:
            print(norm)
            raise NotImplementedError
        # norm --> dropout or dropout --> norm
        if order == 'nd':
            layers = [norm] if norm is not None else []
            if p != 0:
                layers.append(nn.Dropout(p))
        elif order == 'dn':
            layers = [nn.Dropout(p)] if p != 0 else []
            if norm is not None:
                layers.append(norm)
        else:
            print(order)
            raise NotImplementedError

        layers.append(nn.Linear(in_dim, out_dim))
        if actn is not None:
            layers.append(actn)
        return layers

    def forward(self, x):
        output = self.fc(x)
        return output
    
    
class OutputLengthPredictor(nn.Module):
    def __init__(self, base_model: nn.Module, normalize_embeddings: bool, adaptor_output_dim: int, adaptor_hidden_dims: list = [256, 128], adaptor_dropout: float = 0.2, adaptor_norm: str = "ln", adaptor_actn: str = "relu", adaptor_order: str = "nd"):
        super(OutputLengthPredictor, self).__init__()
        self.base_model = base_model
        self.normalize_embeddings = normalize_embeddings
        self.predictor = MLPAdaptor(in_dim=self.base_model.pooler.dense.out_features, hidden_dims=adaptor_hidden_dims, output_dim=adaptor_output_dim, p=adaptor_dropout, norm=adaptor_norm, actn=adaptor_actn, order=adaptor_order) # NOTE: Currenlt inpur dimension is hard-coded (there MUST BE a pooler layer that contains a dense layer)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):
        x = self.base_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = mean_pooling(x, attention_mask)
        if self.normalize_embeddings:
            x = F.normalize(x, p=2, dim=1)
        x = self.predictor(x)
        logits = F.softmax(x, dim=1)
        
        assert labels is not None
        loss = self.loss_fn(x, labels.long())
        
        return {"loss": loss, "logits": logits}


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_token_len(sentences, tokenizer):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    return encoded_input["attention_mask"].detach().cpu().sum(dim=1).numpy()

def get_alpaca_data(tokenizer):
    alpaca = load_dataset("yahma/alpaca-cleaned")
    alpaca_df = list(alpaca.data.values())[0].to_pandas()
    alpaca_df["output_token_len"] = get_token_len(alpaca_df["output"].values.tolist(), tokenizer)
    
    assert (alpaca_df["instruction"] == "").sum() == 0
    assert (alpaca_df["input"] == "").sum() != 0
    
    alpaca_df["input_text"] = alpaca_df[["input", "instruction"]].apply(
        lambda row: 
            row[1] + (
                "" if 
                row[1].endswith(".") or row[1].endswith("?") or row[1].endswith("!") or
                row[1].endswith("\n") or row[1].endswith(":")
                else "."
            ) + (
                ("" if row[1].endswith(" ") else " ") + \
                row[0] + (
                    "" if row[1].endswith(".") or row[1].endswith("?") or 
                    row[1].endswith("!") or row[1].endswith("\n") or row[1].endswith(":") 
                    else "."
                ) if row[0] != "" else ""
            ),
        axis=1
    )

    alpaca_hf = alpaca_df[["input_text", "output_token_len"]].rename(columns={"input_text": "text"})

    # Calculate the percentiles
    percentiles = np.linspace(0, 100, NUM_BUCKETS + 1)

    # Get the bucket boundaries, with the last bucket's max set to 512
    bucket_edges = list(np.percentile(alpaca_hf['output_token_len'], percentiles[:-1])) + [MAX_OUTPUT_TOKENS]

    # Create the bucket labels (max values for each bucket)
    bucket_labels = np.arange(len(bucket_edges)-1)
    bucket_max = bucket_edges[1:]

    # Replace label values with the max value of the bucket
    alpaca_hf['label'] = pd.cut(alpaca_hf['output_token_len'], bins=bucket_edges, include_lowest=True,
                                labels=bucket_labels, 
                                )
    alpaca_hf['label'] = alpaca_hf['label'].astype(float)
    
    return alpaca_hf, bucket_max


def split_data(df, tokenizer):
    from sklearn.model_selection import train_test_split

    # Split into train, validation, and test sets
    train_full, test = train_test_split(df.drop(columns=["output_token_len"]), test_size=0.2, random_state=42)
    train_full, val = train_test_split(train_full, test_size=0.25, random_state=42)

    # Create two train datasets: one half-sized of the other
    _, train_small = train_test_split(train_full, test_size=0.5, random_state=42)

    # Convert DataFrame to Hugging Face Dataset
    train_full_ds = Dataset.from_pandas(train_full.reset_index(drop=True))
    train_small_ds = Dataset.from_pandas(train_small.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val.reset_index(drop=True))
    test_ds = Dataset.from_pandas(test.reset_index(drop=True))
    
    def preprocess_function(ds):
        tokens = tokenizer(ds["text"], padding=True, truncation=True, return_tensors='pt')
        tokens["labels"] = ds["label"]
        return tokens

    train_full_ds_tokenized = train_full_ds.map(preprocess_function, batched=True)
    train_small_ds_tokenized = train_small_ds.map(preprocess_function, batched=True)
    val_ds_tokenized = val_ds.map(preprocess_function, batched=True)
    test_ds_tokenized = test_ds.map(preprocess_function, batched=True)

    # # Combine into a DatasetDict
    # ds_dict = DatasetDict({
    #     "train_full": train_full_ds,
    #     "train_small": train_small_ds,
    #     "validation": val_ds,
    #     "test": test_ds
    # })

    # # Verify the structure of DatasetDict
    # print(ds_dict)
    
    return train_full_ds_tokenized, train_small_ds_tokenized, val_ds_tokenized, test_ds_tokenized


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)  # Get predicted class indices
    
    # Metrics
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")
    try:
        auroc = roc_auc_score(labels, logits, multi_class="ovr")
    except ValueError:
        auroc = float('nan')  # Handle cases where AUROC can't be computed
    mcc = matthews_corrcoef(labels, predictions)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auroc": auroc,
        "mcc": mcc,
    }


def train(bert_model, tokenizer, num_buckets, train_ds_tokenized, val_ds_tokenized, test_ds_tokenized):
    # Model configuration
    # TODO: change to parameters 
    config = {
        "base_model": deepcopy(bert_model), 
        "normalize_embeddings": True, 
        "adaptor_output_dim": num_buckets, 
        "adaptor_hidden_dims": [256, 128], 
        "adaptor_dropout": 0.2, 
        "adaptor_norm": "ln", 
        "adaptor_actn": "relu", 
        "adaptor_order": "nd"
    }
    model = OutputLengthPredictor(**config)

    # Training Arguments
    # TODO: change to parameters 
    training_args = TrainingArguments(
        output_dir="./results",          # Output directory
        evaluation_strategy="epoch",    # Evaluate at the end of each epoch
        per_device_train_batch_size=400,  # Batch size for training
        per_device_eval_batch_size=800,   # Batch size for evaluation
        warmup_steps=50,
        num_train_epochs=50,             # Number of training epochs
        logging_dir="./logs",           # Directory for storing logs
        logging_steps=10,
        save_strategy="epoch",          # Save checkpoint at the end of each epoch
        fp16=False,                      # Enable mixed precision
        dataloader_drop_last=True,      # Drop incomplete batches
        report_to="none",              # Disable Weights & Biases logging
        load_best_model_at_end=True,     # Load the best model based on evaluation
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_tokenized,
        eval_dataset=val_ds_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate on the test set
    test_results = trainer.evaluate(test_ds_tokenized)
    print(test_results)

    return trainer, test_results


def main():
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    bert_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', device_map="cpu")
    
    if DATA_NAME == "alpaca":
        df = get_alpaca_data(tokenizer)
    else:
        raise ValueError(f"Unknown data name: {DATA_NAME}")
    train_full_ds_tokenized, train_small_ds_tokenized, val_ds_tokenized, test_ds_tokenized = split_data(df, tokenizer)
    
    trainer, test_results = train(bert_model, tokenizer, NUM_BUCKETS, train_small_ds_tokenized, val_ds_tokenized, test_ds_tokenized)


if __name__ == "__main__":
    main()

