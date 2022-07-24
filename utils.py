import pandas as pd
import datasets
import numpy as np
import regex as re
import torch
from nltk.stem import PorterStemmer
from transformers import AutoTokenizer ,AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(path):
    data=(pd.read_csv(path, index_col=0, header=[0])).reset_index(drop=True)
    data.columns=['label','text']
    data=data[['text','label']]
    return data

def preprocess_text(text):
    stemmer = PorterStemmer()
    entity_prefixes = ['@']
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                word= stemmer.stem(word)
                words.append(word)
    sentence=' '.join(words)

    # remove stock market tickers
    tweet = re.sub(r'\$\w*', '', sentence)
    # remove twitter abbreviations
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    return tweet


def split_data(data):
    train, validate, test = np.split(data.sample(frac=1), [int(.6*len(data)), int(.8*len(data))])
    return train, validate, test

def create_dateset(train,validate,test):
    train_dataset = datasets.Dataset.from_dict(train)
    test_dataset = datasets.Dataset.from_dict(test)
    validation_dataset=datasets.Dataset.from_dict(validate)
    my_dataset_dict = datasets.DatasetDict({"train":train_dataset,"validation":validation_dataset,"test":test_dataset})
    return my_dataset_dict

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


def train_model(data):

    sentiment_encoded = data.map(tokenize, batched=True, batch_size=None)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (AutoModelForSequenceClassification
                .from_pretrained(model_ckpt, num_labels=2).to(device))
    logging_steps = len(data["train"]) // batch_size
    model_name = f"{model_ckpt}-finetuned-emotion"
    training_args = TrainingArguments(output_dir=model_name,
                                        num_train_epochs=2,
                                        learning_rate=2e-5,
                                        per_device_train_batch_size=batch_size,
                                        per_device_eval_batch_size=batch_size,
                                        weight_decay=0.01,
                                        evaluation_strategy="epoch",
                                        disable_tqdm=False,
                                        logging_steps=logging_steps,
                                        push_to_hub=False, 
                                        log_level="error")
    trainer = Trainer(model=model, args=training_args, 
                        # compute_metrics=compute_metrics,
                        train_dataset=data["train"],
                        eval_dataset=data["validation"],
                        tokenizer=tokenizer)

    trainer.train()
    return model


