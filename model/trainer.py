import datasets
import numpy as np
import pandas as pd
import regex as re
import torch
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class trainer:
    def __init__(
        self,
        model_ckpt="distilbert-base-uncased",
        num_labels=2,
        batch_size=64,
        num_epochs=2,
        # data_path='D:\Codes\sentiment-fastapi/airline_sentiment_analysis.csv',
        save_path="finetuned-emotion-model",
    ):

        self.model_ckpt = model_ckpt
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        # self.data_path=data_path
        self.data_labels = ["positive", "negative"]
        self.save_path = save_path
        self.num_epochs = num_epochs

    def load_data(self, path):
        """
        loads csv data and makes sure column names are as needed

        Attributes:
        ----------
        path: str
        path to the dataset

        """
        data = (pd.read_csv(path, index_col=0, header=[0])).reset_index(drop=True)
        data.columns = ["label", "text"]
        data = data[["text", "label"]]
        return data

    def preprocess_text(self, text):
        """
        cleans the input text by removing unnecessary characters and stems each word
        Attributes:
        -----------
        text: str
        input text that contains the tweet


        """
        stemmer = PorterStemmer()
        entity_prefixes = ["@"]
        words = []
        for word in text.split():
            word = word.strip()
            if word:
                if word[0] not in entity_prefixes:
                    word = stemmer.stem(word)
                    words.append(word)
        sentence = " ".join(words)

        # remove stock market tickers
        tweet = re.sub(r"\$\w*", "", sentence)
        # remove twitter abbreviations
        tweet = re.sub(r"^RT[\s]+", "", tweet)
        # remove hyperlinks
        tweet = re.sub(r"https?:\/\/.*[\r\n]*", "", tweet)
        # only removing the hash # sign from the word
        tweet = re.sub(r"#", "", tweet)
        return tweet

    def split_data(self, data):
        """
        splits data into train validate and test
        Attributes:
        ----------
        data: pandas dataframe obj
        entire dataset of tweets and their labels

        Returns:
        -------
        train: numpy array
        test: numpy array
        validate: numpy array
        """
        train, validate, test = np.split(
            data.sample(frac=1), [int(0.6 * len(data)), int(0.8 * len(data))]
        )
        return train, validate, test

    def create_dateset(self, train, validate, test):
        """
        converts numpy arrays into DatasetDict fornmat

        Attributes:
        ----------
        train: numpy array
        test: numpy array
        validate: numpy array

        Returns:
        -------
        my_dataset_dict: DatasetDict
        contains train test and validation set in a DataSetDict format
        """
        train_dataset = datasets.Dataset.from_dict(train)
        test_dataset = datasets.Dataset.from_dict(test)
        validation_dataset = datasets.Dataset.from_dict(validate)
        my_dataset_dict = datasets.DatasetDict(
            {
                "train": train_dataset,
                "validation": validation_dataset,
                "test": test_dataset,
            }
        )
        return my_dataset_dict

    def tokenize(self, batch):
        """
        Batch tokenizes the entire dataset
        Attributes:
        ----------
        batch: List
        the list containing all the tweets
        Returns:
        List: tokenized list
        """
        return self.tokenizer(batch["text"], padding=True, truncation=True)

    def compute_metrics(self, pred):
        """ "
        function for computing f1 and accuracy scores

        Attributes:
        ----------
        pred: List
        contains the prediciton after each batch

        Returns:
        -------
        Dict:
        containing accuracy and f1 scores

        """

        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    def training(
        self, load_path="D:\Codes\sentiment-fastapi/airline_sentiment_analysis.csv"
    ):
        """
        main training function. trains, saves and returns the model
        Attributes:
        ----------
        path: str
        path to the dataset to be loaded

        Returns:
        -------
        Model: Pytorch model
        Fine tuned pytorch model

        """

        data = self.load_data(path=load_path)
        le = LabelEncoder()
        data.label = le.fit(data.label).transform(data.label)
        data.text = [self.preprocess_text(data.text[i]) for i in range(len(data))]
        train, validate, test = self.split_data(data=data)
        sentiment = self.create_dateset(train, validate, test)

        # tokenize and encode
        sentiment_encoded = sentiment.map(self.tokenize, batched=True, batch_size=None)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_ckpt, num_labels=self.num_labels
        ).to(device)

        logging_steps = len(sentiment_encoded["train"]) // self.batch_size
        model_name = f"finetuned-emotion-model"
        training_args = TrainingArguments(
            output_dir=model_name,
            num_train_epochs=self.num_epochs,
            learning_rate=2e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            disable_tqdm=False,
            logging_steps=logging_steps,
            push_to_hub=True,
            log_level="error",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=sentiment_encoded["train"],
            eval_dataset=sentiment_encoded["validation"],
            tokenizer=self.tokenizer,
        )

        # trainer
        trainer.train()
        trainer.save_model(self.save_path)
        return model
