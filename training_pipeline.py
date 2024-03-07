# General Libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import re
import torch
import os
import json

# Importing the required Transformer models
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, logging

# The training-validation split
from sklearn.model_selection import train_test_split

# The transformers library doesn't work particularly well with pandas dataframes
from datasets import Dataset

# Metrics and Visualizers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support as score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Disable from connecting to Weights and Biases
os.environ["WANDB_DISABLED"] = "true"

class train:
  def __init__(self,config_path):
        self.config_path=config_path
        self.config_file=self.get_config_file(config_path)
        self.data_path=self.config_file["data_path"]
        self.model_name = self.config_file["model_name"]
        self.model_path = self.config_file["model_path"]
        
    
  def get_config_file(self,config_path):
        
        with open(config_path, 'r') as openfile:
                config_file = json.load(openfile)
        
        return config_file

  def get_data(self):
    try:
      data = pd.read_csv(self.data_path, header=0, names=['labels', 'title', 'description'])
      self.data = data
      print("data Loaded")
    except Exception as e:
      print("data Load Failed")
      print(e)

  def data_prep(self):
    #Although we're being provided with two columns worth of text data: Title, and Description we obtain most of the information from the title itself
    self.data['text'] = self.data['title'] + " " + self.data['description']
    self.data = self.data.drop(['title', 'description'], axis=1)

    #Our dataset has class labels in the range 1-4
    #But, the loss calculation that we will be doing will be through Cross Entropy
    #If there are 'c' classes, cross entropy expects labels from 0 to c-1
    self.data['labels'] = self.data['labels'] - 1

    # Defining a Regular Expression to extract only the words and spaces
    self.data['text'] = self.data['text'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))

    # Defining a Regular Expression to convert all words to lowercase
    self.data['text'] = self.data['text'].apply(lambda x: x.lower())

  def tokenize(self,dataset):
    return self.tokenizer(dataset['text'], truncation=True)

  # This function will convert the dataframe to a dataset and then, perform tokenization
  # on the dataset using the above helper function
  def df_to_ds(self,dataframe):
      dataset = Dataset.from_pandas(dataframe, preserve_index=False)
      tokenized_ds = dataset.map(self.tokenize, batched=True)
      tokenized_ds = tokenized_ds.remove_columns('text')

      return tokenized_ds

  def fine_tune_transformer(self):
    #Loading in the HuggingFace Transformer Model
    transformer = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=4)
    # Slightly modified training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        save_strategy = 'epoch',
        optim="adamw_torch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        fp16=True,
        num_train_epochs=5,
        weight_decay=0.01,
    )


    trainer = Trainer (
        model=transformer,
        args=training_args,
        train_dataset=self.tokenized_train,
        eval_dataset=self.tokenized_val,
        tokenizer=self.tokenizer,
        data_collator=self.data_collator,
    )

    trainer.train()
    return trainer
    
  def main(self):
    self.get_data()
    self.data_prep()
    #
    self.tokenizer=AutoTokenizer.from_pretrained(self.model_name)
    #The Data Collator we are using will also be adding padding to the training batches
    self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
    self.df_train_in, self.df_val_in = train_test_split(self.data[['labels', 'text']], test_size=0.2, random_state=42)

    self.tokenized_train = self.df_to_ds(self.df_train_in)
    self.tokenized_val = self.df_to_ds(self.df_val_in)
    trainer=self.fine_tune_transformer()
    trainer.save_model(self.model_path)
c=train("/content/config/train_config.json")
c.main()
