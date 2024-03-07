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
from training_pipeline import train
import warnings
warnings.filterwarnings("ignore")

# Disable from connecting to Weights and Biases
os.environ["WANDB_DISABLED"] = "true"

class inference(train):
  def __init__(self,config_path):
        super().__init__(config_path)
        self.data_path=self.config_file["test_data"]

  def get_model(self):
    try:
      self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
      self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
      self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
      print("Model Loaded")
    except Exception as e:
      print("Model Load Failed")
      print(e)


  def evaluate(self):
    tokenized_tester =self.tokenized_test.remove_columns('labels')
    # Instantiate Trainer
    trainer = Trainer(model=self.model)

    # Make predictions
    predictions = trainer.predict(tokenized_tester)

    preds_flat = [np.argmax(x) for x in predictions[0]]
    print(len(preds_flat))
    cm = confusion_matrix(self.data['labels'], preds_flat)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    precision, recall, fscore, support = score(self.data['labels'], preds_flat)

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))


  def main(self):
    train.get_data(self)
    self.get_model()
    train.data_prep(self)
    self.tokenized_test = train.df_to_ds(self,self.data)

    self.evaluate()

c=inference("/content/config/train_config.json")
c.main()
