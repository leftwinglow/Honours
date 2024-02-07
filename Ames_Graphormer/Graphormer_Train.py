import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

import transformers as tr
from transformers import GraphormerForGraphClassification, GraphormerConfig, TrainingArguments, Trainer
from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator

from datasets import Dataset, load_dataset
from Data_Cleaning import to_graphormer_format

dataset_processed = to_graphormer_format("C:/Users/Luke/Documents/University/5th Year/Honours Python/Transformed_Data/Hansen_HF_Graph.pkl")

# dataset_processed = dataset_processed.train_test_split(test_size=0.2, stratify_by_column="y", seed=42)

print(dataset_processed)

graphormer_config = GraphormerConfig(
    num_classes=2
)

graphormer = GraphormerForGraphClassification(config=graphormer_config)

training_args = TrainingArguments(
    "graph-classification",
    logging_dir="graph-classification",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    auto_find_batch_size=True, # batch size can be changed automatically to prevent OOMs
    gradient_accumulation_steps=10,
    dataloader_num_workers=4, #1,
    num_train_epochs=20,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    push_to_hub=False,
)

trainer = Trainer(
    model=graphormer,
    args=training_args,
    data_collator=GraphormerDataCollator(),
    train_dataset=dataset_processed['train'],
    eval_dataset=dataset_processed['test'],
)

# graphormer_config = tr.GraphormerConfig(num_classes=2)

# graphormer = tr.GraphormerForGraphClassification(config=graphormer_config)
