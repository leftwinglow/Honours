import pickle
from typing import Literal

import torch
import transformers as tr
from transformers import GraphormerForGraphClassification, GraphormerConfig, TrainingArguments, Trainer
from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator

from Data_Cleaning import to_graphormer_format
from datasets import load_from_disk


def graphormer_df_creator(format: Literal["graphormer_format", "hf_format", "test_dataset"]):
    import platform

    if platform.system() == "Linux":
        linux = True

    if format == "graphormer_format":
        if linux is True:
            dataset_processed = load_from_disk("/mnt/c/users/luke/documents/university/5th year/honours python/transformed_data/Hansen_Graphormer_DF")
        if linux is False:
            dataset_processed = load_from_disk("C:/Users/Luke/Documents/University/5th Year/Honours Python/Transformed_Data/Hansen_Graphormer_DF")

    if format == "hf_format":
        if linux is True:
            dataset_processed = to_graphormer_format("/mnt/c/users/luke/documents/university/5th year/honours python/transformed_data/Hansen_HF_Graph.pkl")
        if linux is False:
            dataset_processed = to_graphormer_format("C:/Users/Luke/Documents/University/5th Year/Honours Python/Transformed_Data/Hansen_HF_Graph.pkl")

    if format == "test_dataset":
        from datasets import load_dataset

        dataset_processed = load_dataset("OGB/ogbg-molhiv")
        dataset_processed = dataset_processed.map(preprocess_item, batched=False)

    if format != "test_dataset":
        # dataset_processed = dataset_processed.class_encode_column("y")
        dataset_processed = dataset_processed.train_test_split(test_size=0.2, seed=42)

    return dataset_processed


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dataset_processed = graphormer_df_creator("graphormer_format")

print(dataset_processed["train"])
test = tuple(dataset_processed["train"]["edge_index"])
# print(max(test))

graphormer_config = GraphormerConfig(num_classes=2, num_atoms=99999, num_edges=9999)
graphormer = GraphormerForGraphClassification(graphormer_config)

training_args = TrainingArguments(
    "graph-classification",
    logging_dir="graph-classification",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    auto_find_batch_size=True,  # batch size can be changed automatically to prevent OOMs
    gradient_accumulation_steps=10,
    dataloader_num_workers=4,  # 1,
    num_train_epochs=20,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    push_to_hub=False,
    # use_cpu=True,
    # disable_tqdm=True,
)

trainer = Trainer(
    model=graphormer,
    args=training_args,
    data_collator=GraphormerDataCollator(),
    train_dataset=dataset_processed["train"],
    eval_dataset=dataset_processed["test"],
)

train_results = trainer.train()

print(train_results)

# graphormer_config = tr.GraphormerConfig(num_classes=2)

# graphormer = tr.GraphormerForGraphClassification(config=graphormer_config)
