import torch
from transformers import GraphormerForGraphClassification, GraphormerConfig, TrainingArguments, Trainer
from transformers.models.graphormer.collating_graphormer import GraphormerDataCollator
from Data_Handling import graphormer_df_creator
from Metric_Utilities import compute_metrics

# from Metric_Utilities import compute_metrics

dataset_processed = graphormer_df_creator("graphormer_format", truncated=False)

print(dataset_processed.num_rows)

graphormer_config = GraphormerConfig(num_classes=2, num_atoms=50000, num_edges=9999)
graphormer = GraphormerForGraphClassification(graphormer_config)

training_args = TrainingArguments(
    "graph-classification",
    logging_dir="graph-classification",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
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
    compute_metrics=compute_metrics,
)

train_results = trainer.train()

print(train_results)
