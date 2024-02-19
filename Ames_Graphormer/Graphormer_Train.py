import torch
from transformers import GraphormerForGraphClassification, GraphormerConfig, TrainingArguments, Trainer
from transformers.models.graphormer.collating_graphormer import GraphormerDataCollator
from Data_Handling import graphormer_df_creator


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dataset_processed = graphormer_df_creator("graphormer_format")

graphormer_config = GraphormerConfig(num_classes=2, num_atoms=50000, num_edges=9999)
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
