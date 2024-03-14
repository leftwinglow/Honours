import evaluate
import numpy as np


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]
    else:
        raise TypeError("Logits are not in the expected tuple format.")

    predictions = np.argmax(logits, axis=-1)

    metrics = {"accuracy": evaluate.load("accuracy"), "precision": evaluate.load("precision"), "recall": evaluate.load("recall"), "f1": evaluate.load("f1")}

    results = {}

    for metric_name, metric in metrics.items():
        results[metric_name] = metric.compute(predictions=predictions, references=labels)

    return results


# import evaluate
# import numpy as np


# def compute_metrics1(eval_pred):
#     metric = evaluate.load("accuracy")
#     logits, labels = eval_pred
#     if isinstance(logits, tuple):
#         logits = logits[0]
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)


# from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

# def compute_metrics2(eval_pred):
#     logits, labels = eval_pred
#     if isinstance(logits, tuple):
#         logits = logits[0]
#     preds = np.argmax(logits, axis=-1)

#     # Compute your metrics here
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
#     acc = balanced_accuracy_score(labels, preds)

#     return {
#         'accuracy': acc,
#         'f1': f1,
#         'precision': precision,
#         'recall': recall
#     }
