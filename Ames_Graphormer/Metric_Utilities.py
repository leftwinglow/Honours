import evaluate

evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(logits, axis = -1)
    return metric.compute(predictions = predictions, references = labels)