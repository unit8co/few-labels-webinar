import torch
import random
import elasticsearch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import concatenate_datasets
from sklearn.metrics import f1_score


def train_agreement(
    dataset_name,
    trainl_dataset,
    val_dataset,
    base_model_name,
    max_length,
    agreement_epochs,
    pair_multiplier,
    tokenizer,
    iteration_writer,
    iteration,
    experiment_name,
    seed,
):

    # Create paired dataset
    paired_train_val_datasets = create_paired_datasets(
        trainl_dataset, pair_multiplier, tokenizer, max_length, seed
    )

    # Train agreement model
    agreement_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name
    )
    agreement_training_args = TrainingArguments(
        output_dir=f"./models/{dataset_name}/{experiment_name}/agreement/{iteration}",
        logging_dir=f"./models/{dataset_name}/{experiment_name}/agreement/{iteration}",
        overwrite_output_dir=True,
        num_train_epochs=agreement_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        # save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        evaluation_strategy="epoch",
        eval_steps="epoch",
    )
    agreement_trainer = Trainer(
        model=agreement_model,
        args=agreement_training_args,
        train_dataset=paired_train_val_datasets["train"],
        eval_dataset=paired_train_val_datasets["test"],
        compute_metrics=lambda p: {
            "accuracy": np.mean(np.argmax(p.predictions, axis=1) == p.label_ids)
        },
    )
    agreement_train_result = agreement_trainer.train()
    agreement_eval_result = agreement_trainer.evaluate()
    iteration_writer.add_scalar(
        "agreement/train/loss", agreement_train_result.training_loss, iteration
    )
    iteration_writer.add_scalar(
        "agreement/eval/loss", agreement_eval_result["eval_loss"], iteration
    )
    iteration_writer.add_scalar(
        "agreement/eval/accuracy",
        agreement_eval_result["eval_accuracy"],
        iteration,
    )
    return agreement_model


def get_label_candidates(
    agreement_model,
    trainl_dataset,
    trainu_dataset,
    max_length,
    tokenizer,
    iteration_writer,
    iteration,
):

    # Use elasticsearch to index for fast retrieval of neighbours
    es_client = elasticsearch.Elasticsearch()
    es_client.indices.delete(index="trainu", ignore=[400, 404])
    trainu_dataset.add_elasticsearch_index(
        column="text", es_client=es_client, es_index_name="trainu"
    )

    device = "cuda"
    k = 10  # Number of neighbours in elastic
    best_per_class = 2  # Number of candidates per class at the end

    # Use elasticsearch on trainu to fish for new candidates.
    # Then pair these candidates with trainl to see if they have the same label.
    elastic_bad_match_count = 0
    candidates = defaultdict(list)
    for datapoint_l in tqdm(trainl_dataset):
        text_l = datapoint_l["text"]
        scores, retrieved_examples = trainu_dataset.get_nearest_examples(
            index_name="text", query=text_l, k=k
        )
        retrieved_count = len(retrieved_examples["text"])
        if retrieved_count == 0:
            continue
        tokenized_pairs = tokenizer(
            [text_l] * retrieved_count,
            retrieved_examples["text"],
            truncation=True,
            max_length=2 * max_length,
            padding="max_length",
        )
        batch = {
            k: torch.tensor(v).to(device) for k, v in tokenized_pairs.items()
        }
        output = agreement_model(**batch)
        for i in range(retrieved_count):
            if output["logits"][i][1] > output["logits"][i][0]:  # positive pair
                item_number = retrieved_examples["item_number"][i]
                confidence = (
                    torch.nn.Softmax()(output["logits"][i])
                    .detach()
                    .cpu()
                    .numpy()[1]
                    .item()
                )
                true_label = retrieved_examples["label"][i]
                candidates[datapoint_l["label"]].append(
                    (confidence, item_number, true_label)
                )
            else:  # negative pair
                elastic_bad_match_count += 1

    # List top candidates
    # We also calculate the real accuracy on the new labels, and this is only possible since we have the "real" label on trainu
    trainu_dataset.drop_index("text")
    y_true = []
    y_pred = []
    item_number_to_label = {}
    for predicted_label, candidates_for_label in tqdm(candidates.items()):
        top_candidates = sorted(candidates_for_label, reverse=True)[
            :best_per_class
        ]
        for candidate in top_candidates:
            true_label = candidate[2]
            item_number = candidate[1]
            y_true.append(true_label)
            y_pred.append(predicted_label)
            item_number_to_label[item_number] = predicted_label

    elastic_accuracy = len(candidates) / (
        elastic_bad_match_count + len(candidates)
    )
    fishing_f1 = f1_score(y_true, y_pred, average="micro")
    iteration_writer.add_scalar(
        "agreement/fishing/elastic_accuracy", elastic_accuracy, iteration
    )
    iteration_writer.add_scalar(
        "agreement/fishing/top_candidates_f1", fishing_f1, iteration
    )

    return item_number_to_label


def create_paired_datasets(
    trainl_dataset, pair_multiplier, tokenizer, max_length, seed
):
    paired_datasets = []
    for _ in range(pair_multiplier):
        paired_dataset_positive = trainl_dataset.map(
            lambda examples: pair_up(examples, match="positive"),
            batched=True,
            load_from_cache_file=False,
        )
        paired_dataset_negative = trainl_dataset.map(
            lambda examples: pair_up(examples, match="negative"),
            batched=True,
            load_from_cache_file=False,
        )
        paired_datasets.append(paired_dataset_positive)
        paired_datasets.append(paired_dataset_negative)

    paired_dataset = concatenate_datasets(paired_datasets)

    paired_dataset = paired_dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            examples["text_other"],
            truncation=True,
            max_length=2 * max_length,
            padding="max_length",
        ),
        batched=True,
        load_from_cache_file=False,
    )

    paired_dataset = paired_dataset.map(
        lambda examples: {"labels": examples["match"]},
        batched=True,
        load_from_cache_file=False,
    )

    paired_train_val_datasets = paired_dataset.train_test_split(
        test_size=0.2, shuffle=True, seed=seed
    )

    paired_train_val_datasets.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
    )
    return paired_train_val_datasets


def pair_up(batch, match):
    """
    This function creates positive or negative pair for a given labeled batch
    """
    result = {
        "text": [],
        "text_other": [],
        "match": [],
    }
    label_to_indices = defaultdict(list)
    for index, label in enumerate(batch["label"]):
        label_to_indices[label].append(index)
    for label, text in zip(batch["label"], batch["text"]):
        if match == "positive":
            random_positive_index = random.choice(label_to_indices[label])
            text_to_append = batch["text"][random_positive_index]
            match_to_append = 1
        elif match == "negative":
            labels_wihtout_this = [
                l
                for l in label_to_indices.keys()
                if label_to_indices[l] and l != label
            ]
            # labels_wihtout_this.remove(label)
            random_negative_label = random.choice(labels_wihtout_this)
            random_negative_index = random.choice(
                label_to_indices[random_negative_label]
            )
            text_to_append = batch["text"][random_negative_index]
            match_to_append = 0
        result["match"].append(match_to_append)
        if random.choice([0, 1]):
            # swap the texts
            result["text"].append(text_to_append)
            result["text_other"].append(text)
        else:
            result["text"].append(text)
            result["text_other"].append(text_to_append)
    return result
