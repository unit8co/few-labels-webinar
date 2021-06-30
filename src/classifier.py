import torch
import fasttext
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def process_dataset(dataset, tokenizer, max_length):

    # Tokenize the text
    processed_dataset = dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        ),
        batched=True,
        load_from_cache_file=False,
    )

    # One-hot encoding of the label
    processed_dataset = processed_dataset.map(
        lambda examples: {
            "labels": [
                1.0 if i == examples["label"] else 0.0
                for i in range(dataset.features["label"].num_classes)
            ]
        },
        batched=False,
        load_from_cache_file=False,
    )

    # Set format for dataloader
    processed_dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
    )

    return processed_dataset


def train_mlm(
    dataset_name,
    trainu_dataset,
    val_dataset,
    base_model_name,
    max_length,
    mlm_epochs,
    tokenizer,
):

    processed_trainu_dataset = process_dataset(trainu_dataset, tokenizer, max_length)
    processed_val_dataset = process_dataset(val_dataset, tokenizer, max_length)

    mlm_model = AutoModelForMaskedLM.from_pretrained(base_model_name)
    mlm_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    mlm_training_args = TrainingArguments(
        output_dir=f"./models/{dataset_name}/mlm",
        logging_dir=f"./models/{dataset_name}/mlm",
        overwrite_output_dir=True,
        num_train_epochs=mlm_epochs,
        per_device_train_batch_size=8,
        save_strategy="epoch",
        save_total_limit=2,
        evaluation_strategy="epoch",
    )

    mlm_trainer = Trainer(
        model=mlm_model,
        args=mlm_training_args,
        data_collator=mlm_data_collator,
        train_dataset=processed_trainu_dataset,
        eval_dataset=processed_val_dataset,
    )

    mlm_trainer.train()

    return mlm_model


def train_classifier(
    dataset_name,
    trainl_dataset,
    val_dataset,
    base_model_name,
    max_length,
    classifier_epochs,
    tokenizer,
    iteration_writer,
    iteration,
    experiment_name,
    init_weights,
):

    processed_trainl_dataset = process_dataset(trainl_dataset, tokenizer, max_length)
    processed_val_dataset = process_dataset(val_dataset, tokenizer, max_length)

    classifier_config = AutoConfig.from_pretrained(
        base_model_name, num_labels=trainl_dataset.features["label"].num_classes
    )
    classifier_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, config=classifier_config
    )

    if init_weights:
        print("Initializing weights...")

        def init_weights(m):
            if type(m) == torch.nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        classifier_model.apply(init_weights)

    classifier_training_args = TrainingArguments(
        output_dir=f"./models/{dataset_name}/{experiment_name}/classifier/{iteration}",
        logging_dir=f"./models/{dataset_name}/{experiment_name}/classifier/{iteration}",
        overwrite_output_dir=True,
        num_train_epochs=classifier_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=64,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=500,
    )

    classifier_trainer = Trainer(
        model=classifier_model,
        args=classifier_training_args,
        train_dataset=processed_trainl_dataset,
        eval_dataset=processed_val_dataset,
        compute_metrics=lambda p: {
            "accuracy": np.mean(
                np.argmax(p.predictions, axis=1) == np.argmax(p.label_ids, axis=1)
            )
        },
    )

    classifier_train_result = classifier_trainer.train()
    classifier_eval_result = classifier_trainer.evaluate()

    iteration_writer.add_scalar(
        "classifier/train/loss",
        classifier_train_result.training_loss,
        iteration,
    )
    iteration_writer.add_scalar(
        "classifier/eval/loss", classifier_eval_result["eval_loss"], iteration
    )
    iteration_writer.add_scalar(
        "classifier/eval/accuracy",
        classifier_eval_result["eval_accuracy"],
        iteration,
    )

    return classifier_model


def train_classifier_fasttext(
    trainl_dataset, val_dataset, iteration_writer, iteration, epochs=5000
):
    with open("trainl.fasttext.txt", "w") as fp:
        for item in trainl_dataset:
            fp.write(f'__label__{item["label"]}\t{item["text"]}\n')
    with open("val.fasttext.txt", "w") as fp:
        for item in val_dataset:
            fp.write(f'__label__{item["label"]}\t{item["text"]}\n')

    model = fasttext.train_supervised("trainl.fasttext.txt", epoch=epochs)
    accuracy = model.test("val.fasttext.txt")[2]
    iteration_writer.add_scalar(
        "classifier/eval/accuracy",
        accuracy,
        iteration,
    )
