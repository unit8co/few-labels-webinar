import random
import datetime
from tqdm import tqdm
from fire import Fire
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils import tensorboard
from src.classifier import train_mlm, train_classifier
from src.agreement import train_agreement, get_label_candidates


def main(
    train_mlm=False,
    dataset_name="nlu_evaluation_data",
    base_model_name="bert-base-uncased",
    max_iterations=8,
    mlm_epochs=1,  # 10,
    classifier_epochs=1,  # 30,
    agreement_epochs=1,
    max_length=64,
    pair_multiplier=1,  # 5,
    val_ratio=0.2,
    label_ratio=0.02,
    seed=42,
):

    random.seed(seed)

    # Loading dataset
    dataset = load_dataset(dataset_name, split="train")

    # We add the item index as it is useful later
    dataset = dataset.map(
        lambda examples, idx: {"item_number": idx}, with_indices=True
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Creating the splits
    train_val_datasets = dataset.train_test_split(
        test_size=val_ratio, shuffle=True, seed=seed
    )
    trainl_trainu_datasets = train_val_datasets["train"].train_test_split(
        test_size=1.0 - label_ratio, shuffle=True, seed=seed
    )
    trainl_dataset = trainl_trainu_datasets["train"]
    trainu_dataset = trainl_trainu_datasets["test"]
    val_dataset = train_val_datasets["test"]

    # Train MLM model if needed
    if train_mlm:
        print("\n#####\nTraining MLM\n#####\n")
        train_mlm(base_model_name, trainu_dataset, max_length)
        print(
            "MLM model trained. "
            "Please rerun the script using train_mlm=False and "
            "setting the base_model_name to the desired MLM model path."
        )
        exit(0)

    # Iterate: train classifier, train agreement model, extended labeled set
    iteration_writer = tensorboard.SummaryWriter(
        f'tensorboard/{dataset_name}/{datetime.datetime.now().strftime("%Y%m%d%H%M")}'
    )
    iteration_writer.add_text("dataset_name", dataset_name)
    iteration_writer.add_text("base_model_name", base_model_name)
    iteration_writer.add_text("max_length", str(max_length))
    iteration_writer.add_text("classifier_epochs", str(classifier_epochs))
    iteration_writer.add_text("agreement_epochs", str(agreement_epochs))
    iteration_writer.add_text("pair_multiplier", str(pair_multiplier))

    for iteration in range(max_iterations):

        # Log data stats
        iteration_writer.add_scalar(
            "trainl_count", len(trainl_dataset), iteration
        )
        iteration_writer.add_scalar(
            "trainu_count", len(trainu_dataset), iteration
        )

        # Training the classifier
        print("\n#####\nTraining classifier\n#####\n")
        classifier_model = train_classifier(
            dataset_name,
            trainl_dataset,
            val_dataset,
            base_model_name,
            max_length,
            classifier_epochs,
            tokenizer,
            iteration_writer,
            iteration,
        )

        # Train agreement model
        print("\n#####\nTraining agreement model\n#####\n")
        agreement_model = train_agreement(
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
            seed,
        )

        # Get new candidates to add to label set
        print(
            "\n#####\nCollecting new candidates to add to labeled set\n#####\n"
        )
        item_number_to_label = get_label_candidates(
            agreement_model,
            trainl_dataset,
            trainu_dataset,
            max_length,
            tokenizer,
            iteration_writer,
            iteration,
        )

        # Add new items to trainl
        to_add_to_trainl_dataset = trainu_dataset.filter(
            lambda example: example["item_number"]
            in item_number_to_label.keys()
        )
        for datapoint in tqdm(to_add_to_trainl_dataset):
            datapoint_with_estimated_label = datapoint
            datapoint_with_estimated_label["label"] = item_number_to_label[
                datapoint_with_estimated_label["item_number"]
            ]
            trainl_dataset = trainl_dataset.add_item(
                datapoint_with_estimated_label
            )

        # Remove these items now from trainu
        trainu_dataset = trainu_dataset.filter(
            lambda example: example["item_number"]
            not in item_number_to_label.keys()
        )


if __name__ == "__main__":
    Fire(main)
