
import random
import torch
import pathlib
import datetime
import elasticsearch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datasets import list_datasets, load_dataset, concatenate_datasets
from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoTokenizer, AutoConfig, AutoModel
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, EvalPrediction
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import f1_score, classification_report
from torch.utils import tensorboard


# Setup
dataset_name = 'nlu_evaluation_data'
base_model_name = 'bert-base-uncased'
max_iterations = 10
classifier_epochs = 10
agreement_epochs = 1
writer = tensorboard.SummaryWriter(f'tensorboard/{dataset_name}/{datetime.datetime.now().strftime("%Y%m%d%H%M")}')
writer.add_text('dataset_name', dataset_name)
writer.add_text('base_model_name', base_model_name)


def pair_up(batch, match):
    """
    This function creates positive or negative pair for a given labeled batch
    """
    result = {
        'text': [],
        'text_other': [],
        'match': [],
    }
    label_to_indices = defaultdict(list)
    for index, label in enumerate(batch['label']):
        label_to_indices[label].append(index)
    for label, text in zip(batch['label'], batch['text']):
        if match == 'positive':
            random_positive_index = random.choice(label_to_indices[label])
            text_to_append = batch['text'][random_positive_index]
            match_to_append = 1
        elif match == 'negative':
            labels_wihtout_this = [l for l in label_to_indices.keys() if label_to_indices[l] and l != label]
            # labels_wihtout_this.remove(label)
            random_negative_label = random.choice(labels_wihtout_this)
            random_negative_index = random.choice(label_to_indices[random_negative_label])
            text_to_append = batch['text'][random_negative_index]
            match_to_append = 0
        result['match'].append(match_to_append)
        if random.choice([0, 1]):
            # swap the texts 
            result['text'].append(text_to_append)
            result['text_other'].append(text)
        else:
            result['text'].append(text)
            result['text_other'].append(text_to_append)
    return result


# Loading dataset
dataset = load_dataset('nlu_evaluation_data', split='train')

# We add the item index as it is useful later
dataset = dataset.map(lambda examples, idx: {'item_number': idx}, with_indices=True)
print(dataset)
print(dataset.features)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Creating the splits
train_val_datasets = dataset.train_test_split(test_size=5715, shuffle=True, seed=42)
trainl_trainu_datasets = train_val_datasets['train'].train_test_split(test_size=19000, shuffle=True, seed=42)
trainl_dataset = trainl_trainu_datasets['train']
trainu_dataset = trainl_trainu_datasets['test']
val_dataset = train_val_datasets['test']

for iteration in range(max_iterations):

    # Add stats to tensorboard
    writer.add_scalar('trainl_count', len(trainl_dataset), iteration)
    writer.add_scalar('trainu_count', len(trainu_dataset), iteration)

    # Tokenization, one-hot encoding, and formatting the get_item behavior
    processed_trainl_dataset = trainl_dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, max_length=256, padding='max_length'), batched=True)
    processed_trainl_dataset = processed_trainl_dataset.map(lambda examples: {'labels': [1.0 if i == examples['label'] else 0.0 for i in range(dataset.features['label'].num_classes)]}, batched=False)
    processed_trainl_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    processed_trainu_dataset = trainu_dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, max_length=256, padding='max_length'), batched=True)
    processed_trainu_dataset = processed_trainu_dataset.map(lambda examples: {'labels': [1.0 if i == examples['label'] else 0.0 for i in range(dataset.features['label'].num_classes)]}, batched=False)
    processed_trainu_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    processed_val_dataset = val_dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, max_length=256, padding='max_length'), batched=True)
    processed_val_dataset = processed_val_dataset.map(lambda examples: {'labels': [1.0 if i == examples['label'] else 0.0 for i in range(dataset.features['label'].num_classes)]}, batched=False)
    processed_val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    # Training the classifier
    print('\n#####\nTraining classifier...\n#####\n')
    classifier_config = AutoConfig.from_pretrained('bert-base-uncased', num_labels=dataset.features['label'].num_classes)
    classifier_model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', config=classifier_config)
    classifier_training_args = TrainingArguments(
        output_dir='./models/nlu_evaluation_data/classifier',
        overwrite_output_dir=True,
        num_train_epochs=classifier_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=64,
        save_strategy='epoch',
        save_total_limit=2,
        evaluation_strategy='epoch'
    )
    classifier_trainer = Trainer(
        model=classifier_model,
        args=classifier_training_args,
        train_dataset=processed_trainl_dataset,
        eval_dataset=processed_val_dataset,
        compute_metrics=lambda p: {'accuracy': np.mean(np.argmax(p.predictions, axis=1) == np.argmax(p.label_ids, axis=1))},
    )
    classifier_train_result = classifier_trainer.train()
    classifier_eval_result = classifier_trainer.evaluate()
    writer.add_scalar('classifier/train/loss', classifier_train_result.training_loss, iteration)
    writer.add_scalar('classifier/eval/loss', classifier_eval_result['eval_loss'], iteration)
    writer.add_scalar('classifier/eval/accuracy', classifier_eval_result['eval_accuracy'], iteration)
    del classifier_model
    del classifier_trainer
    del classifier_training_args
    del processed_trainl_dataset
    del processed_trainu_dataset
    del processed_val_dataset

    # Create paired dataset
    paired_dataset_positive = trainl_dataset.map(lambda examples: pair_up(examples, match='positive'), batched=True)
    paired_dataset_negative = trainl_dataset.map(lambda examples: pair_up(examples, match='negative'), batched=True)
    paired_dataset = concatenate_datasets([paired_dataset_positive, paired_dataset_negative])
    paired_dataset = paired_dataset.map(lambda examples: tokenizer(examples['text'], examples['text_other'], truncation=True, max_length=256, padding='max_length'), batched=True)
    paired_dataset = paired_dataset.map(lambda examples: {'labels': examples['match']}, batched=True)
    paired_train_val_datasets = paired_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    paired_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    # Train agreement model
    print('\n#####\nTraining agreement model...\n#####\n')
    agreement_model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')        
    agreement_training_args = TrainingArguments(
        output_dir='./models/nlu_evaluation_data/agreement',
        overwrite_output_dir=True,
        num_train_epochs=agreement_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        save_strategy='epoch',
        save_total_limit=2,
        evaluation_strategy='epoch'
    )
    agreement_trainer = Trainer(
        model=agreement_model,
        args=agreement_training_args,
        train_dataset=paired_train_val_datasets['train'],
        eval_dataset=paired_train_val_datasets['test'],
        compute_metrics=lambda p: {'accuracy': np.mean(np.argmax(p.predictions, axis=1) == p.label_ids)},
    )
    agreement_train_result = agreement_trainer.train()
    agreement_eval_result = agreement_trainer.evaluate()
    writer.add_scalar('agreement/train/loss', agreement_train_result.training_loss, iteration)
    writer.add_scalar('agreement/eval/loss', agreement_eval_result['eval_loss'], iteration)
    writer.add_scalar('agreement/eval/accuracy', agreement_eval_result['eval_accuracy'], iteration)
    del paired_dataset_negative
    del paired_dataset_positive
    del paired_dataset

    # Get new candidates to add to label set
    print('\n#####\nCollecting new candidates to add to labeled set...\n#####\n')
    es_client = elasticsearch.Elasticsearch()
    es_client.indices.delete(index='trainu', ignore=[400, 404])
    trainu_dataset.add_elasticsearch_index(column='text', es_client=es_client, es_index_name='trainu')
    device = 'cuda'
    # agreement_model = agreement_model.to(device)
    k = 5
    candidates = defaultdict(list)
    for datapoint_l in tqdm(trainl_dataset):
        text_l = datapoint_l['text']
        scores, retrieved_examples = trainu_dataset.get_nearest_examples(index_name='text', query=text_l, k=k)
        retrieved_count = len(retrieved_examples['text'])
        if retrieved_count == 0:
            continue
        tokenized_pairs = tokenizer([text_l] * retrieved_count, retrieved_examples['text'], truncation=True, max_length=256, padding='max_length')
        batch = {k: torch.tensor(v).to(device) for k, v in tokenized_pairs.items()}
        output = agreement_model(**batch)
        for i in range(retrieved_count):
            if output['logits'][i][1] > output['logits'][i][0]:  # positive pair
                item_number = retrieved_examples['item_number'][i]
                confidence = torch.nn.Softmax()(output['logits'][i]).detach().cpu().numpy()[1].item()
                true_label = retrieved_examples['label'][i]
                candidates[datapoint_l['label']].append((confidence, item_number, true_label))
    del batch
    del output
    del agreement_model
    del agreement_trainer
    del agreement_training_args

    # List all new candidates and print the real accuracy on the new labels
    best_per_class = 5
    trainu_dataset.drop_index('text')
    y_true = []
    y_pred = []
    item_number_to_label = {}
    for predicted_label, candidates_for_label in tqdm(candidates.items()):
        top_candidates = sorted(candidates_for_label, reverse=True)[:best_per_class]
        for candidate in top_candidates:
            true_label = candidate[2]
            item_number = candidate[1]
            y_true.append(true_label)
            y_pred.append(predicted_label)
            item_number_to_label[item_number] = predicted_label
    print(classification_report(y_true, y_pred))
    fishing_f1 = f1_score(y_true, y_pred, average='micro')
    writer.add_scalar('agreement/fishing/f1', fishing_f1, iteration)
    del candidates

    # Add new items to trainl
    to_add_to_trainl_dataset = trainu_dataset.filter(lambda example: example['item_number'] in item_number_to_label.keys())
    for datapoint in tqdm(to_add_to_trainl_dataset):
        datapoint_with_estimated_label = datapoint
        datapoint_with_estimated_label['label'] = item_number_to_label[datapoint_with_estimated_label['item_number']]
        trainl_dataset = trainl_dataset.add_item(datapoint_with_estimated_label)

    # Remove these items now from trainu
    trainu_dataset = trainu_dataset.filter(lambda example: example['item_number'] not in item_number_to_label.keys())
