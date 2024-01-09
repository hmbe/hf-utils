import boto3
import os
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from functools import partial

import evaluate
import torch
import numpy as np


def init_s3_resource(aws_access_key_id, aws_secret_access_key, region_name='ap-northeast-2'):
    s3_resource = boto3.resource(
        's3',
        aws_access_key_id = aws_access_key_id,
        aws_secret_access_key = aws_secret_access_key,
        region_name = region_name,
        )
    
    return s3_resource

def init_s3_client(aws_access_key_id, aws_secret_access_key, region_name='ap-northeast-2'):
    ### print object list
    s3_client = boto3.client(
        's3',
        aws_access_key_id = aws_access_key_id,
        aws_secret_access_key = aws_secret_access_key,
        region_name = region_name,
        )
    
    return s3_client
    

def download_s3_folder(s3_resource, bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """

    bucket = s3_resource.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)

# download_s3_folder(s3_resource, bucket_name, s3_dir_name)

def prepare_examples_layoutlm(batch, processor):
    # images = [img for img in batch['image']]
    images = batch['image']
    words = batch['tokens']
    boxes = batch['bboxes']
    word_labels = batch['ner_tags']

    encoding = processor(images, words, boxes=boxes, word_labels=word_labels, truncation=True, stride =128,
        padding="max_length", max_length=512, return_overflowing_tokens=True, return_offsets_mapping=True, return_token_type_ids=True)

    offset_mapping = encoding.pop('offset_mapping')
    overflow_to_sample_mapping = encoding.pop('overflow_to_sample_mapping')

    return encoding\

def prepare_examples_lilt(batch, tokenizer):
    encoding = tokenizer(batch["tokens"],
                        boxes=batch["bboxes"],
                        word_labels=batch["ner_tags"],
                        padding="max_length",
                        max_length=512,
                        truncation=True,
                        return_tensors="pt",
                        stride=128,
                        return_overflowing_tokens=True,
                        return_offsets_mapping=True
                        )
    
    offset_mapping = encoding.pop('offset_mapping')
    overflow_to_sample_mapping = encoding.pop('overflow_to_sample_mapping')
    for key in encoding.keys():
        ### is it correct?
        encoding[key] = encoding[key].numpy()
        print(key, encoding[key].shape)

    return encoding

def preprocess_dataset_lilt(dataset, tokenizer):
    features = Features({
        # 'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'labels': Sequence(feature=Value(dtype='int64')),
    })
    
    prepare_examples = partial(prepare_examples_lilt, tokenizer=tokenizer)

    column_names = dataset["train"].column_names
    train_dataset = dataset["train"].map(
        prepare_examples,
        batched=True,
        remove_columns=column_names,
        features=features,
    )

    eval_dataset = dataset["test"].map(
        prepare_examples,
        batched=True,
        remove_columns=column_names,
        features=features,
    )

    return train_dataset, eval_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Metric
metric = evaluate.load("seqeval")
return_entity_level_metrics = False

### use partial for huggingface trainer
def compute_metrics(p, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }