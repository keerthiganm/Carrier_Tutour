# ============================================================
# TRAIN KNOWLEDGE EXTRACTOR - SMOKE TEST
#
# Goal:
# Fine-tune DistilBERT on SkillSpan tags_knowledge.
#
# Labels:
#     O = 0
#     B = 1
#     I = 2
#
# IMPORTANT:
# This is a small smoke-test training run.
# It is NOT our final trained model.
# ============================================================


import json
from pathlib import Path

import torch

from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 128

TRAIN_SAMPLE_SIZE = 300
DEV_SAMPLE_SIZE = 100

NUM_LABELS = 3


LABEL_TO_ID = {
    "O": 0,
    "B": 1,
    "I": 2
}


ID_TO_LABEL = {
    0: "O",
    1: "B",
    2: "I"
}


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


PROCESSED_DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
)


MODEL_OUTPUT_DIR = (
    PROJECT_DIR
    / "models"
    / "knowledge_extractor_smoke_test"
)


# ============================================================
# LOAD PROCESSED DATA
# ============================================================

def load_json(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


train_records = load_json(
    PROCESSED_DATA_DIR
    / "skillspan_train.json"
)


dev_records = load_json(
    PROCESSED_DATA_DIR
    / "skillspan_dev.json"
)


# Use only a small sample for our first run
train_records = train_records[
    :TRAIN_SAMPLE_SIZE
]


dev_records = dev_records[
    :DEV_SAMPLE_SIZE
]


print(
    "Training examples:",
    len(train_records)
)

print(
    "Validation examples:",
    len(dev_records)
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)


# ============================================================
# LABEL ALIGNMENT
# ============================================================

def tokenize_and_align_labels(
    words,
    word_labels
):
    """
    Tokenize original SkillSpan words and align
    word-level BIO labels with DistilBERT subwords.

    Example:

        Docker -> B

    may become:

        dock -> B
        ##er -> I
    """

    encoding = tokenizer(
        words,

        is_split_into_words=True,

        truncation=True,

        max_length=MAX_LENGTH
    )


    word_ids = encoding.word_ids()


    aligned_labels = []

    previous_word_id = None


    for word_id in word_ids:

        # ----------------------------------------
        # Special tokens
        # ----------------------------------------

        if word_id is None:

            aligned_labels.append(
                -100
            )


        # ----------------------------------------
        # First subword of original word
        # ----------------------------------------

        elif word_id != previous_word_id:

            original_label = (
                word_labels[word_id]
            )

            aligned_labels.append(
                original_label
            )


        # ----------------------------------------
        # Additional subword
        # ----------------------------------------

        else:

            original_label = (
                word_labels[word_id]
            )


            if original_label == 1:

                # B becomes I for continuation
                aligned_labels.append(
                    2
                )

            else:

                aligned_labels.append(
                    original_label
                )


        previous_word_id = word_id


    encoding["labels"] = aligned_labels


    return encoding


# ============================================================
# PYTORCH DATASET
# ============================================================

class SkillSpanKnowledgeDataset(Dataset):

    def __init__(self, records):

        self.records = records


    def __len__(self):

        return len(
            self.records
        )


    def __getitem__(self, index):

        record = self.records[
            index
        ]


        words = record[
            "tokens"
        ]


        # IMPORTANT:
        # Our first model learns ONLY
        # knowledge annotations.
        word_labels = record[
            "knowledge_labels"
        ]


        encoding = (
            tokenize_and_align_labels(
                words,
                word_labels
            )
        )


        return {
            key: torch.tensor(
                value
            )

            for key, value
            in encoding.items()
        }


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = (
    SkillSpanKnowledgeDataset(
        train_records
    )
)


dev_dataset = (
    SkillSpanKnowledgeDataset(
        dev_records
    )
)


# ============================================================
# LOAD DISTILBERT
# ============================================================

model = (
    AutoModelForTokenClassification
    .from_pretrained(

        MODEL_NAME,

        num_labels=NUM_LABELS,

        id2label=ID_TO_LABEL,

        label2id=LABEL_TO_ID
    )
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

training_args = TrainingArguments(

    output_dir=str(
        MODEL_OUTPUT_DIR
    ),

    # Smoke test only
    num_train_epochs=1,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    learning_rate=5e-5,

    weight_decay=0.01,

    logging_steps=10,

    save_strategy="no",

    report_to="none"
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=dev_dataset
)


# ============================================================
# START TRAINING
# ============================================================

print()
print("=" * 70)
print("STARTING SMOKE-TEST TRAINING")
print("=" * 70)


trainer.train()


# ============================================================
# SIMPLE VALIDATION
# ============================================================

print()
print("=" * 70)
print("VALIDATING MODEL")
print("=" * 70)


evaluation_results = (
    trainer.evaluate()
)


print(
    evaluation_results
)


# ============================================================
# SAVE OUR MODEL
# ============================================================

FINAL_MODEL_DIR = (
    PROJECT_DIR
    / "models"
    / "knowledge_extractor_v1_smoke"
)


model.save_pretrained(
    FINAL_MODEL_DIR
)


tokenizer.save_pretrained(
    FINAL_MODEL_DIR
)


print()
print("=" * 70)
print("SMOKE TEST COMPLETE")
print("=" * 70)

print(
    "Model saved to:"
)

print(
    FINAL_MODEL_DIR
)