# ============================================================
# PREPARE SKILLSPAN DATA
#
# Purpose:
# Convert raw SkillSpan data into a simpler format that our
# future transformer model can consume.
#
# NOTE:
# We preserve BOTH:
#   - Skill annotations
#   - Knowledge annotations
#
# because the same token can belong to both annotation streams.
# ============================================================


import json
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# prepare_skillspan.py lives inside:
#
# Career_tutor/scripts/
#
# Therefore parent.parent takes us back to:
#
# Career_tutor/
PROJECT_DIR = Path(__file__).resolve().parent.parent


RAW_DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "skillspan"
    / "json"
)


PROCESSED_DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
)


# Create folder if it does not already exist
PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LABEL MAPPING
# ============================================================

# Current SkillSpan labels:
#
# O = Outside a span
# B = Beginning of a span
# I = Inside a span

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
# LOAD JSON-LINES FILE
# ============================================================

def load_json_lines(file_path):

    records = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


# ============================================================
# PROCESS ONE RECORD
# ============================================================

def process_record(record, record_id):

    tokens = record["tokens"]

    skill_tags = record["tags_skill"]

    knowledge_tags = record["tags_knowledge"]


    # --------------------------------------------
    # Validate the source record
    # --------------------------------------------

    if not (
        len(tokens)
        == len(skill_tags)
        == len(knowledge_tags)
    ):

        raise ValueError(
            f"Token/label mismatch in {record_id}"
        )


    # --------------------------------------------
    # Convert labels into numbers
    # --------------------------------------------

    skill_labels = [
        LABEL_TO_ID[tag]
        for tag in skill_tags
    ]


    knowledge_labels = [
        LABEL_TO_ID[tag]
        for tag in knowledge_tags
    ]


    # --------------------------------------------
    # Return our simplified structure
    # --------------------------------------------

    return {

        "id": record_id,

        "tokens": tokens,

        "skill_labels": skill_labels,

        "knowledge_labels": knowledge_labels,

        "source": record.get(
            "source",
            "unknown"
        )
    }


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split_name):

    input_file = (
        RAW_DATA_DIR
        / f"{split_name}.json"
    )

    output_file = (
        PROCESSED_DATA_DIR
        / f"skillspan_{split_name}.json"
    )


    print(
        f"\nProcessing {split_name}..."
    )


    raw_records = load_json_lines(
        input_file
    )


    processed_records = []


    for index, record in enumerate(
        raw_records
    ):

        record_id = (
            f"{split_name}_{index:06d}"
        )

        processed_records.append(
            process_record(
                record,
                record_id
            )
        )


    # Save processed data
    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            processed_records,
            file,
            indent=2,
            ensure_ascii=False
        )


    print(
        f"Saved {len(processed_records)} records"
    )

    print(
        f"Output: {output_file}"
    )


    return processed_records


# ============================================================
# PROCESS TRAIN / DEV / TEST
# ============================================================

train_data = process_split(
    "train"
)

dev_data = process_split(
    "dev"
)

test_data = process_split(
    "test"
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)

print(
    "SKILLSPAN PREPARATION COMPLETE"
)

print("=" * 70)

print(
    f"Training records:    {len(train_data)}"
)

print(
    f"Development records: {len(dev_data)}"
)

print(
    f"Test records:        {len(test_data)}"
)

print()

print(
    "Label mapping:",
    LABEL_TO_ID
)