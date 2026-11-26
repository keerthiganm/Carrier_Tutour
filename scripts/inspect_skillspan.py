# ============================================================
# SKILLSPAN DATASET INSPECTION
#
# Purpose:
# Understand the structure of the SkillSpan dataset before
# building our own skill-extraction model.
#
# We are NOT training anything here.
# This file only reads and inspects the raw dataset.
# ============================================================


import json
from pathlib import Path


# ============================================================
# 1. LOCATE THE DATASET FILES
# ============================================================

# Folder containing this Python file
BASE_DIR = Path(__file__).parent

# SkillSpan dataset folder
SKILLSPAN_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "skillspan"
    / "json"
)

# Individual dataset files
TRAIN_FILE = SKILLSPAN_DIR / "train.json"
DEV_FILE = SKILLSPAN_DIR / "dev.json"
TEST_FILE = SKILLSPAN_DIR / "test.json"


# ============================================================
# 2. FUNCTION TO READ SKILLSPAN FILES
# ============================================================

def load_json_lines(file_path):
    """
    Read a JSON Lines / JSONL-style file.

    SkillSpan stores one JSON object per line instead of
    storing all records inside one JSON list.

    Example file:

        {"record": 1}
        {"record": 2}
        {"record": 3}

    Parameters:
        file_path:
            Path to the dataset file.

    Returns:
        list:
            List containing each JSON record as a
            Python dictionary.
    """

    records = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        # Read the file one line at a time
        for line_number, line in enumerate(
            file,
            start=1
        ):

            # Remove blank spaces/newlines
            line = line.strip()

            # Ignore empty lines
            if not line:
                continue

            try:

                # Convert this individual JSON line
                # into a Python dictionary
                record = json.loads(line)

                records.append(record)

            except json.JSONDecodeError as error:

                print(
                    f"Unable to read line "
                    f"{line_number} in "
                    f"{file_path.name}"
                )

                print(error)

    return records


# ============================================================
# 3. LOAD TRAIN / DEV / TEST DATA
# ============================================================

train_data = load_json_lines(
    TRAIN_FILE
)

dev_data = load_json_lines(
    DEV_FILE
)

test_data = load_json_lines(
    TEST_FILE
)


# ============================================================
# 4. DISPLAY BASIC DATASET INFORMATION
# ============================================================

print()
print("=" * 70)
print("SKILLSPAN DATASET SUMMARY")
print("=" * 70)

print()

print("Python type:")
print(type(train_data))

print()

print(
    "Training records:",
    len(train_data)
)

print(
    "Development records:",
    len(dev_data)
)

print(
    "Test records:",
    len(test_data)
)

print()

print(
    "Total records:",
    len(train_data)
    + len(dev_data)
    + len(test_data)
)


# ============================================================
# 5. INSPECT THE FIRST TRAINING RECORD
# ============================================================

print()
print("=" * 70)
print("FIRST TRAINING RECORD")
print("=" * 70)

print(
    json.dumps(
        train_data[0],
        indent=4,
        ensure_ascii=False
    )
)


# ============================================================
# 6. DISPLAY THE AVAILABLE FIELDS
# ============================================================

print()
print("=" * 70)
print("FIELDS AVAILABLE IN FIRST RECORD")
print("=" * 70)

for key in train_data[0].keys():

    print("-", key)


# ============================================================
# 7. INSPECT A FEW TRAINING RECORDS
# ============================================================

print()
print("=" * 70)
print("FIRST 3 TRAINING RECORDS")
print("=" * 70)


# min() protects us in case a dataset contains
# fewer than 3 records.
number_of_examples = min(
    3,
    len(train_data)
)


for index in range(number_of_examples):

    print()

    print(
        f"RECORD {index + 1}"
    )

    print("-" * 50)

    print(
        json.dumps(
            train_data[index],
            indent=4,
            ensure_ascii=False
        )
    )


# ============================================================
# 8. BASIC SAFETY CHECKS
# ============================================================

print()
print("=" * 70)
print("BASIC DATASET CHECKS")
print("=" * 70)


if len(train_data) == 0:

    print(
        "WARNING: Training dataset is empty."
    )

else:

    print(
        "✓ Training dataset loaded successfully."
    )


if len(dev_data) == 0:

    print(
        "WARNING: Development dataset is empty."
    )

else:

    print(
        "✓ Development dataset loaded successfully."
    )


if len(test_data) == 0:

    print(
        "WARNING: Test dataset is empty."
    )

else:

    print(
        "✓ Test dataset loaded successfully."
    )


print()
print("=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)

# ============================================================
# 9. FIND EXAMPLES THAT CONTAIN SKILL LABELS
# ============================================================

print()
print("=" * 70)
print("EXAMPLES CONTAINING SKILLS")
print("=" * 70)


skill_examples_found = 0


for record in train_data:

    skill_tags = record["tags_skill"]

    # A record contains an annotated skill
    # if at least one label is not "O"
    contains_skill = any(
        tag != "O"
        for tag in skill_tags
    )

    if contains_skill:

        skill_examples_found += 1

        print()
        print(
            f"SKILL EXAMPLE "
            f"{skill_examples_found}"
        )

        print("-" * 50)

        # Reconstruct the readable sentence
        sentence = " ".join(
            record["tokens"]
        )

        print("Sentence:")
        print(sentence)

        print()

        print("Tokens and labels:")

        # Display each token beside its skill label
        for token, tag in zip(
            record["tokens"],
            record["tags_skill"]
        ):

            print(
                f"{token:25} {tag}"
            )

        # We only need the first 5 examples
        if skill_examples_found == 5:

            break
            
# ============================================================
# 10. INSPECT SKILL AND KNOWLEDGE LABELS TOGETHER
# ============================================================

print()
print("=" * 70)
print("SKILL + KNOWLEDGE LABEL EXAMPLES")
print("=" * 70)


examples_found = 0


for record in train_data:

    skill_tags = record["tags_skill"]
    knowledge_tags = record["tags_knowledge"]

    # We want records containing either
    # a skill annotation or knowledge annotation
    contains_annotation = any(
        skill_tag != "O"
        or knowledge_tag != "O"

        for skill_tag, knowledge_tag
        in zip(
            skill_tags,
            knowledge_tags
        )
    )

    if contains_annotation:

        examples_found += 1

        print()
        print(
            f"EXAMPLE {examples_found}"
        )

        print("-" * 70)

        sentence = " ".join(
            record["tokens"]
        )

        print("Sentence:")
        print(sentence)

        print()

        # Column headings
        print(
            f"{'TOKEN':25}"
            f"{'SKILL':12}"
            f"{'KNOWLEDGE':12}"
        )

        print("-" * 50)


        # Display token with BOTH labels
        for (
            token,
            skill_tag,
            knowledge_tag
        ) in zip(
            record["tokens"],
            skill_tags,
            knowledge_tags
        ):

            print(
                f"{token:25}"
                f"{skill_tag:12}"
                f"{knowledge_tag:12}"
            )


        if examples_found == 10:

            break

# ============================================================
# 11. COUNT SKILL AND KNOWLEDGE LABELS
# ============================================================

from collections import Counter


skill_label_counts = Counter()
knowledge_label_counts = Counter()


for record in train_data:

    skill_label_counts.update(
        record["tags_skill"]
    )

    knowledge_label_counts.update(
        record["tags_knowledge"]
    )


print()
print("=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)


print()
print("Skill labels:")

for label, count in skill_label_counts.items():

    print(
        label,
        ":",
        count
    )


print()
print("Knowledge labels:")

for label, count in knowledge_label_counts.items():

    print(
        label,
        ":",
        count
    )