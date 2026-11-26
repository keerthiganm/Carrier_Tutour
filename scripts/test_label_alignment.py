# ============================================================
# TEST LABEL ALIGNMENT
#
# Purpose:
#
# Convert SkillSpan word-level BIO labels into
# DistilBERT subword-level labels.
#
# O -> 0
# B -> 1
# I -> 2
#
# Special tokens such as [CLS] and [SEP] -> -100
# ============================================================


from transformers import AutoTokenizer


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MODEL_NAME = "distilbert-base-uncased"

O_LABEL = 0
B_LABEL = 1
I_LABEL = 2

IGNORE_LABEL = -100


# ------------------------------------------------------------
# Load tokenizer
# ------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)


# ------------------------------------------------------------
# Function: align original labels with transformer tokens
# ------------------------------------------------------------

def align_labels_with_tokens(
    words,
    word_labels
):
    """
    Tokenize a list of original words and align their BIO
    labels with DistilBERT's subword tokens.

    Example:

        Docker -> B

    may become:

        dock -> B
        ##er -> I
    """

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=True,
        truncation=True
    )

    word_ids = encoding.word_ids()

    aligned_labels = []

    previous_word_id = None


    for word_id in word_ids:

        # ----------------------------------------
        # Special transformer tokens
        # ----------------------------------------

        if word_id is None:

            aligned_labels.append(
                IGNORE_LABEL
            )


        # ----------------------------------------
        # First transformer token belonging
        # to an original word
        # ----------------------------------------

        elif word_id != previous_word_id:

            original_label = (
                word_labels[word_id]
            )

            aligned_labels.append(
                original_label
            )


        # ----------------------------------------
        # Additional subword belonging
        # to the SAME original word
        # ----------------------------------------

        else:

            original_label = (
                word_labels[word_id]
            )

            # If original word begins an entity,
            # continuation subwords should become I.
            if original_label == B_LABEL:

                aligned_labels.append(
                    I_LABEL
                )

            else:

                # O stays O.
                # I stays I.
                aligned_labels.append(
                    original_label
                )


        previous_word_id = word_id


    return encoding, aligned_labels


# ============================================================
# Test example
# ============================================================

words = [
    "Experience",
    "in",
    "working",
    "on",
    "a",
    "cloud-based",
    "application",
    "running",
    "on",
    "Docker"
]


knowledge_labels = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1
]


encoding, aligned_labels = (
    align_labels_with_tokens(
        words,
        knowledge_labels
    )
)


# Convert IDs back into readable tokens
tokens = tokenizer.convert_ids_to_tokens(
    encoding["input_ids"]
)


# ============================================================
# Display result
# ============================================================

print()

print("=" * 70)
print("TOKEN / LABEL ALIGNMENT")
print("=" * 70)

print(
    f"{'TOKEN':20}"
    f"{'LABEL ID':12}"
    f"{'LABEL':15}"
)

print("-" * 50)


label_names = {
    -100: "IGNORE",
    0: "O",
    1: "B",
    2: "I"
}


for token, label_id in zip(
    tokens,
    aligned_labels
):

    print(
        f"{token:20}"
        f"{label_id:<12}"
        f"{label_names[label_id]:15}"
    )