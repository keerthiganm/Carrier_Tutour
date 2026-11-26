# ============================================================
# TEST DISTILBERT TOKENIZATION
#
# Purpose:
# Understand how DistilBERT tokenizes SkillSpan words before
# we build the training pipeline.
#
# We are NOT training the model yet.
# ============================================================

from transformers import AutoTokenizer


# ============================================================
# 1. LOAD DISTILBERT TOKENIZER
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

print("Tokenizer loaded successfully.")


# ============================================================
# 2. CREATE A SKILLSPAN-LIKE EXAMPLE
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


# These labels are similar to the knowledge labels
# we observed in SkillSpan.
#
# O = 0
# B = 1
# I = 2

knowledge_labels = [
    0,   # Experience
    0,   # in
    0,   # working
    0,   # on
    0,   # a
    0,   # cloud-based
    0,   # application
    0,   # running
    0,   # on
    1    # Docker → B-KNOWLEDGE
]


# ============================================================
# 3. TOKENIZE THE WORDS
# ============================================================

encoding = tokenizer(
    words,

    # Tells tokenizer the input is already split into words
    is_split_into_words=True,

    # Add DistilBERT special tokens
    add_special_tokens=True
)


# ============================================================
# 4. INSPECT TOKENIZER OUTPUT
# ============================================================

token_ids = encoding["input_ids"]

tokens = tokenizer.convert_ids_to_tokens(
    token_ids
)

# word_ids tells us which original word
# each DistilBERT token belongs to.
word_ids = encoding.word_ids()


print()
print("=" * 70)
print("ORIGINAL WORDS")
print("=" * 70)

for index, word in enumerate(words):

    print(
        index,
        word,
        "label =",
        knowledge_labels[index]
    )


print()
print("=" * 70)
print("DISTILBERT TOKENS")
print("=" * 70)

print(
    f"{'TOKEN':20}"
    f"{'WORD ID':12}"
    f"{'ORIGINAL WORD':25}"
)

print("-" * 60)


for token, word_id in zip(
    tokens,
    word_ids
):

    if word_id is None:

        original_word = "SPECIAL TOKEN"

    else:

        original_word = words[word_id]

    print(
        f"{token:20}"
        f"{str(word_id):12}"
        f"{original_word:25}"
    )