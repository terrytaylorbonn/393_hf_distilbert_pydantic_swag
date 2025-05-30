from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.utils import logging

logging.set_verbosity_error()  # Hide warnings

model_id = "facebook/bart-large-cnn"

try:
    # Try offline first
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, local_files_only=True)
    print("Loaded model from local cache.")
except:
    # Fallback to online download (only if internet is available)
    print("Model not cached. Downloading...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
print("Model downloaded and cached for future offline use.")

