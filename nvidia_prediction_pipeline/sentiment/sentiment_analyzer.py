# sentiment/sentiment_analyzer.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load FinBERT model
_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


def compute_sentiment_scores(articles):
    all_pos = []
    all_neu = []
    all_neg = []

    for article in articles:
        text = (article.get("title") or "") + ". " + (article.get("description") or "")
        if not text.strip():
            continue

        inputs = _tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = _model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

        all_neg.append(probs[0].item())
        all_neu.append(probs[1].item())
        all_pos.append(probs[2].item())

    if not all_pos:
        return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

    return {
        "positive": sum(all_pos) / len(all_pos),
        "neutral": sum(all_neu) / len(all_neu),
        "negative": sum(all_neg) / len(all_neg)
    }