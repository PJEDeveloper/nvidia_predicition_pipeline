# sentiment/sentiment_confidence.py

def classify_confidence(scores):
    """
    Classify overall confidence level based on sentiment scores.
    
    Args:
        scores (dict): Dictionary with keys 'positive', 'neutral', 'negative'

    Returns:
        str: One of "STRONG", "WEAK", or "NEUTRAL"
    """
    pos = scores.get("positive", 0.0)
    neg = scores.get("negative", 0.0)

    if pos >= 0.5:
        return "STRONG"
    elif neg >= 0.5:
        return "WEAK"
    else:
        return "NEUTRAL"