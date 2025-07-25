# Nvidia Stock Price Prediction Pipeline

This project predicts Nvidia's next-day closing price using a multi-model deep learning ensemble with attention mechanisms and sentiment-aware confidence scoring.

---

## Project Structure

![Pipeline Architecture](nvidia_prediction_pipeline/nvidia_prediction_pipeline_architecture.png)

---

## How It Works

### Step 1: Base Models

Eight GRU-based models with Bahdanau Attention are trained using different lookback windows:

* 1, 14, 30, 60, 90, 180, 270, 365 days

Each outputs predictions and loss graphs to:

```
pipeline/ensemble_inputs/{lookback}D/run_{timestamp}/
```

### Step 2: Ridge Ensemble

A Ridge regressor is trained on the latest predictions from all eight models to form a meta-model.

* Saves to `pipeline/meta_model/`

### Step 3: Sentiment-Augmented Prediction

News headlines/descriptions from the past 5 days are:

* Fetched via NewsAPI
* * Register for NewsAPI API Key: [Newsapi/Register for API Key](https://newsapi.org/register)
* Scored via FinBERT (positive, neutral, negative)
* Classified as STRONG, NEUTRAL, or WEAK confidence

Logs and prediction are stored in:

```
pipeline/meta_model/ensemble_prediction_log.csv
```

---

## Running the Pipeline

### 1.  Environment
This project was developed and tested in:

Windows 11 + WSL2 (Ubuntu 22.04)

Python 3.12.11

GPU-accelerated pipelines may require CUDA-compatible GPU access inside WSL.

### 2. Install Dependencies

* Python==3.12.11

```bash
pip install -r requirements.txt
```

### 3. Save your NewsAPI key

```
keys/newsapi_key.txt
```

### 4. Add Nvidia Stock CSV

```
data/nvidia_stock_data.csv
```

***Update Nvidia Stock dataset with latest data.***

### 5. Run Full Pipeline

```bash
python pipeline/orchestrator.py
```

---

## Model Architecture

* GRU + Bahdanau Attention
* Dropout + Conv1D preprocessor (varies by lookback)
* Ridge ensemble regression
* FinBERT for sentiment confidence

---

## FinBERT Citation

This project uses the [**ProsusAI/finbert**](https://huggingface.co/ProsusAI/finbert) model for sentiment scoring of financial news.

**License & Contact:**  
For license information or FinBERT-related questions, please contact:  
- Dogu Araci — `dogu.araci[at]prosus[dot]com`  
- Zulkuf Genc — `zulkuf.genc[at]prosus[dot]com`  

For more information, visit [www.prosus.com](https://www.prosus.com)

---

## Dataset

This project uses the **[NVIDIA STOCK MARKET HISTORY](https://www.kaggle.com/datasets/adilshamim8/nvidia-stock-market-history)** dataset by **Adil Shamim**, originally published on [Kaggle](https://www.kaggle.com/), and **extended by PJEDeveloper to include data through July 22, 2025**.

**License:** CC0 Public Domain  
**Update Frequency:** Monthly  
**Source File:** `data/nvidia_stock_data.csv`

---

### Description

This dataset provides a detailed historical overview of NVIDIA Corporation’s stock market performance, capturing daily trading records across multiple years.

---

### Included Columns

| Column   | Description                           |
|----------|---------------------------------------|
| `Date`   | Trading date (YYYY-MM-DD)             |
| `Open`   | Opening price for the day             |
| `High`   | Highest price during the session      |
| `Low`    | Lowest price during the session       |
| `Close`  | Closing price of the stock            |
| `Volume` | Total number of shares traded         |

---

### Use Cases

- Time series forecasting  
- Financial modeling  
- Volatility & trend analysis  
- Educational demos in finance/data science  
- Algorithmic trading simulations
- **Academic research and educational use (non-commercial)**

---

## Outputs

* Prediction plots
* Loss plots
* Final predicted close with sentiment
* Logged predictions

---

## License

Apache 2.0 license

## Developer

**PJEDeveloper**
