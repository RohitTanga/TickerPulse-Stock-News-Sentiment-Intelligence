# 📈 TickerPulse — Stock News Sentiment Intelligence

An MLOps pipeline that scrapes financial news, labels it, retrains a BERT classifier every week, and serves live sentiment for any stock through a web dashboard.

🔗 **Live app:** https://frontend-iazu.onrender.com/
🤗 **Model:** https://huggingface.co/dhanushbitra/bert_sentiment_trainer

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Hopsworks](https://img.shields.io/badge/Hopsworks-1EB182?style=for-the-badge&logoColor=white)
![Modal](https://img.shields.io/badge/Modal-7FEE64?style=for-the-badge&logoColor=black)
![Optuna](https://img.shields.io/badge/Optuna-2C7BB6?style=for-the-badge&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Node.js](https://img.shields.io/badge/Node.js-339933?style=for-the-badge&logo=nodedotjs&logoColor=white)

---

## 🧠 What it does

- Fine-tunes `bert-base-cased` for 3-class financial sentiment (negative / positive / neutral)
- Scrapes Yahoo News every week for fresh headlines and auto-labels them
- Stores features in a Hopsworks feature store and grows the dataset over time
- Retrains on Modal (A10G GPU) and **only publishes the new model if it beats the current one**
- Lets users search any ticker or company and see a 7-day sentiment breakdown

## 📊 Results

| Metric | Value |
|--------|-------|
| Accuracy | **89.47%** |
| Loss | 0.5985 |
| Base model | `bert-base-cased` |
| Classes | Negative (0), Positive (1), Neutral (2) |

Hyperparameters came from an Optuna search (10 trials, about 6 hours on a Colab T4).

## 🏗️ Architecture

```
Yahoo News (AAPL, AMZN, GOOGL, MSFT, TSLA)
        ↓
Scraper → distilRoBERTa labels each headline
        ↓
Text encoded with ada-002 tokenizer
        ↓
Balanced + stratified 80/20 split
        ↓
Hopsworks Feature Store (train / test feature groups)
        ↓
Modal weekly cron (Mon 06:00 UTC) → fine-tune BERT
        ↓
New accuracy > old accuracy?  →  push to Hugging Face
        ↓
React frontend ← Node API (live scraping) + HF Inference API
```

## 🗂️ Datasets

| Dataset | Labels | Notes |
|---------|--------|-------|
| [Financial PhraseBank](https://huggingface.co/datasets/financial_phrasebank) | negative / neutral / positive | 75% annotator agreement subset |
| [Zeroshot Twitter Financial News](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | bearish / bullish / neutral | Remapped to the same 3 labels |

Weekly scraped headlines are added on top: 5 tickers, up to 50 headlines each, past 7 days, class-balanced before upload.

## 📁 Project Structure

| Path | Purpose |
|------|---------|
| `preprocessing_pipeline.ipynb` | Preprocess base data, test the tokenizer encode/decode round trip |
| `data_mod.py` | Convert raw Financial PhraseBank text into CSV |
| `feature_pipeline.ipynb` | One-time upload of base data, creates train/test feature groups |
| `feature_pipeline_weekly.py` | Weekly scrape → label → encode → upload to Hopsworks |
| `yahoo_finance_news_scraper.py` | Scraper + labeling + encoding (use Python 3.8 or 3.9) |
| `hyperparameter_search.ipynb` | Optuna search via Hugging Face `Trainer` |
| `training_pipeline_notebook.ipynb` | First training run and model upload |
| `training_pipeline.py` | Weekly retraining on Modal with accuracy gate |
| `deploy_weekly_training.sh` | Deploys the training job to Modal |
| `sentiment_analysis_backend/` | Node.js API (JS port of the scraper) |
| `sentiment_analysis_frontend/` | React dashboard |

## ⚙️ Training Config

```python
TrainingArguments(
    per_device_train_batch_size=16,
    learning_rate=2.754984679344267e-05,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,
    max_steps=3000,
    eval_steps=250,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    seed=42,
)
```

## 🚀 Setup

### Pipelines

```bash
pip install -r requirements.txt
```

Set your `HOPSWORKS_API_KEY`, then run the notebooks or scripts directly. To deploy weekly retraining:

```bash
modal deploy --name project_weekly_training training_pipeline.py
```

### Backend

```bash
cd sentiment_analysis_backend
npm install
npm run dev
```

For local use, uncomment line 15 and comment line 14 in `app.js` to allow frontend requests (CORS).

Endpoint: `GET /sentiment-analysis?searchKey=TSLA&maxArticlesPerSearch=20`

```json
{
  "result": [
    { "headline": "string", "posted": "Date | null", "text": "string", "href": "string" }
  ]
}
```

### Frontend

```bash
cd sentiment_analysis_frontend
npm install
npm run dev
```

Create `credentials.json` in the frontend folder:

```json
{ "huggingface": "<your huggingface API key>" }
```

Then uncomment line 130 and comment line 129 in `src/pages/Index.jsx` to point at your local API.

## 🔭 What's next

- Correlate sentiment trends with historical price data
- Forecast price movement from sentiment signals
- Explore deeper models for sharper market-shift detection

## 👥 Built by

Dhanush Bitra, Nived Krishna, Sasi Kiran Reddy — ICS322 course project
