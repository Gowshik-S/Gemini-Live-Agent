# Rio ML — Dataset & Model Reference
#
# This file documents the datasets used for pre-training the ensemble model
# and how to download them. The model also learns online from each user's
# own interaction data (stored in rio_patterns.db).

## Required Datasets for Pre-Training

### 1. Chat/Conversation Style Classification
# Used to bootstrap the chat-style classifier (verbose/concise, formal/casual)

# **Kaggle — Customer Support on Twitter**
# ~3M customer support tweets — good for assistant interaction patterns
# https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
# Download: customer-support-on-twitter.zip → extract to ml/datasets/twitter_support/

# **Hugging Face — ShareGPT (Cleaned)**
# ~90K multi-turn conversations with AI assistants
# https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
# Download: ShareGPT_V3_unfiltered_cleaned_split.json → ml/datasets/sharegpt/

### 2. Developer Activity / Error Patterns
# Used to pre-train error classification and developer behavior models

# **GitHub — CodeSearchNet**
# 2M code+comment pairs across 6 languages (Python, JS, Ruby, Go, Java, PHP)
# https://huggingface.co/datasets/code_search_net/code_search_net
# Download via: datasets library → ml/datasets/codesearchnet/

# **Stack Overflow Questions Dataset**
# Tagged Q&A for error pattern classification
# https://www.kaggle.com/datasets/stackoverflow/stackoverflow
# Download: Questions.csv → ml/datasets/stackoverflow/

### 3. User Engagement / Session Patterns
# Used to pre-train session timing and engagement models

# **UCI — Online Retail Dataset**
# 500K user sessions with temporal patterns (good for session modeling)
# https://archive.ics.uci.edu/dataset/352/online+retail
# Download: Online Retail.xlsx → ml/datasets/uci_retail/

### 4. Sentiment / Frustration Detection
# Used to pre-train the frustration/struggle classifier

# **Kaggle — Sentiment140**
# 1.6M tweets with sentiment labels (proxy for frustration detection)
# https://www.kaggle.com/datasets/kazanova/sentiment140
# Download: training.1600000.processed.noemoticon.csv → ml/datasets/sentiment140/

# **GoEmotions (Google)**
# 58K Reddit comments labeled with 27 emotions including frustration, confusion
# https://huggingface.co/datasets/google-research-datasets/go_emotions
# Download via: datasets library → ml/datasets/goemotions/

## Dataset Directory Structure
# ml/
# └── datasets/
#     ├── README.md              ← this file
#     ├── twitter_support/       ← Customer Support on Twitter
#     ├── sharegpt/              ← ShareGPT conversations
#     ├── codesearchnet/         ← Code+comment pairs
#     ├── stackoverflow/         ← Stack Overflow Q&A
#     ├── uci_retail/            ← UCI Online Retail sessions
#     ├── sentiment140/          ← Sentiment dataset
#     └── goemotions/            ← GoEmotions multi-label

## Notes:
# - Pre-training is OPTIONAL. The model works from day 1 with cold-start defaults.
# - Online learning from actual user data begins immediately on first session.
# - Pre-training simply gives better initial predictions.
# - Total dataset size if all downloaded: ~2-5 GB
# - Minimum viable: sentiment140 + sharegpt (~500 MB)
