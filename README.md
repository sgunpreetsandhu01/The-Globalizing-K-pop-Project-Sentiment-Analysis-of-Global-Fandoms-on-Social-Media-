# The Globalizing K-pop Project: Sentiment Analysis of Global Fandoms on Social Media

## Project Overview

This project focuses on analyzing the sentiment of K-pop fandoms using data collected from Twitter and Reddit. The main goal is to understand how social media interactions within global K-pop fandoms promote community well-being and emotional support. Several topic modeling techniques such as LDA, NMF, LSA, and BERTopic have been applied to analyze the data.

## Files and Directories

- **`embeddings_extraction.ipynb`**: This notebook is responsible for extracting text embeddings from social media data using multiple techniques like TF-IDF and Voyage-AI embeddings.
  
- **`Comparison_of_Voyage_AI_and_TF_IDF_final.ipynb`**: This notebook contains the comparison of BERTopic models based on different embeddings (Voyage-AI and TF-IDF). The comparison includes coherence scores and visualization of top words for each topic.

- **`Documentation.pdf`**: This file contains the detailed documentation of the project, including an abstract, methodology, evaluation, and conclusions.

## Project Workflow

1. **Data Collection**:
   - Twitter data was collected using the Twitter API.
   - Reddit data was collected using PRAW (Python Reddit API Wrapper).

2. **Data Preprocessing**:
   - Cleaned text by removing emojis, URLs, special characters, hashtags, and stopwords.
   - Tokenized and lemmatized the text for further analysis.
   - Conversations were linked by concatenating replies with the first tweet in the conversation.

3. **Text Representation**:
   - Applied Term Frequency-Inverse Document Frequency (TF-IDF) for text representation.
   - Used Voyage-AI embeddings for semantic analysis.

4. **Topic Modeling**:
   - Applied LDA, NMF, LSA, and BERTopic techniques to model topics within the K-pop fandom discussions.
   - Reduced dimensionality using UMAP and clustered text data using k-means clustering.

5. **Evaluation**:
   - Used coherence scores (Cv and U_mass) to evaluate model performance.
   - Visualized topic models using word clouds and bar charts.

## Key Results

- **Best Performing Model**: Latent Dirichlet Allocation (LDA) performed the best with a coherence score of 0.63.
- **BERTopic Comparison**: BERTopic with Voyage AI embeddings outperformed the TF-IDF embeddings with a Cv score of 0.59 and U_mass score of -3.10.
- **Visualization**: Word clouds and bar charts provide insights into the most significant words within each topic, highlighting the focus on emotional support and community interaction in the fandoms.

## Dependencies

The following Python packages are required to run the project:

- `pandas`
- `numpy`
- `matplotlib`
- `nltk`
- `sklearn`
- `bertopic`
- `umap-learn`
- `transformers`
- `torch`

## Note:

API keys needed for LLama2 and Voyage-AI

Install the required packages using the following command:

```bash
pip install pandas numpy matplotlib nltk sklearn bertopic umap-learn transformers torch
 
