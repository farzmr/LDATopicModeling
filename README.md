
# Patent Data Topic Modeling and Evaluation

This repository contains Python scripts for topic modeling and evaluation using Latent Dirichlet Allocation (LDA) on patent data. The scripts perform topic modeling, evaluate model performance, and visualize results.  This work is part of my thesis project focused on analyzing blockchain patents.

The primary goal of these scripts is to identify dominant topics within blockchain patents, effectively categorizing various aspects of blockchain technology. This analysis aids in understanding the blockchain industry's landscape, tracking its evolution and trajectory, and pinpointing which companies excel in specific areas of blockchain technology.


## Prerequisites

Before running the scripts, ensure you have the necessary data and libraries installed:

1.**Python Packages**  
   Ensure you have the following Python packages installed:
   - pandas
   - gensim
   - nltk
   - matplotlib

2.**Dataset**

Download and use the dataset file `all_data_blockchain_2023_cited.csv` which is uploaded in this repository. 

This file includes a comprehensive list of all cited blockchain patents up to the end of 2023. It was compiled using the scripts from the *USPatentDataAnalysis* repository.



## Scripts Overview

### 1. `lda_topic_modeling.py`

**Purpose:**  
Performs topic modeling using Latent Dirichlet Allocation (LDA) on patent data to identify topics and analyze their distribution across the dataset.

**Inputs:**
- `all_data_blockchain_2023_cited.csv`: CSV file containing patent data with columns `patent_title` and `patent_abstract`.

**Process:**
1. Loads and preprocesses patent data.
2. Creates a customized list of stopwords and removes them from the text.
3. Tokenizes the text and adds bigrams and trigrams.
4. Trains multiple LDA models with different parameters (seed, passes, alpha) using 12 different setups: seed values [0, 42, 123], passes [5, 20], alpha [0.1, 1].
5. Evaluates and writes results to a text file, including the number of patents per topic.
6. Saves the DataFrame with dominant topics to a CSV file.

**Sample Result:**
- **Topic 1:**  
  Top words: data, system, block, ledger, device, distributed, transaction, information, storage, associated  
  Number of patents = 1403 out of 3810 (36.82%)

- **Topic 2:**  
  Top words: transaction, contract, smart, network, digital, system, ledger, distributed, node, user  
  Number of patents = 1376 out of 3810 (36.12%)

- **Topic 3:**  
  Top words: key, network, system, data, node, user, service, public, block, device  
  Number of patents = 1031 out of 3810 (27.06%)

### 2. `lda_evaluation.py`

**Purpose:**  
Evaluates LDA models to find the optimal number of topics by assessing coherence and perplexity values. This script helps identify the number of topics that provide the best balance between coherence and perplexity for blockchain patents.

**Inputs:**
- `all_data_blockchain_2023_cited.csv`: CSV file containing patent data with columns `patent_title` and `patent_abstract`.

**Process:**
1. Loads and preprocesses patent data.
2. Creates a customized list of stopwords and removes them from the text.
3. Tokenizes the text and adds bigrams and trigrams.
4. Trains LDA models for various numbers of topics and parameters (seed, passes, alpha) using 12 different setups: seed values [0, 42, 123], passes [5, 20], alpha [0.1, 1].
5. Computes coherence and perplexity values for each model.
6. Plots and saves graphs of coherence and perplexity versus the number of topics to identify the best model.

**Sample Result:**
- Saves plots showing coherence and perplexity values against the number of topics, helping to determine the optimal number of topics for the dataset.
