import pandas as pd
import gensim
from gensim import corpora
from gensim.models import CoherenceModel, Phrases
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
import time
import matplotlib.pyplot as plt
import itertools

if __name__ == '__main__':
    print("Import completed")

    # Load the data into a pandas dataframe
    df = pd.read_csv("all_data_blockchain_2023_cited.csv")
    print('Reading file completed')

    # Convert 'patent_title' and 'patent_abstract' columns to strings
    print('Converting title and abstract to str')
    df['patent_title'] = df['patent_title'].astype(str)
    df['patent_abstract'] = df['patent_abstract'].astype(str)

    # Concatenate the 'title' and 'text' columns
    df['text_combined'] = df['patent_title'] + ' ' + df['patent_abstract']

    # Create a list of stopwords
    nltk.download('stopwords')
    stop_words = stopwords.words('english')

    stop_words.extend(['from', 'first', 'second', 'one','using' ,'method', 'methods', 'blockchain', 'may', 'based', 'includes','include'])  # blockchain

    print("Stopword list created")

    docs = []
    for doc in df['text_combined']:
        tokens = [token for token in simple_preprocess(doc, deacc=True) if token not in stop_words]
        docs.append(tokens)

    # Add bigrams and trigrams to docs (only those that appear 5 times or more)
    bigram = Phrases(docs, min_count=5)
    trigram = Phrases(bigram[docs])

    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)
        for token in trigram[docs[idx]]:
            if '_' in token:
                # Token is a trigram, add to document.
                docs[idx].append(token)

    # Create a dictionary and corpus
    dictionary = corpora.Dictionary(docs)

    # Remove words that appear in less than 2 documents and more than 70% of the documents
    dictionary.filter_extremes(no_below=4, no_above=0.9)

    corpus = [dictionary.doc2bow(doc) for doc in docs]

    num_topics_list = range(2, 15)

    # Calculate coherence and perplexity values
    coherence_values = []
    perplexity_values = []

    condition_counter = 1

    for seed, passes, alpha in itertools.product([0, 42, 123], [5, 20], [0.1, 1]):
        print(f'Condition{condition_counter}: Seed={seed}, Passes={passes}, Alpha={alpha}')
        condition_counter += 1
        coherence_values = []
        perplexity_values = []
        fig, ax1 = plt.subplots(figsize=(10, 6))

        for num_topics in num_topics_list:
            print(f'    Iteration: Number of topics = {num_topics}')

            # Train the LDA model
            lda_model = gensim.models.ldamodel.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                random_state=seed,
                passes=passes,
                alpha=alpha,
                eta='auto'
            )
            print('    Model training done')

            # Calculate coherence
            coherence_model_lda = CoherenceModel(model=lda_model, texts=docs, dictionary=dictionary, coherence='c_v')
            coherence = coherence_model_lda.get_coherence()
            coherence_values.append(coherence)

            # Calculate perplexity
            perplexity = lda_model.log_perplexity(corpus)
            perplexity_values.append(perplexity)

            print(f'    Coherence: {coherence}, Perplexity: {perplexity}')

        # Plot coherence and perplexity values for the current condition
        color = 'tab:red'
        ax1.set_xlabel('Number of Topics')
        ax1.set_ylabel('Perplexity', color=color)
        ax1.plot(num_topics_list, perplexity_values, color=color, marker='s')
        ax1.tick_params(axis='y', labelcolor=color)

        color = 'tab:blue'
        ax2 = ax1.twinx()
        ax2.set_ylabel('Coherence', color=color)
        ax2.plot(num_topics_list, coherence_values, color=color, marker='o')
        ax2.tick_params(axis='y', labelcolor=color)
        ax1.set_xticks(num_topics_list)

        fig.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust the spacing between the plot and the title
        original_title = 'Coherence and Perplexity vs. Number of Topics'
        additional_info = f'seed={seed}, passes={passes}, alpha={alpha}'
        title = original_title + '\n' + additional_info

        plt.title(title, fontsize=12)
        plt.savefig(f'cited_block_2023_bigram_LowFreq_4_0.9_coherence_perplexity_seed_{seed}_passes_{passes}_alpha_{alpha}.png')
        plt.show()
