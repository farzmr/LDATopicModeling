import pandas as pd
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
import nltk
from nltk.corpus import stopwords
import time

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
stop_words.extend(['from', 'first', 'second', 'using','one', 'method', 'methods', 'blockchain','may','based','includes','include']) #blockchain

print("Stopword list created")

docs = []

# Detect and remove bigrams and trigrams
bigram = gensim.models.Phrases(docs, min_count=5)
trigram = gensim.models.Phrases(bigram[docs])

for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            docs[idx].append(token)
    for token in trigram[docs[idx]]:
        if '_' in token:
            docs[idx].append(token)

for doc in df['text_combined']:
    tokens = [token for token in simple_preprocess(doc, deacc=True) if token not in stop_words]
    docs.append(tokens)

# Create a dictionary and corpus
dictionary = corpora.Dictionary(docs)
dictionary.filter_extremes(no_below=4, no_above=0.9)  # Remove low frequency words
corpus = [dictionary.doc2bow(doc) for doc in docs]

# Define values for seed, passes, and alpha
seed_values = [0, 42, 123]
passes_values = [5, 20]
alpha_values = [0.1, 1]

print('Tokenizing completed, start going to the loop')

# Open a text file to store the results
with open("lda_results_cited_block_2023_4_LF_Bi.txt", "w") as file:
    iteration_number = 0  # Initialize iteration counter
    for seed in seed_values:
        for passes in passes_values:
            for alpha in alpha_values:
                iteration_number += 1  # Increment iteration counter
                print(f"Iteration {iteration_number}:")
                num_topics = 4
                file.write(f'Running iteration {iteration_number} with seed={seed}, passes={passes}, alpha={alpha}\n')
                start_time = time.time()

                file.write('Train model started\n')
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

                file.write("Train model done\n")
                file.write(f"Time taken: {time.time() - start_time:.2f} seconds\n\n")

                # Calculate the number of patents in each topic
                topic_patent_counts = [0] * num_topics
                total_patents = len(corpus)

                # Count the number of patents in each topic
                for i, doc in enumerate(corpus):
                    dominant_topic = max(lda_model[doc], key=lambda x: x[1])[0]
                    topic_patent_counts[dominant_topic] += 1
                    df.at[i, f'dominant_topic_{seed}_{passes}_{alpha}'] = dominant_topic

                # Write the results for each topic to the file and print them
                for topic in range(num_topics):
                    topic_words_str = ', '.join([word for word, prob in lda_model.show_topic(topic)])
                    topic_patents = topic_patent_counts[topic]
                    topic_percentage = (topic_patents / total_patents) * 100
                    file.write(f"Topic {topic}:\n")
                    file.write(f"Top words: {topic_words_str}\n")
                    file.write(f"Number of patents = {topic_patents} out of {total_patents} ({topic_percentage:.2f}%)\n\n")
                    print(f"Topic {topic}:")
                    print(f"Top words: {topic_words_str}")
                    print(f"Number of patents = {topic_patents} out of {total_patents} ({topic_percentage:.2f}%)\n")

                file.write("-------------------------------------------------------\n")
                print("-------------------------------------------------------")

# Save the modified DataFrame to a CSV file
df.to_csv('cited_block_2023_LDA_with_topics_4_LF_Bi_4_09.csv', index=False)
print('CSV result file was stored')

current_time = time.strftime("%H:%M:%S", time.localtime())
print("Current time:", current_time)
