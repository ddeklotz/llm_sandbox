import time
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize

# Download nltk data
nltk.download('punkt')

def read_file_lines(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            return lines
    except FileNotFoundError:
        print("File not found.")
        return []
    

    
def compute_word2vec_embeddings(lines):
    # Tokenize each line into words
    tokenized_lines = [word_tokenize(line.lower()) for line in lines]

    # Train Word2Vec model
    model = Word2Vec(tokenized_lines, vector_size=100, window=5, min_count=1, workers=4)

    # Compute embeddings for each line
    embeddings = []
    for line in tokenized_lines:
        line_embedding = [model.wv[word] for word in line if word in model.wv]
        if line_embedding:
            # Compute average embedding for the line
            line_embedding_avg = sum(line_embedding) / len(line_embedding)
            embeddings.append(line_embedding_avg)
        else:
            # If no word in the line is present in the vocabulary, add a zero vector
            embeddings.append([0] * model.vector_size)
    
    return embeddings

def main():
    start_time = time.time()
    file_path = "..\\client\\src\\core\\wordlist.txt"
    #file_path = ".\\wordlist_short.txt"
    lines = read_file_lines(file_path)

    if lines:
        print("Computing Word2Vec embeddings for each entry...")
        embeddings = compute_word2vec_embeddings(lines)
        print("Word2Vec embeddings computed successfully.")
        print("Shape of the embeddings array:", len(embeddings), "x", len(embeddings[0]))
    else:
        print("No lines to display.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")

if __name__ == "__main__":
    main()
