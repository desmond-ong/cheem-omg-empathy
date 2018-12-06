from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors
import time
import sys
sys.path.append("/raid/omgempathy/")  # path to the main dir
from src.textbased_model import utils, config
import json
import os
from nltk import word_tokenize

# glove_file = "/Users/sonnguyen/Datasets/glove/glove.840B.300d.txt"
# w2v_output = "/Users/sonnguyen/Datasets/glove/glove.840B.300d.txt.word2vec"
# glove_file = "/Users/sonnguyen/Datasets/glove/glove.6B/glove.6B.50d.txt"
# w2v_output = "/Users/sonnguyen/Datasets/glove/glove.6B/glove.6B.50d.txt.word2vec"


# convert glove to word2vec format
# start = time.time()
# glove2word2vec(glove_file, w2v_output)
# print("Time: {}".format(utils.time_since(start)))

# load word2vec-format word embeddings
# model = KeyedVectors.load_word2vec_format(w2v_output, binary=False)
# # calculate: (king - man) + woman = ?
# result = model.most_similar(positive=['usa', 'france'], negative=['paris'], topn=1)
# print(result)
# len(model.wv.vocab)

# only save words appear in the text
def load_contents(json_dir, words):
    for filename in os.listdir(json_dir):
        if filename.startswith(".") or not filename.endswith('.json'):
            continue
        with open(os.path.join(json_dir, filename), 'r') as reader:
            data = json.load(reader)
            for utter in data:
                ws = word_tokenize(utter['text'])
                for w in ws:
                    words.append(w)


glove_file = config.glove_word2vec_file
we_dim = 300
output_file = config.glove_word2vec_file_filtered
content_dirs = ['data/Validation/transcripts_with_scores/',
                'data/Training/transcripts_with_scores/',
                'data/Testing/transcripts_with_scores/']
words = []
for i in range(len(content_dirs)):
    load_contents(content_dirs[i], words)
words = list(set(words))


start = time.time()
print("Load word embeddings")
model = KeyedVectors.load_word2vec_format(glove_file, binary=False)
utils.print_time(start)

embeddings = []
emb_words = []
for word in words:
    if word in model:
        emb_words.append(word)
        embeddings.append(model[word])

we = Word2VecKeyedVectors(we_dim)
we.add(emb_words, embeddings)
we.save_word2vec_format(output_file)
