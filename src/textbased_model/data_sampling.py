import time
from gensim.models import KeyedVectors
import sys
sys.path.append("/raid/omgempathy/")  # path to the main dir
from src.textbased_model import utils, config
from collections import OrderedDict


training_dir = config.training_data_dir
# load data
data = utils.load_data_dir(training_dir)

# data sampling
sampled_data = utils.apply_data_sampling(data, -1)

for filename, d in sampled_data.items():
    print(filename, len(data[filename]), "#set: ", len(d))
    print(d)


# word_embeddings_file = config.glove_word2vec_file
# start = time.time()
# print("Loading word embeddings...")
# word_embeddings = KeyedVectors.load_word2vec_format(word_embeddings_file, binary=False)
# vocabs = word_embeddings.wv.vocab
# utils.print_time(start)
#
# data_tensor = utils.prepare_data(data, vocabs, word_embeddings)

# data sampling
