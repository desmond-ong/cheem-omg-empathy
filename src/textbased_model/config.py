import torch

# glove_word2vec_file = "/Users/sonnguyen/Datasets/glove/glove.6B/glove.6B.50d_word2vec.txt"
# glove_word2vec_file = "/home/son/data/glove/glove.6B.50d_word2vec.txt"
# glove_word2vec_file = "/home/son/data/glove/glove.6B.100d.txt.word2vec"
glove_word2vec_file = "/home/son/data/glove/glove.840B.300d.txt.word2vec"

glove_word2vec_file_filtered = 'data/glove.840B.300d.filtered.word2vec'

input_dim = 300

gpu = 2

use_stopwords = False

training_data_dir = "data/Training/transcripts_with_scores/"
training_data_difference_dir = "data/Training/transcripts_with_scores_difference/"

groundtruth_sequences_training_dir = "data/Training/Annotations/"

validation = {"data": "data/Validation/transcripts_with_scores/",
              'labels': "data/Validation/Annotations/",
              'prediction_output': 'data/Validation/model_predictions/'}

training = {"data": "data/Training/transcripts_with_scores/",
            'labels': "data/Training/Annotations/",
            'prediction_output': 'data/Training/model_predictions/'}

models_dir = "data/Training/models/"
log_dir = "data/Training/training_logs/"
model_save_epoch_num = 50  # save model after every ... epochs

chunk_size = 15
batch_size = 20

iter_print = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dropout = 0.2
learning_rate = 0.01
epoch_num = 5000
lstm_dim = 256
ln1_output_dim = 128

# sequence_size = 5  # number of utterances in a sequence

gpu_for_testing = 2

config_keywords = {"dropout": "dropout",
                   "lr": "learning_rate",
                   "epoch": "epoch_num",
                   "lstm_dim": "lstm_dim",
                   "ln1_output_dim": "ln1_output_dim",
                   "batch_size": "batch_size",
                   "sequence": "sequence_size",
                   "word_embedding": "word_embedding_source"}

