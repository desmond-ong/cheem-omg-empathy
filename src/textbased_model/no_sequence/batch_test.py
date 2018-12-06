from torch import optim, nn
import sys
from gensim.models import KeyedVectors
import torch
import time
from torch.utils.data import DataLoader
sys.path.append("/raid/omgempathy/")  # path to the main dir
from src.textbased_model import utils, config
from src.textbased_model.models import LSTMEmo_noSequence
from src.textbased_model.datasets import Utterances
from src.textbased_model.no_sequence.padding import PadCollate

# from importlib import reload

# reload(config)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if torch.cuda.is_available():
    torch.cuda.set_device(2)
    print("#GPUs: {}".format(torch.cuda.device_count()))
    print("Current GPU: {}".format(torch.cuda.current_device()))

# from importlib import reload
device = config.device

time_label = "{:.0f}".format(time.time())

training_dir = config.training_data_dir
model_output = config.models_dir
utils.mkdir(model_output)
log_dir = config.log_dir
utils.mkdir(log_dir)

iter_print = config.iter_print

input_dim = config.input_dim
lstm_dim = config.lstm_dim
ln1_dim = config.ln1_output_dim

# with open(os.path.join(log_dir, "{}.config".format(time_label)), 'w') as config_writer:
#     config_writer.write("dropout={}\n"
#                         "learning_rate={}\n"
#                         "epoch_num={}\n"
#                         "lstm_dim={}\n"
#                         "ln1_output_dim={}\n"
#                         "batch_size={}\n"
#                         "word_embedding_source={}\n".format(config.dropout, config.learning_rate, config.epoch_num,
#                                                             config.lstm_dim, config.ln1_output_dim, config.batch_size,
#                                                             config.glove_word2vec_file))

print("Loading training data...")
start = time.time()
data = utils.load_data_dir(training_dir)  # (text, valence, start frame, end frame, video, utter id)
utils.print_time(start)

print("Loading groundtruth sequences...")
start = time.time()
gt_sequences = utils.load_groundtruth_sequences(config.groundtruth_sequences_training_dir)
utils.print_time(start)

word_embeddings_file = config.glove_word2vec_file
start = time.time()
print("Loading word embeddings...")
we_model = KeyedVectors.load_word2vec_format(word_embeddings_file, binary=False)
vocabs = we_model.wv.vocab
utils.print_time(start)


model = LSTMEmo_noSequence(input_dim, lstm_dim, ln1_dim, batch_size=config.batch_size).to(device)
loss_func = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), config.learning_rate)

data_tensor = utils.prepare_data(data, vocabs, we_model)  # {file: {OrderedDict index:(x, y)} }

utterances = Utterances(data_tensor)
dataloader = DataLoader(utterances, collate_fn=PadCollate(), batch_size=config.batch_size, shuffle=False)

counter = 0
total_loss = 0
start = time.time()
#
# for epoch in range(config.epoch_num):
#     file_count = 0
#     counter = 0
#     total_loss = 0
#     for X, y, filenames, indices in dataloader:
#         counter += 1
#         # print(X.shape)
#         # print(y.shape)
#         # print(filenames)
#         # print(indices)
#         X = X.to(device)
#         X = X.view(X.shape[1], config.batch_size, -1)
#         # print("----------")
#         # print(X)
#         y = y.to(device)
#         pred = model(X)
#         last_output = pred[-1]
#
#         # print("pred shape:", pred.shape)
#         # print(pred)
#         # print("------")
#         # print(last_output.shape)
#         # print(last_output)
#         # print("------")
#         # print(y.shape)
#         # print(y)
#
#         loss = loss_func(last_output, y)
#         total_loss += loss.cpu().item()
#         model.zero_grad()
#         loss.backward()
#         optimizer.step()
#     # # print after each epoch
#     print("AvgLoss: {:.4f}. Epoch: {}. Time: {}.".format(total_loss / counter, epoch, utils.time_since(start)))
#
#
# output_file = os.path.join(model_output, "model_{}_{}.pt".format(time_label, config.epoch_num-1))
# if not os.path.isfile(output_file):
#     print("Saving model to {}".format(output_file))
#     torch.save(model.state_dict(), output_file)
