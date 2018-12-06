from torch import optim, nn
import sys
from gensim.models import KeyedVectors
import torch
import os
import time
from tensorboardX import SummaryWriter
sys.path.append("/raid/omgempathy/")  # path to the main dir
from src.textbased_model import utils, config
from src.sample import calculateCCC
from src.textbased_model.models import LSTMEmo_duo_sequence
# from importlib import reload
import argparse
from nltk.corpus import stopwords

en_stopwords = None
if not config.use_stopwords:
    en_stopwords = set(stopwords.words('english'))

# whether train on the change differences
train_on_differences = True

parser = argparse.ArgumentParser("Training model")
parser.add_argument("-s", "--seqlen", help="Sequence's length", required=False, default=10, type=int)
parser.add_argument("-g", "--gpu", help="gpu", required=False, default=0, type=int)
parser.add_argument("-opt", "--optimizer", help="optimizer (SGD/Adam)", required=False, default='SGD', type=str)


args = vars(parser.parse_args())
sequence_size = int(args['seqlen'])
gpu = int(args['gpu'])

print("Sequence size: {}".format(sequence_size))

# reload(config)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if torch.cuda.is_available():
    torch.cuda.set_device(gpu)
    print("#GPUs: {}".format(torch.cuda.device_count()))
    print("Current GPU: {}".format(torch.cuda.current_device()))

# from importlib import reload
device = config.device

time_label = "{:.0f}".format(time.time())

training_dir = config.training_data_dir
if train_on_differences:
    training_dir = config.training_data_difference_dir

model_output = config.models_dir
utils.mkdir(model_output)
log_dir = config.log_dir
utils.mkdir(log_dir)
log_writer = open(os.path.join(log_dir, "{}_train_log.txt".format(time_label)), 'w')

output_pred_dir = os.path.join(log_dir, '{}_predicted'.format(time_label))
utils.mkdir(output_pred_dir)

iter_print = config.iter_print

input_dim = config.input_dim
lstm_dim = config.lstm_dim
ln1_dim = config.ln1_output_dim

with open(os.path.join(log_dir, "{}.config".format(time_label)), 'w') as config_writer:
    config_writer.write("dropout={}\n"
                        "learning_rate={}\n"
                        "epoch_num={}\n"
                        "lstm_dim={}\n"
                        "ln1_output_dim={}\n"
                        "batch_size={}\n"
                        "sequence_size={}\n"
                        "word_embedding_source={}\n".format(config.dropout, config.learning_rate, config.epoch_num,
                                                            config.lstm_dim, config.ln1_output_dim, config.batch_size,
                                                            sequence_size, config.glove_word2vec_file))
    if train_on_differences:
        config_writer.write("Train on average's differences\n")

# load data
print("Loading training data: {}".format(training_dir))
start = time.time()
data = utils.load_data_dir(training_dir)
utils.print_time(start)

# data sampling
sampled_data = utils.apply_data_sampling(data, sequence_size)

print("Loading groundtruth sequences...")
start = time.time()
gt_sequences = utils.load_groundtruth_sequences(config.groundtruth_sequences_training_dir)
utils.print_time(start)

# load word embeddings
# word_embeddings_file = config.glove_word2vec_file
word_embeddings_file = config.glove_word2vec_file_filtered
start = time.time()
print("Loading word embeddings...")
word_embeddings = KeyedVectors.load_word2vec_format(word_embeddings_file, binary=False)
vocabs = word_embeddings.wv.vocab
utils.print_time(start)

# convert to tensor
data_tensor = utils.prepare_data(data, vocabs, word_embeddings, en_stopwords)

model = LSTMEmo_duo_sequence(input_dim, lstm_dim, ln1_dim).to(device)
loss_func = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), config.learning_rate)
opt_text = 'SGD'
if args['optimizer'].lower() == "adam":
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    opt_text = 'adam'
log_writer.write("Optimizer: {}\n".format(opt_text.upper()))
print("Optimizer: {}".format(opt_text.upper()))
counter = 0
total_loss = 0
start = time.time()

log_writer.write("#sequences: {}\n".format(utils.get_num_sequences(sampled_data)))
log_writer.write("Use stopwords: {}\n\n".format(config.use_stopwords))
print("#sequences: {}".format(utils.get_num_sequences(sampled_data)))
print("Start training!")

writer = SummaryWriter()
iter_counter = 0
for epoch in range(config.epoch_num):
    file_count = 0
    counter = 0
    total_loss = 0

    for fname, sequences in sampled_data.items():
        tensors = data_tensor[fname]

        # print("File: {}".format(fname))
        for sequence in sequences:
            # print("\tSequence: {}".format(sequence))
            hidden = model.init_hidden()  # initialize hidden (zeros) for a new sequence
            model.zero_grad()  # zero the grad for a new sequence
            seq_loss = 0.0
            iter_counter += 1
            for utter_id in sequence:
                counter += 1

                utter = tensors[utter_id]
                if utter[0] is None:
                    # no valid words
                    continue
                X = utter[0].to(device)
                X = X.view(X.shape[0], 1, -1)
                y = utter[1].to(device)
                pred, hidden = model(X, hidden)

                # compute the loss
                last_output = pred[-1].view(1, -1)
                loss = loss_func(last_output[0], y)
                loss.backward(retain_graph=True)
                total_loss += loss.cpu().item()
                seq_loss += loss.cpu().item()
            writer.add_scalar("mse_loss", seq_loss/len(sequence), iter_counter)
            optimizer.step()  # only update parameters after going through the whole sequence

    # print after each epoch
    print("AvgLoss: {:.4f}. Epoch: {}. Time: {}.".format(total_loss / counter, epoch, utils.time_since(start)))
    log_writer.write("AvgLoss: {:.4f}. Epoch: {}. Time: {}.\n".format(total_loss / counter, epoch, utils.time_since(start)))

    # save model after config.epoch_save
    if epoch > 0 and epoch % config.model_save_epoch_num == 0:
        output_file = os.path.join(model_output, "model_{}_{}.pt".format(time_label, epoch))
        print("Saving model to {}".format(output_file))
        log_writer.write("Saving model to {}\n".format(output_file))
        torch.save(model.state_dict(), output_file)
        log_writer.flush()

# make sure to save the last epoch
output_file = os.path.join(model_output, "model_{}_{}.pt".format(time_label, config.epoch_num-1))
if not os.path.isfile(output_file):
    print("Saving model to {}".format(output_file))
    log_writer.write("Saving model to {}\n".format(output_file))
    torch.save(model.state_dict(), output_file)
print("Sequence size: {}".format(sequence_size))
log_writer.write("Sequence size: {}\n".format(sequence_size))

writer.export_scalars_to_json(os.path.join(log_dir, "{}_tensorboard.json".format(time_label)))
writer.close()
log_writer.flush()
log_writer.close()

