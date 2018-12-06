'''
utter_chunk

One video is splitted into chunks (sequences of "utterances")
'''

from torch import optim, nn
from gensim.models import KeyedVectors
import torch
import os
import time
from tensorboardX import SummaryWriter
import argparse
from nltk.corpus import stopwords
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import sys
sys.path.append("/raid/omgempathy/")  # path to the main dir
from src.text_based_submit import utils


class LSTMUtterChunk(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, ln1_out_dim, dropout=0.2, batch_size=1, device="cpu"):
        super(LSTMUtterChunk, self).__init__()
        self.hidden_dim = lstm_hidden_dim
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim)
        self.dropout = nn.Dropout2d(dropout)
        self.linear1 = nn.Linear(lstm_hidden_dim, ln1_out_dim)
        self.linear2 = nn.Linear(ln1_out_dim, 1)
        self.device = device

    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_dim).to(self.device),
                torch.zeros(1, self.batch_size, self.hidden_dim).to(self.device))

    def forward(self, utterances, use_dropout=True):
        # input is one utterance
        hidden = self.init_hidden()
        lstm_output, hidden = self.lstm(utterances, hidden)
        dropout_output = lstm_output
        if use_dropout:
            dropout_output = self.dropout(lstm_output)
        ln1_output = self.linear1(dropout_output)
        ln1_output_relu = F.relu(ln1_output)
        ln2_output = self.linear2(ln1_output_relu)
        final_output = torch.tanh(ln2_output)
        return final_output


class UtterancesChunk(Dataset):
    def __init__(self, data_tensor, chunk_size, chunk_step=-1):
        '''
        :param data_tensor: {fname: (features, values, valid_indices)}
        '''
        self.chunks = []  # [(Xs, ys)]
        self.chunk_size = self.validate_chunk_size(data_tensor, chunk_size)  # if chunk size is too big, get the minimum sequence size as chunk size
        self.chunk_step = self.chunk_size
        if self.chunk_size > chunk_step > 0:
            self.chunk_step = chunk_step
        self.get_chunks(data_tensor)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, item):
        return self.chunks[item]

    def validate_chunk_size(self, data_tensor, chunk_size):
        tmp = chunk_size
        for fname, utterances in data_tensor.items():
            if len(utterances[0]) < tmp:
                tmp = len(utterances[0])
        return tmp

    def get_chunks(self, data_tensor):
        for fname, utterances in data_tensor.items():
            features = utterances[0]
            gt = utterances[1]
            num_utt = len(features)
            # count = 0
            ind = 0
            while ind < num_utt:
                start_ind = ind
                end_ind = ind + self.chunk_size
                if end_ind > num_utt:
                    # the last chunk, "borrow" previous utterance if needed
                    end_ind = num_utt
                    start_ind = num_utt - self.chunk_size
                # count += 1
                self.chunks.append((features[start_ind:end_ind], gt[start_ind:end_ind]))
                ind = ind + self.chunk_step


def train():
    # initialize lstm model
    model = LSTMUtterChunk(input_dim, lstm_dim, ln1_dim, dropout=dropout, batch_size=batch_size, device=device).to(device)
    loss_func = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    if args['optimizer'].lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    log_writer.write("Optimizer: {}\n".format(args['optimizer'].upper()))
    print("Optimizer: {}".format(args['optimizer'].upper()))

    start = time.time()

    log_writer.write("Use stopwords: {}\n\n".format(en_stopwords is None))
    print("Start training!")

    writer = SummaryWriter()
    iter_counter = 0
    best_model_output = os.path.join(model_output, "model_{}_best.pt".format(time_label))
    best_model_epoch = -1
    best_model_avg_loss = -1

    for epoch in range(epoch_num):
        counter = 0
        total_loss = 0

        for data_inst in dataloader:
            X = data_inst[0].to(device)
            y = data_inst[1].to(device)

            if X.shape[0] != batch_size:
                continue
            X = X.view(X.shape[1], batch_size, -1)
            iter_counter += 1
            counter += 1
            model.zero_grad()  # zero the grad for a new sequence

            pred = model(X)
            pred = pred.view(batch_size, -1, 1)
            loss = loss_func(pred, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.cpu().item()
            writer.add_scalar("chunk_level", loss.cpu().item(), iter_counter)

        # check if best model
        epoch_avg_loss = total_loss / counter
        writer.add_scalar("epoch_level", epoch_avg_loss, epoch)

        if best_model_avg_loss < 0 or epoch_avg_loss < best_model_avg_loss:
            # save best model
            best_model_avg_loss = epoch_avg_loss
            best_model_epoch = epoch
            # print("\tSaving BEST model to {}".format(best_model_output))
            log_writer.write("\tSaving BEST model to {}\n".format(best_model_output))
            torch.save(model.state_dict(), best_model_output)
            log_writer.flush()

        if epoch > 0 and epoch % iter_print == 0:
            print("AvgLoss: {:.4f}. Epoch: {}. Time: {}.".format(epoch_avg_loss, epoch, utils.time_since(start)))
            log_writer.write(
                "AvgLoss: {:.4f}. Epoch: {}. Time: {}.\n".format(epoch_avg_loss, epoch, utils.time_since(start)))

        # save model after config.epoch_save
        if epoch > 0 and epoch % num_model_save == 0:
            output_file = os.path.join(model_output, "model_{}_{}.pt".format(time_label, epoch))
            print("Saving model to {}".format(output_file))
            log_writer.write("Saving model to {}\n".format(output_file))
            torch.save(model.state_dict(), output_file)
            log_writer.flush()

    # make sure to save the last epoch
    output_file = os.path.join(model_output, "model_{}_{}.pt".format(time_label, epoch_num - 1))
    if not os.path.isfile(output_file):
        print("Saving model to {}".format(output_file))
        log_writer.write("Saving model to {}\n".format(output_file))
        torch.save(model.state_dict(), output_file)

    writer.export_scalars_to_json(os.path.join(log_dir, "{}_tensorboard.json".format(time_label)))
    writer.close()

    print("Best model epoch: {}".format(best_model_epoch))
    log_writer.write("Best model epoch: {}\n".format(best_model_epoch))


if __name__ == "__main__":
    en_stopwords = set(stopwords.words('english'))
    # whether train on the change differences

    parser = argparse.ArgumentParser("Model Training")
    parser.add_argument("--gpu", help="gpu", required=False, default=0, type=int)
    parser.add_argument("--chunk", help="Chunk size: number of utterances in a chunk", default=10, type=int)
    parser.add_argument("--step", help="Chunk step: number of utterances to jump for the next chunk", default=5,
                        type=int)
    parser.add_argument("--optimizer", help="optimizer (SGD/Adam)", required=False, choices=["sgd", "adam"],
                        default='sgd', type=str)
    parser.add_argument("--train", help="Training data dir's path", required=False,
                        default='data/Training/transcripts_with_scores_difference/', type=str)
    parser.add_argument("--output", help="Model's output dir", required=False, default='data/Training/models/',
                        type=str)
    parser.add_argument("--logdir", help="Log dir", required=False, default='data/Training/training_logs/', type=str)
    parser.add_argument("--input_dim", help="Input dimension", default=300, type=int)
    parser.add_argument("--lstm_dim", help="LSTM hidden's dimension", default=256, type=int)
    parser.add_argument("--linear1_dim", help="FC 1's output dimension", default=128, type=int)
    parser.add_argument("--batch", help="Batch size", default=20, type=int)
    parser.add_argument("--embeddings", help="Word embeddings' file", default='data/glove.840B.300d.filtered.word2vec',
                        type=str)
    parser.add_argument("--dropout", help="Dropout", default=0.2, type=float)
    parser.add_argument("--lr", help="learning rate", default=0.01, type=float)
    parser.add_argument("--epoch", help="Number of training epochs", default=5000, type=int)
    parser.add_argument("--iter_print", help="Number of iteration before printing", default=100, type=int)
    parser.add_argument("--shuffle", help='Shuffle training instances?', default=True, type=bool)
    parser.add_argument("--iter_model_save", help='Save the model after every n iterations', default=2000, type=int)
    parser.add_argument("--groundtruth", help="Groundtruth for training data", default="data/Training/Annotations/")

    args = vars(parser.parse_args())

    gpu = int(args['gpu'])

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        print("#GPUs: {}".format(torch.cuda.device_count()))
        print("Current GPU: {}".format(torch.cuda.current_device()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_dir = args['train']
    model_output = args['output']
    utils.mkdir(model_output)
    log_dir = args['logdir']
    utils.mkdir(log_dir)

    time_label = "{:.0f}".format(time.time())

    log_writer = open(os.path.join(log_dir, "{}_train_log.txt".format(time_label)), 'w')

    chunk_size = int(args['chunk'])
    chunk_step = int(args['step'])
    iter_print = int(args['iter_print'])
    input_dim = int(args['input_dim'])
    lstm_dim = int(args['lstm_dim'])
    ln1_dim = int(args['linear1_dim'])
    batch_size = int(args['batch'])
    dropout = float(args['dropout'])
    lr = float(args['lr'])
    epoch_num = int(args['epoch'])
    num_model_save = int(args['iter_model_save'])

    # load word embeddings
    word_embeddings_file = args['embeddings']
    start = time.time()
    print("Loading word embeddings...")
    word_embeddings = KeyedVectors.load_word2vec_format(word_embeddings_file, binary=False)
    vocabs = word_embeddings.wv.vocab
    # utils.print_time(start)

    # write config file
    with open(os.path.join(log_dir, "{}.config".format(time_label)), 'w') as config_writer:
        config_writer.write("dropout={}\n"
                            "learning_rate={}\n"
                            "epoch_num={}\n"
                            "lstm_dim={}\n"
                            "ln1_output_dim={}\n"
                            "batch_size={}\n"
                            "chunk_size={}\n"
                            "chunk_step={}\n"
                            "train_dir={}\n"
                            "word_embedding_source={}\n".format(dropout, lr, epoch_num,
                                                                lstm_dim, ln1_dim, batch_size,
                                                                chunk_size, chunk_step, training_dir,
                                                                word_embeddings_file))
        config_writer.write("Use stopwords: {}\n".format(en_stopwords is None))

    # load data
    print("Loading training data: {}".format(training_dir))
    data = utils.load_data_dir(training_dir)  # {filename: OrderDict {utter_id: (text, score, startframe, endframe)}}
    # utils.print_time(start)

    print("Loading groundtruth sequences...")
    gt_sequences = utils.load_groundtruth_sequences(args['groundtruth'])
    # utils.print_time(start)

    # convert to tensor
    print("Compute avg word embeddings and convert to tensor")
    data_tensor = utils.get_average_word_embeddings_for_utters(data, vocabs, word_embeddings,
                                                               en_stopwords)  # {fname: [list of utterances' embeddings (i.e., avg words' embeddings)]}
    # utils.print_time(start)

    chunks = UtterancesChunk(data_tensor, chunk_size, chunk_step=chunk_step)
    dataloader = DataLoader(chunks, batch_size=batch_size, shuffle=True)
    print("Chunk size: {}".format(chunk_size))
    print("Chhunk step (stride): {}".format(chunk_step))
    print("Number of chunks: {}".format(len(chunks)))
    log_writer.write("Chunk size: {}\n".format(chunk_size))
    log_writer.write("Number of chunks: {}\n".format(len(chunks)))

    train()

    log_writer.flush()
    log_writer.close()




