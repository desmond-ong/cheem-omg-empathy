from torch import optim, nn
import sys
from gensim.models import KeyedVectors
import torch
import os
import time
sys.path.append("/raid/omgempathy/")  # path to the main dir
from src.textbased_model import utils, config
from src.sample import calculateCCC
from src.textbased_model.models import LSTMEmo_noSequence
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
                        "word_embedding_source={}\n".format(config.dropout, config.learning_rate, config.epoch_num,
                                                            config.lstm_dim, config.ln1_output_dim, config.batch_size,
                                                            config.glove_word2vec_file))

ccc_writer = open(os.path.join(log_dir, "{}_ccc.txt".format(time_label)), 'w')
ccc_writer.write("@epoch\tvideo\tpearson\tccc\n")

print("Loading training data...")
start = time.time()
data = utils.load_data_dir(training_dir)
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


# test print training data
# for key, value in data.items():
#     print(key)
#     for k, v in value.items():
#         print(k, v)
#     break



# test ccc
# a = gt_sequences['Subject_10_Story_5']
# print(calculateCCC.ccc(a,a))
# utils.write_predicted_sequences(a, 'test.csv')
# b = gt_sequences['Subject_6_Story_4']
# print(len(a))
# la = len(a)
# lb = len(b)
# a[la:lb] = [a[la-1] for _ in range(la, lb)]
# calculateCCC.ccc(a, b)

model = LSTMEmo_noSequence(input_dim, lstm_dim, ln1_dim).to(device)
loss_func = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), config.learning_rate)

data_tensor = utils.prepare_data(data, vocabs, we_model)

counter = 0
total_loss = 0
total_pearson = 0
total_ccc = 0
start = time.time()

batch_loss = 0

for epoch in range(config.epoch_num):
    file_count = 0
    counter = 0
    total_loss = 0
    total_ccc = 0
    total_pearson = 0
    for fname, utters in data_tensor.items():
        file_count += 1
        predicted_sequence = []  # store the predicted sequence
        prev_predicted = 0.0  # in case the "utterance" tensor is None --> use the previous score (to keep continuity)
        tmp_info = data[fname]
        for index, utter in utters.items():
            counter += 1
            tmp = tmp_info[index]
            start_index = tmp[2]  # start frame
            end_index = tmp[3]  # end frame
            if utter[0] is None:
                # in case the "utterance" tensor is None --> use the previous score (to keep continuity)
                utils.add_value_to_sequence(prev_predicted, predicted_sequence, start_index, end_index)
                continue
            X = utter[0].to(device)
            X = X.view(X.shape[0], 1, -1)
            y = utter[1].to(device)
            pred = model(X)
            last_output = pred[-1].view(1, -1)
            loss = loss_func(last_output[0], y)

            # update sequence
            prev_predicted = last_output[0].cpu().item()
            utils.add_value_to_sequence(prev_predicted, predicted_sequence, start_index, end_index)

            total_loss += loss.cpu().item()

            model.zero_grad()
            loss.backward()
            optimizer.step()

        # after finish for 1 file, compute ccc
        gt_sequence = gt_sequences[fname]
        predicted_sequence = utils.refine_predicted_sequence(predicted_sequence, gt_sequence)
        if epoch == config.epoch_num - 1:
            pred_output_file = os.path.join(output_pred_dir, "{}.csv".format(fname))
            utils.write_predicted_sequences(predicted_sequence, pred_output_file)
        ccc, pearson = calculateCCC.ccc(gt_sequence, predicted_sequence)
        ccc_writer.write("{}\t{}\t{}\t{}\n".format(epoch, fname, pearson, ccc))
        ccc_writer.flush()
        total_ccc += ccc
        total_pearson += pearson
    # print after each epoch
    print("AvgLoss: {:.4f}. AvgPearson: {:.4f}. AvgCCC: {:.4f}. Epoch: {}. Time: {}.".format(total_loss / counter,
                                                                                             total_pearson/file_count,
                                                                                             total_ccc/file_count,
                                                                                             epoch,
                                                                                             utils.time_since(start)))
    # save model after config.epoch_save
    if epoch % config.model_save_epoch_num == 0:
        output_file = os.path.join(model_output, "model_{}_{}.pt".format(time_label, epoch))
        print("Saving model to {}".format(output_file))
        torch.save(model.state_dict(), output_file)


# make sure to save the last epoch
output_file = os.path.join(model_output, "model_{}_{}.pt".format(time_label, config.epoch_num-1))
if not os.path.isfile(output_file):
    print("Saving model to {}".format(output_file))
    torch.save(model.state_dict(), output_file)

ccc_writer.flush()
ccc_writer.close()

