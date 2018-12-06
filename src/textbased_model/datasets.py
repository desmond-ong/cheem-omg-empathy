

from torch.utils.data import Dataset

class Utterances(Dataset):
    def __init__(self, data_tensor):
        '''
        :param data_tensor: # {file: {OrderedDict index:(x, y)} }
        '''
        self.utterances = []  # stores all the utterances (don't care about the video source)
        self.load_data(data_tensor)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, item):
        utter = self.utterances[item]
        return utter[0], utter[1], utter[2], utter[3]  # x, y, filename, index

    def load_data(self, data_tensor):
        for fname, data in data_tensor.items():
            for ind, utter in data.items():
                if utter[0] is not None:
                    self.utterances.append((utter[0], utter[1], fname, ind))


class Utterances_Chunk(Dataset):
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
        # get chunks from input (utterances from all videos)
        # tmp = {}

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
            # tmp[fname] = count

        # for key, value in tmp.items():
        #     print("{}\t{}".format(key, value))


class Utterances_Chunk_Eval():
    def __init__(self, data_tensor, chunk_size, chunk_step=-1):
        '''

        :param data_tensor: {fname: (features, values, valid_indices)}
        '''
        self.chunks = {}  # {filename: [(Xs, ys)]}
        self.chunk_size = self.validate_chunk_size(data_tensor, chunk_size)  # if chunk size is too big, get the minimum sequence size as chunk size
        self.chunk_step = self.chunk_size
        if self.chunk_size > chunk_step > 0:
            self.chunk_step = chunk_step
        self.get_chunks(data_tensor)

    def len(self):
        return len(self.chunks)

    def validate_chunk_size(self, data_tensor, chunk_size):
        tmp = chunk_size
        for fname, utterances in data_tensor.items():
            if len(utterances[0]) < tmp:
                tmp = len(utterances[0])
        return tmp

    def get_chunks(self, data_tensor):
        # get chunks from input (utterances from all videos)
        # tmp = {}
        for fname, utterances in data_tensor.items():
            tmp = []
            features = utterances[0]
            gt = utterances[1]
            valid_indices = utterances[2]
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
                tmp.append((features[start_ind:end_ind], gt[start_ind:end_ind], valid_indices[start_ind: end_ind]))
                ind = ind + self.chunk_step
            self.chunks[fname] = tmp

