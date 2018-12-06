'''
link: https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/8
author: Felix_Kreuk
'''
import torch

def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label, filename, index)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence

        # print("Printing out instances in a batch")
        # for x in batch:
        #     print(x[0].shape)
        #     print("{}\t{}\t{}\t{}".format(x[1], x[2], x[3], x[4]))

        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        # print("MAX LEN: ", max_len)
        # pad according to max_len
        # batch = map(lambda (x, y):
        #             (pad_tensor(x, pad=max_len, dim=self.dim), y), batch)
        # padded_Xs = map(lambda p: pad_tensor(p[0], pad=max_len, dim=self.dim), batch)  # modified sn

        padded_Xs = [pad_tensor(p[0], pad=max_len, dim=self.dim) for p in batch]
        # print(padded_Xs)
        # stack all
        # xs = torch.stack(map(lambda x: x[0], batch), dim=0)
        xs = torch.stack([x for x in padded_Xs], dim=0)
        ys = [x[1] for x in batch]
        ys = torch.stack(ys, dim=0)

        files = [x[2] for x in batch]
        indices = [x[3] for x in batch]
        # print("Ys: ", ys)
        # print(xs)

        return xs, ys, files, indices

    def __call__(self, batch):
        return self.pad_collate(batch)


#to be used with the data loader:
#train_loader = DataLoader(ds, ..., collate_fn=PadCollate(dim=0))