import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class EmbedRNN(nn.Module):
    """Recurrent neural network"""

    def __init__(self, input_size, hidden_size, output_size, embed_size=16):
        """Args:
                input_size (int): input dimension of a time step.
                hidden_size (int): dimesion of hidden layer.
                output_size (int): number of output categories.  
        """
        super(EmbedRNN, self).__init__()

        print(f"input_size:{input_size}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embed_size = embed_size

        self.dropout = nn.Dropout(p=0.5)
        self.embed = nn.Embedding(input_size, embed_size)
        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size)
        self.fc1 = nn.Linear(in_features=hidden_size,
                             out_features=output_size)
        self.fc2 = nn.Linear(in_features=2 * hidden_size,
                             out_features=output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.pack_padded_sequence = pack_padded_sequence

    def forward(self, sfs, lfs, sf_lens, lf_lens):
        """Args:
                seqs (PackedSequence): Packed padded sequence.
        """
        sf_embedding = self.embed(sfs)
        lf_embedding = self.embed(lfs)
        sfs = self.pack_padded_sequence(
            sf_embedding, sf_lens.cpu(), enforce_sorted=False)
        lfs = self.pack_padded_sequence(
            lf_embedding, lf_lens.cpu(), enforce_sorted=False)

        sf_output, (sf_hidden, sf_cell) = self.rnn(sfs)
        lf_output, (lf_hidden, lf_cell) = self.rnn(lfs)

        sf_hidden = self.dropout(sf_hidden)
        lf_hidden = self.dropout(lf_hidden)

        sf_output = self.fc1(sf_hidden[0])
        # hidden[0] -> remove zeroth dimension
        pair_output = torch.cat([sf_hidden[0], lf_hidden[0]], dim=1)
        pair_output = self.fc2(pair_output)

        return self.softmax(sf_output), self.softmax(pair_output)


class EmbedRNNSequenceAvg(nn.Module):
    """Recurrent neural network"""

    def __init__(self, input_size, hidden_size, output_size, max_length, device=torch.device("cpu")):
        """Args:
                input_size (int): input dimension of a time step.
                hidden_size (int): dimesion of hidden layer.
                output_size (int): number of output categories.
                max_length (int): maximum length of the LF
        """
        super(EmbedRNNSequenceAvg, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        self.dropout = nn.Dropout(p=0.5)
        self.embed = nn.Embedding(input_size, self.hidden_size)
        self.rnn_sf = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.rnn_lf = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc = nn.Linear(in_features=hidden_size,
                            out_features=output_size)
        self.pack_padded_sequence = pack_padded_sequence
        self.device = device

    def forward(self, sfs, lfs, sf_lens, lf_lens):
        """Args:
                seqs (PackedSequence): Packed padded sequence.
        """
        sf_embeddings = self.embed(sfs)
        lf_embeddings = self.embed(lfs)
        lf_embeddings = pack_padded_sequence(
            lf_embeddings, lf_lens.cpu(), enforce_sorted=False)
        lf_output, lf_hidden = self.rnn_lf(lf_embeddings)
        lf_output, _ = pad_packed_sequence(
            lf_output, total_length=self.max_length)
        batch_size = sfs.size()[1]

        hidden = self.initHidden(batch_size)
        attn_weights_list = []
        outputs = []
        for sf_embedding in sf_embeddings:
            attn_weights = F.softmax(
                self.attn(torch.cat([sf_embedding, hidden[0]], dim=1)), dim=1)
            attn_weights_list.append(attn_weights.unsqueeze(0))
            attn_applied = torch.bmm(
                attn_weights.unsqueeze(1), lf_output.transpose(0, 1))
            output = torch.cat([sf_embedding, attn_applied.squeeze(1)], dim=1)
            output = self.attn_combine(output).unsqueeze(0)
            output = F.relu(output)
            output, hidden = self.rnn_sf(output, hidden)
            outputs.append(output)

        output = torch.cat(outputs, dim=0)
        output = torch.mean(output, dim=0)
        prob = F.log_softmax(self.fc(output), dim=1)

        return prob, torch.cat(attn_weights_list, dim=0)

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)



# Testing the ToyEmbedRNNSequence model.
def testing_EmbedRNNSequenceAvg():
    import sys
    sys.path.insert(0, "./model/qa_reject/")
    from data import ToyData, WrappedDataLoader
    from torch.utils.data import DataLoader
    from pathlib import Path
    data_dir = Path(
        "../processed_data/model/qa_reject/QA_output_to_LSTM_input/")
    eval_sets = ["medstract"]
    sf_eval = SFLFData([data_dir / x for x in eval_sets], one_hot=False)
    weighted_sampler = sf_eval.get_weighted_sampler()

    batch_size = 4
    train_loader = DataLoader(
        sf_eval, batch_size=batch_size, sampler=weighted_sampler, collate_fn=sf_eval._pad_seq)
    input_size = sf_eval.n_characters
    hidden_size = 512
    output_size = 2
    max_length = max([len(lf) for lf in sf_eval.data["lf"]]) # 10 char per word * two words + space

    batch = next(iter(train_loader))
    model = EmbedRNNSequenceAvg(
        input_size, hidden_size, output_size, max_length)
    sfs = batch[0]
    lfs = batch[1]
    sf_label = batch[2]
    pair_label = batch[3]
    is_gold = batch[4]
    sf_lens = batch[5]
    lf_lens = batch[6]
    sf = batch[7]
    lf = batch[8]

    prob, attn = model(sfs, lfs, sf_lens, lf_lens)


class ToyEmbedRNN(nn.Module):
    """Recurrent neural network"""

    def __init__(self, input_size, hidden_size, output_size, embed_size=16):
        """Args:
                input_size (int): input dimension of a time step.
                hidden_size (int): dimesion of hidden layer.
                output_size (int): number of output categories.  
        """
        super(ToyEmbedRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embed_size = embed_size

        self.dropout = nn.Dropout(p=0.5)
        self.embed = nn.Embedding(input_size, embed_size)
        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size)
        self.fc = nn.Linear(in_features=2 * hidden_size,
                            out_features=output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.pack_padded_sequence = pack_padded_sequence

    def forward(self, sfs, lfs, sf_lens, lf_lens):
        """Args:
                seqs (PackedSequence): Packed padded sequence.
        """
        sf_embedding = self.embed(sfs)
        lf_embedding = self.embed(lfs)
        sfs = self.pack_padded_sequence(
            sf_embedding, sf_lens.cpu(), enforce_sorted=False)
        lfs = self.pack_padded_sequence(
            lf_embedding, lf_lens.cpu(), enforce_sorted=False)

        sf_output, (sf_hidden, sf_cell) = self.rnn(sfs)
        lf_output, (lf_hidden, lf_cell) = self.rnn(lfs)

        sf_hidden = self.dropout(sf_hidden)
        lf_hidden = self.dropout(lf_hidden)

        pair_output = torch.cat([sf_hidden[0], lf_hidden[0]], dim=1)
        pair_output = self.fc(pair_output)

        return self.softmax(pair_output), 0


class ToyEmbedRNNSequence(nn.Module):
    """Recurrent neural network"""

    def __init__(self, input_size, hidden_size, output_size, max_length, device=torch.device("cpu")):
        """Args:
                input_size (int): input dimension of a time step.
                hidden_size (int): dimesion of hidden layer.
                output_size (int): number of output categories.  
        """
        super(ToyEmbedRNNSequence, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        self.dropout = nn.Dropout(p=0.5)
        self.embed = nn.Embedding(input_size, self.hidden_size)
        self.rnn_sf = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.rnn_lf = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc = nn.Linear(in_features=hidden_size,
                            out_features=output_size)
        self.pack_padded_sequence = pack_padded_sequence
        self.device = device

    def forward(self, sfs, lfs, sf_lens, lf_lens):
        """Args:
                seqs (PackedSequence): Packed padded sequence.
        """
        sf_embeddings = self.embed(sfs)
        lf_embeddings = self.embed(lfs)
        lf_embeddings = pack_padded_sequence(
            lf_embeddings, lf_lens.cpu(), enforce_sorted=False)
        lf_output, lf_hidden = self.rnn_lf(lf_embeddings)
        lf_output, _ = pad_packed_sequence(
            lf_output, total_length=self.max_length)
        batch_size = sfs.size()[1]

        hidden = self.initHidden(batch_size)
        attn_weights_list = []
        for sf_embedding in sf_embeddings:
            attn_weights = F.softmax(
                self.attn(torch.cat([sf_embedding, hidden[0]], dim=1)), dim=1)
            attn_weights_list.append(attn_weights.unsqueeze(0))
            attn_applied = torch.bmm(
                attn_weights.unsqueeze(1), lf_output.transpose(0, 1))
            output = torch.cat([sf_embedding, attn_applied.squeeze(1)], dim=1)
            output = self.attn_combine(output).unsqueeze(0)
            output = F.relu(output)
            output, hidden = self.rnn_sf(output, hidden)

        prob = F.log_softmax(self.fc(output[0]), dim=1)

        return prob, torch.cat(attn_weights_list, dim=0)

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)


# Testing the ToyEmbedRNNSequence model.
def testing_ToyEmbedRNNSequence():
    import sys
    sys.path.insert(0, "./model/qa_reject/")
    from data import ToyData, WrappedDataLoader
    from torch.utils.data import DataLoader

    batch_size = 4
    train_data = ToyData('../processed_data/model/qa_reject/toy_data/train')
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=train_data._pad_seq)
    input_size = train_data.n_characters
    hidden_size = 512
    output_size = 2
    max_length = 21  # 10 char per word * two words + space
    batch = next(iter(train_loader))
    model = ToyEmbedRNNSequence(
        input_size, hidden_size, output_size, max_length)
    sfs = batch[0]
    lfs = batch[1]
    sf_lens = batch[3]
    lf_lens = batch[4]
    prob, attn = model(sfs, lfs, sf_lens, lf_lens)


class ToyEmbedRNNSequenceAvg(nn.Module):
    """Recurrent neural network"""

    def __init__(self, input_size, hidden_size, output_size, max_length, device=torch.device("cpu")):
        """Args:
                input_size (int): input dimension of a time step.
                hidden_size (int): dimesion of hidden layer.
                output_size (int): number of output categories.  
        """
        super(ToyEmbedRNNSequenceAvg, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        self.dropout = nn.Dropout(p=0.5)
        self.embed = nn.Embedding(input_size, self.hidden_size)
        self.rnn_sf = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.rnn_lf = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc = nn.Linear(in_features=hidden_size,
                            out_features=output_size)
        self.pack_padded_sequence = pack_padded_sequence
        self.device = device

    def forward(self, sfs, lfs, sf_lens, lf_lens):
        """Args:
                seqs (PackedSequence): Packed padded sequence.
        """
        sf_embeddings = self.embed(sfs)
        lf_embeddings = self.embed(lfs)
        lf_embeddings = pack_padded_sequence(
            lf_embeddings, lf_lens.cpu(), enforce_sorted=False)
        lf_output, lf_hidden = self.rnn_lf(lf_embeddings)
        lf_output, _ = pad_packed_sequence(
            lf_output, total_length=self.max_length)
        batch_size = sfs.size()[1]

        hidden = self.initHidden(batch_size)
        attn_weights_list = []
        outputs = []
        for sf_embedding in sf_embeddings:
            attn_weights = F.softmax(
                self.attn(torch.cat([sf_embedding, hidden[0]], dim=1)), dim=1)
            attn_weights_list.append(attn_weights.unsqueeze(0))
            attn_applied = torch.bmm(
                attn_weights.unsqueeze(1), lf_output.transpose(0, 1))
            output = torch.cat([sf_embedding, attn_applied.squeeze(1)], dim=1)
            output = self.attn_combine(output).unsqueeze(0)
            output = F.relu(output)
            output, hidden = self.rnn_sf(output, hidden)
            outputs.append(output)

        output = torch.cat(outputs, dim=0)
        output = torch.mean(output, dim=0)
        prob = F.log_softmax(self.fc(output), dim=1)

        return prob, torch.cat(attn_weights_list, dim=0)

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)


# Testing the ToyEmbedRNNSequence model.
def testing_ToyEmbedRNNSequenceAvg():
    import sys
    sys.path.insert(0, "./model/qa_reject/")
    from data import ToyData, WrappedDataLoader
    from torch.utils.data import DataLoader

    batch_size = 4
    train_data = ToyData('../processed_data/model/qa_reject/toy_data/train')
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=train_data._pad_seq)
    input_size = train_data.n_characters
    hidden_size = 512
    output_size = 2
    max_length = 21  # 10 char per word * two words + space
    batch = next(iter(train_loader))
    model = ToyEmbedRNNSequenceAvg(
        input_size, hidden_size, output_size, max_length)
    sfs = batch[0]
    lfs = batch[1]
    sf_lens = batch[3]
    lf_lens = batch[4]
    prob, attn = model(sfs, lfs, sf_lens, lf_lens)
