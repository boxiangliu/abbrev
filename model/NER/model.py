import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class NER(nn.Module):
    """Recurrent neural network"""

    def __init__(self, input_size, hidden_size, output_size, max_sf_length, device=torch.device("cpu")):
        """Args:
                input_size (int): input dimension of a time step.
                hidden_size (int): dimesion of hidden layer.
                output_size (int): number of output categories.
                max_sf_length (int): maximum length of the LF
        """
        super(NER, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_sf_length = max_sf_length

        self.input_dropout = nn.Dropout(p=0.1)
        self.hidden_dropout = nn.Dropout(p=0.5)

        self.embed = nn.Embedding(input_size, self.hidden_size)
        self.rnn_sf = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.rnn_lf = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)

        self.attn = nn.Linear(self.hidden_size * 2, self.max_sf_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.sf_fc = nn.Linear(in_features=hidden_size,
                               out_features=output_size)

        self.device = device

    def forward(self, sfs, lfs, sf_lens, lf_lens):
        """Args:
                seqs (PackedSequence): Packed padded sequence.
        """
        sf_embeddings = self.embed(sfs)
        lf_embeddings = self.embed(lfs)

        sf_embeddings = pack_padded_sequence(
            sf_embeddings, sf_lens.cpu(), enforce_sorted=False)
        sf_output, sf_hidden = self.rnn_sf(sf_embeddings)
        sf_output, _ = pad_packed_sequence(
            sf_output, total_length=self.max_sf_length)
        batch_size = sfs.size()[1]

        hidden = self.initHidden(batch_size)
        attn_weights_list = []
        outputs = []
        for lf_embedding in lf_embeddings:
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


class SF_Encoder(nn.Module):
    """Short form encoder network"""

    def __init__(self, input_size, hidden_size, output_size, max_sf_length):
        """Args:
                input_size (int): input dimension of a time step.
                hidden_size (int): dimesion of hidden layer.
                output_size (int): number of output categories.
                max_sf_length (int): maximum length of the LF
        """
        super(SF_Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_sf_length = max_sf_length

        self.input_dropout = nn.Dropout(p=0.1)
        self.hidden_dropout = nn.Dropout(p=0.5)

        self.embed = nn.Embedding(input_size, self.hidden_size)
        self.rnn = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)

        self.fc = nn.Linear(in_features=hidden_size,
                            out_features=output_size)

    def forward(self, sfs, sf_lens):
        """Args:
                sfs (PaddedSequence): padded short form sequence.
                sf_lens (tensor): short form sequence lengths.
        """
        sf_embeddings = self.embed(sfs)
        sf_embeddings = self.input_dropout(sf_embeddings)
        sf_embeddings = pack_padded_sequence(
            sf_embeddings, sf_lens.cpu(), enforce_sorted=False)

        sf_output, sf_hidden = self.rnn(sf_embeddings)
        sf_output, _ = pad_packed_sequence(
            sf_output, total_length=self.max_sf_length)
        sf_output = self.hidden_dropout(sf_output)

        sf_mean = torch.mean(sf_output, dim=0)
        prob = F.log_softmax(self.fc(sf_mean), dim=1)

        return prob, sf_output


def test_SF_Encoder():
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
    import torch.nn.functional as F
    import sys
    sfs = [torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2, 3])]
    sf_lens = torch.tensor([len(x) for x in sfs])
    max_sf_length = max(sf_lens)
    sfs = pad_sequence(sfs)
    encoder = SF_Encoder(input_size=4, hidden_size=1, output_size=2, max_sf_length=max_sf_length)
    prob, sf_output = encoder(sfs, sf_lens)
    assert prob.size() == torch.Size([2,2])
    assert sf_output.size() == torch.Size([4,2,1])


class LF_NER_Decoder(nn.Module):
    """Long form NER decoder"""

    def __init__(self, input_size, hidden_size, output_size):
    """Args:
            input_size (int): input dimension of a time step.
            hidden_size (int): dimesion of hidden layer.
            output_size (int): number of output categories.
    """
        super(LF_NER_Decoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input_dropout = nn.Dropout(p=0.1)
        self.hidden_dropout = nn.Dropout(p=0.5)

        self.embed = nn.Embedding(input_size, self.hidden_size)
        self.rnn = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)

        self.attn = nn.Linear(self.hidden_size * 2, self.max_sf_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.fc = nn.Linear(in_features=hidden_size,
                               out_features=output_size)

        self.device = device



    def forward(self, lfs, lf_lens, hidden, sf_output):
        """Args:
                lfs (PaddedSequence): padded long form sequence.
                lf_lens (tensor): long form lengths.
                hidden (tensor): hidden unit from timestep t - 1.
                sf_output: output from SF_Encoder.
        """
        lf_embeddings = self.embed(lfs)
        lf_embeddings = self.input_dropout(lf_embeddings)


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
