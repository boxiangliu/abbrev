from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


class LF_SF_Data(Dataset):

    def __init__(self, files_list, vocab, device):
        self.files_list = files_list
        self.data = self.read_files_list(files_list)
        self.convert_PLF_texts_to_tensors()
        self.convert_PLF_labels_to_tensors()
        self.convert_PSF_texts_to_embeddings()

    def __len__(self):
        return len(self.data["PSF"])

    def __getitem__(self, key):
        data = self.data
        return data["PSF_embedding"][key], \
            data["PLF_token_tensor"][key], \
            data["PLF_char_tensor"][key], \
            data["PLF_token_lengths"][key], \
            data["PLF_label_tensor"][key], \
            data["PSF"][key], \
            data["PLF"][key], \
            data["PSF_label"][key], \
            data["PLF_word_labels"][key]

    def __iter__(self):
        data = self.data
        for stuff in zip(data["PSF_embedding"][key],
                         data["PLF_token_tensor"][key],
                         data["PLF_char_tensor"][key],
                         data["PLF_token_lengths"][key],
                         data["PLF_label_tensor"][key],
                         data["PSF"][key],
                         data["PLF"][key],
                         data["PSF_label"][key],
                         data["PLF_word_labels"][key]):
            yield stuff

    @staticmethod
    def read_files_list(files_list):
        data = defaultdict(list)

        for file in files_list:
            with open(file) as f:
                header = self.get_header(f)
                for line in f:
                    parsed_line = self.parse_body_line(line, header)
                    self.add_parsed_line_to_data(parsed_line, data)

        return data

    @staticmethod
    def get_header(f):
        header_line = next(f)
        header = header_line.split("\t")
        return header

    @staticmethod
    def parse_body_line(line, header):
        split_line = line.rstrip("\n").split("\t")
        return dict(zip(header, split_line))

    @staticmethod
    def add_parsed_line_to_data(parsed_line, data):
        for k, v in parse_data.items():
            data[k].append(v)

    def convert_PLF_texts_to_tensors(self):
        data = self.data
        for PLF in data["PLF"]:
            token_tensor, char_tensor, token_lengths = vocab.text_to_tensor(
                PLF)
            data["PLF_token_tensor"].append(token_tensor)
            data["PLF_char_tensor"].append(char_tensor)
            data["PLF_token_lengths"].append(token_lengths)

    def convert_PLF_labels_to_tensors(self):
        data = self.data
        for label in data["PLF_word_labels"]:
            label_tensor = vocab.label_to_tensor(label)
            data["PLF_label_tensor"].append(label_tensor)

    def convert_PSF_texts_to_embeddings(self):
        data = self.data
        for PSF in data["PSF"]:
            tensor = vocab.SF_to_embedding(PSF)
            data["PSF_embedding"].append(tensor)




