import json
import random
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from statistics import mean


class CustomDataset(Dataset):
    def __init__(self, df):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'input_ids': self.data.iloc[idx]['input_ids'].flatten(),
                'attention_mask': self.data.iloc[idx]['attention_mask'].flatten(),
                # 'text': self.data.iloc[idx]['text'],
                }


class BenchmarkDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.questions, self.answers = data_reader(args)
        self.len = len(self.questions)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        input = self.questions[index].strip()
        output = self.answers[index].strip()
        return input, output


def data_reader(args):

    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if args.dataset == "math500":
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["problem"]
                a = line["answer"]
                questions.append(q)
                answers.append(a)

    else:
        raise ValueError("dataset is not properly defined ...")

    q_len_list = []
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)

    print("dataset : {}".format(args.dataset))
    print("data size : {}".format(len(answers)))
    print("average num of words for each sample : {}".format(q_len_mean))

    return questions, answers


def shuffleDict(d):
    keys = list(d.keys())
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    keys = [(key, d[key]) for key in keys]
    # keys = d(keys)
    return dict(keys)
