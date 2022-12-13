import random
import torch.utils.data as data


class Mix(data.Dataset):
    def __init__(self, args, datasets_list):
        self.datasets_list = datasets_list
        self.args = args
        self.counter = 0
        self.choice = 0
        self.length = min([len(i) for i in self.datasets_list])

    def __getitem__(self, idx):
        if self.counter % self.args.batch_size == 0:
            self.choice = random.randint(0, len(self.datasets_list)-1)

        dataset_chosen = self.datasets_list[self.choice]
        self.counter += 1

        return dataset_chosen[idx % len(dataset_chosen)]

    def __len__(self):
        return self.length
