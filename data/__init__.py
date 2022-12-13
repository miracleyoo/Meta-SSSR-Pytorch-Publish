from importlib import import_module
from torch.utils.data import DataLoader


class Data:
    def __init__(self, args):
        self.loader_train = None
        self.args=args
        module_train = import_module('data.sat_data')
        trainset = getattr(module_train, 'SatData')(args, train=True)
        self.loader_train = DataLoader(
            dataset=trainset,
            num_workers=args.n_threads,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=not args.cpu,
            drop_last=False
        )

        testset = getattr(module_train, 'SatData')(args, train=False)
        self.loader_test = [DataLoader(
            dataset=testset,
            num_workers=args.n_threads,
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu,
            drop_last=False
        )]
