import torch

import utility
import data
import model
import loss
from option import init
from trainer import Trainer

# torch.manual_seed(args.seed)
args = init()
utility.set_random_seed(args.seed)
# setting the log and the train information
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    checkpoint.write_log(f'==> Model Type: {args.model}')
    checkpoint.write_log(f'==> Save Folder Name: {args.save}')
    
    loader = data.Data(args)  # data loader
    model = model.Model(args, checkpoint)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():

        t.train()
        if not args.no_test:
            t.test()
        t.first_epoch = False
        t.scheduler.step()

    checkpoint.done()
