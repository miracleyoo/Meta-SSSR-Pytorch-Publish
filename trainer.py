import os
import time
import numpy as np
import pdb
import torch
import warnings
from decimal import Decimal
from copy import deepcopy

import utility
from data.common import dotdict

warnings.simplefilter("ignore")


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.scale_test = args.scale_test

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_tests = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.wrap_info()
        
        self.optimizer = utility.make_optimizer(args, self.model)
        self.first_epoch = True

        if self.args.load != '.':
            epoch = len(ckp.log) if self.args.start_epoch == 0 else self.args.start_epoch
            # print("EPOCH: ", epoch)
            self.scheduler = utility.make_scheduler(args, self.optimizer)
            for _ in range(epoch):
                self.scheduler.step()

            if self.args.cpu:
                self.optimizer.load_state_dict(
                    torch.load(os.path.join(ckp.dir, 'optimizer.pt'),
                               map_location=torch.device('cpu'))
                )
            else:
                self.optimizer.load_state_dict(
                    torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
                )
            
            self.first_epoch = False
        else:
            self.scheduler = utility.make_scheduler(args, self.optimizer)

        self.error_last = 1e8

    def wrap_info(self):
        info = self.loader_train.dataset.info
        self.info = dotdict({})
        self.info.root = self.args.dir_data
        self.info.cwl_in = torch.tensor(info.sent_wl, dtype=torch.float32).to(self.device)
        self.info.cwl_out = torch.tensor(info.planet_wl, dtype=torch.float32).to(self.device)
        self.info.cwl_rgb = torch.tensor(info.rgb_wl, dtype=torch.float32).to(self.device)
        
        self.info.bw_in = torch.tensor(info.sent_bw, dtype=torch.float32).to(self.device)
        self.info.bw_out = torch.tensor(info.planet_bw, dtype=torch.float32).to(self.device)
        self.info.bw_rgb = torch.tensor(info.rgb_bw, dtype=torch.float32).to(self.device)
        
        self.info.mean_in = torch.tensor(info.sent_mean, dtype=torch.float32).to(self.device)
        self.info.mean_out = torch.tensor(info.planet_mean, dtype=torch.float32).to(self.device)
        self.info.mean_rgb = torch.tensor(info.rgb_mean, dtype=torch.float32).to(self.device)

        self.info.std_in = torch.tensor(info.sent_std, dtype=torch.float32).to(self.device)
        self.info.std_out = torch.tensor(info.planet_std, dtype=torch.float32).to(self.device)
        self.info.std_rgb = torch.tensor(info.rgb_std, dtype=torch.float32).to(self.device)

    def train(self):
        self.loss.step()
        epoch = self.scheduler.last_epoch
        learning_rate = self.scheduler.get_last_lr()[0]
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(
                epoch+1, Decimal(learning_rate))
        )

        # Add one line of zeros which has the same number
        # as applied loss types and use it as logs.
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model, timer_total = utility.timer(), utility.timer(), utility.timer()

        batch_num = len(self.loader_train)
        timer_total.tic()
        for batch, data in enumerate(self.loader_train):
            lr, hr, rgb_lr, rgb_hr, _ = data
            rgb_lr, rgb_hr = self.prepare(rgb_lr, rgb_hr)
            lr, hr = self.prepare(lr, hr)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, rgb_lr, rgb_hr, self.info)
            # loss = self.loss(sr, hr, lr)
            loss = self.loss(sr=sr, lr=lr, rgb_hr=rgb_hr)

            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
                self.loss.step()
            else:
                print('Skip this batch {}! (Loss: {}). Loss, SR deleted.'.format(
                    batch + 1, loss.item()
                ))
                del loss, sr
            timer_model.hold()

            if (batch + 1) % (batch_num // 10) == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.ckp.write_log(f'Epoch {epoch+1} total time: {timer_total.toc()}')

        if self.args.no_test:
            self.ckp.add_log(torch.zeros(1, len(self.loader_tests), 1))
            self.ckp.save(self, epoch+1, False)

    def test(self):
        epoch = self.scheduler.last_epoch
        scale_test = self.args.scale_test

        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.loader_tests), 1))
        self.model.eval()
        timer_test = utility.timer()
        with torch.no_grad():
            for loader_idx in range(len(self.loader_tests)):
                self.loader_test = self.loader_tests[loader_idx]

                for idx_img, data in enumerate(self.loader_test):
                    lr, hr, rgb_lr, rgb_hr, filename = data
                    save_list = [lr, rgb_hr]
                    rgb_lr, rgb_hr = self.prepare(rgb_lr, rgb_hr)
                    lr, hr = self.prepare(lr, hr)

                    filename = filename[0]

                    timer_test.tic()
                    sr = self.model(lr, rgb_lr, rgb_hr, self.info)
                    timer_test.hold()
                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_list.append(sr)
                    

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale_test)

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch+1, is_best=False)#(best[1] == epoch))

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.args.epochs
