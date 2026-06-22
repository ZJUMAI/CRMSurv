import os
import numpy as np
from tqdm import tqdm

from sksurv.metrics import concordance_index_censored

import torch.optim
import torch.nn.parallel
from torch.utils.tensorboard import SummaryWriter

class Engine(object):
    def __init__(self, args, results_dir, fold):
        self.args = args
        self.results_dir = results_dir
        self.fold = fold

        # tensorboard
        if args.log_data:
            # from tensorboardX import SummaryWriter
            writer_dir = os.path.join(results_dir, 'fold_' + str(fold))
            if not os.path.isdir(writer_dir):
                os.mkdir(writer_dir)
            self.writer = SummaryWriter(writer_dir, flush_secs=15)
        self.best_score = 0
        self.best_epoch = 0
        self.filename_best = None

    def learning(self, model, train_loader, val_loader, criterion, optimizer, scheduler):
        if torch.cuda.is_available():
            model = model.cuda()
        self.epoch = 0
        self.validate(val_loader, model, criterion)

    def validate(self, data_loader, model, criterion):
        model.eval()
        dataloader = tqdm(data_loader, desc='Test Epoch: {}'.format(self.epoch))
        save_path = os.path.join(self.args.save_cluster_path, self.args.cancer_type)
        os.makedirs(save_path, exist_ok=True) 
        for batch_idx, (wsi, omic, label, event_time, c, case_id) in enumerate(dataloader):

            print('dataload,',case_id)
            if torch.cuda.is_available():
                wsi = wsi.cuda()

            with torch.no_grad():
                centers = model(x_path=wsi)
                np.savez(os.path.join(save_path, f'{case_id[0]}.npz'), cluster=centers.detach().cpu().numpy())
