import os
import numpy as np
from tqdm import tqdm

from sksurv.metrics import concordance_index_censored
import pandas as pd
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

        if self.args.resume is not None:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.best_score = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint (score: {})".format(checkpoint['best_score']))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))

        if self.args.evaluate:
            self.epoch=-1
            self.validate_test(val_loader, model, criterion)
            return

        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            # train for one epoch
            self.train(train_loader, model, criterion, optimizer)
            # evaluate on validation set
            c_index = self.validate(val_loader, model, criterion)

            is_best = c_index > self.best_score
            if is_best:
                self.best_score = c_index
                self.best_epoch = self.epoch
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_score': self.best_score})
            print(' *** best c-index={:.4f} at epoch {}'.format(self.best_score, self.best_epoch))
            if scheduler is not None:
                scheduler.step()
            print('>')
        return self.best_score, self.best_epoch

    def train(self, data_loader, model, criterion, optimizer):
        model.train()
        train_loss = 0.0
        rl_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc='Train Epoch: {}'.format(self.epoch))
        for batch_idx, (wsi_txt, gene_txt, cluster, wsi, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(dataloader):

            if torch.cuda.is_available():
                wsi = wsi.cuda()
                wsi_txt = wsi_txt.type(torch.FloatTensor).cuda()
                gene_txt = gene_txt.type(torch.FloatTensor).cuda()
                cluster = cluster.type(torch.FloatTensor).cuda()
                data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
                data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
                data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
                data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
                data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
                data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
                label = label.type(torch.LongTensor).cuda()
                c = c.type(torch.FloatTensor).cuda()

            hazards, S,  P, G, fusion = model(wsi_txt=wsi_txt, gene_txt=gene_txt, cluster=cluster, x_path=wsi, x_omic1=data_omic1, x_omic2=data_omic2,
                            x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)

            # survival loss + rl loss
            sur_loss = criterion[0](hazards=hazards, S=S, Y=label, c=c)

            if self.args.use_rl_loss:
                loss_rl = criterion[1](P, label, c=c) + criterion[2](G, label, c=c)
                loss = sur_loss + self.args.beta * loss_rl
                rl_loss += loss_rl.item()
            else:
                loss = sur_loss 


            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if self.args.use_rl_loss:
                model.wsi_bank.update(P.detach(), label, c=c)
                model.omic_bank.update(G.detach(), label, c=c)

        # calculate loss and error for epoch
        train_loss /= len(dataloader)
        if self.args.use_rl_loss:
            rl_loss /= len(dataloader)
        c_index = concordance_index_censored((1-all_censorships).astype(bool),
                                             all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(train_loss, c_index))

        if self.writer:
            self.writer.add_scalar('train/loss', train_loss, self.epoch)
            if self.args.use_rl_loss:
                self.writer.add_scalar('train/rl_loss', rl_loss, self.epoch)
            self.writer.add_scalar('train/c_index', c_index, self.epoch)

    def validate(self, data_loader, model, criterion):
        model.eval()
        val_loss = 0.0
        val_rl_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        all_labels = np.zeros((len(data_loader)))
        if self.args.evaluate:
            f_gene = []
            f_wsi = []
            f_fusion = []
            f_error = []
            f_keepid = []
            f_alpha = []
            f_cluster = []
            f_shape = []
            f_caseid= []


        dataloader = tqdm(data_loader, desc='Test Epoch: {}'.format(self.epoch))
        for batch_idx, (wsi_txt, gene_txt, cluster, wsi, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6,
                        label, event_time,
        c) in enumerate(dataloader):

            if torch.cuda.is_available():
                wsi = wsi.cuda()
                wsi_txt = wsi_txt.type(torch.FloatTensor).cuda()
                gene_txt = gene_txt.type(torch.FloatTensor).cuda()
                cluster = cluster.type(torch.FloatTensor).cuda()
                data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
                data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
                data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
                data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
                data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
                data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
                label = label.type(torch.LongTensor).cuda()
                c = c.type(torch.FloatTensor).cuda()

            with torch.no_grad():
                if not self.args.evaluate:
                    hazards, S, P, G, fusion = model(wsi_txt=wsi_txt, gene_txt=gene_txt, cluster=cluster, x_path=wsi, x_omic1=data_omic1, x_omic2=data_omic2,
                                       x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
                else:
                    hazards, S, P, G, fusion, aux = model(wsi_txt=wsi_txt, gene_txt=gene_txt, cluster=cluster, x_path=wsi,
                                                     x_omic1=data_omic1, x_omic2=data_omic2,
                                                     x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5,
                                                     x_omic6=data_omic6,)

            # survival loss + rl loss
            sur_loss = criterion[0](hazards=hazards, S=S, Y=label, c=c)

            if self.args.use_rl_loss:
                loss_rl = criterion[1](P,label,c=c) + criterion[2](G,label,c=c)
                loss = sur_loss + self.args.beta * loss_rl
                val_rl_loss += loss_rl.item()
            else:
                loss = sur_loss 

            if self.args.evaluate:
                f_wsi.append(P.squeeze().detach().cpu().numpy())
                f_gene.append(G.squeeze().detach().cpu().numpy())
                f_fusion.append(fusion.squeeze().detach().cpu().numpy())
                f_error.append(aux['error'].squeeze().detach().cpu().numpy())
                f_keepid.append(aux['keep_idx'].squeeze().detach().cpu().numpy())
                f_alpha.append(aux['alpha'].squeeze().detach().cpu().numpy())
                f_cluster.append(aux['cluster'].squeeze().detach().cpu().numpy())
                f_shape.append(aux['wsi_shape'])

            risk = -torch.sum(S, dim=1).cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.cpu().numpy()
            all_event_times[batch_idx] = event_time
            all_labels[batch_idx] = label.cpu().numpy()
            val_loss += loss.item()

        val_loss /= len(dataloader)
        if self.args.use_rl_loss:
            val_rl_loss /= len(dataloader)
        if self.args.evaluate:
            self.save_prototype_test(model)

            predict_data = {'OS': all_event_times, 'alive': all_censorships, 'risk': all_risk_scores,
                            'label': all_labels}
            df = pd.DataFrame(predict_data)

            df.to_csv(self.args.results_dir +f'/fold_{self.fold}'+ '/predict_label.csv')

            f_wsi = np.vstack([f_wsi])
            f_gene = np.vstack([f_gene])
            f_fusion = np.vstack([f_fusion])
            f_shape=  np.vstack([f_shape])


            np.save(self.args.results_dir +f'/fold_{self.fold}'+ '/f_wsi', f_wsi)
            np.save(self.args.results_dir +f'/fold_{self.fold}'+ '/f_gene', f_gene)
            np.save(self.args.results_dir +f'/fold_{self.fold}'+ '/f_fusion', f_fusion)
            np.save(self.args.results_dir + f'/fold_{self.fold}' + '/f_shape', f_shape)
            np.save(self.args.results_dir +f'/fold_{self.fold}'+ '/f_error.npy', np.array(f_error, dtype=object), allow_pickle=True)
            np.save(self.args.results_dir +f'/fold_{self.fold}'+ '/f_keepid.npy', np.array(f_keepid, dtype=object), allow_pickle=True)
            np.save(self.args.results_dir +f'/fold_{self.fold}'+ '/f_alpha.npy', np.array(f_alpha, dtype=object), allow_pickle=True)
            np.save(self.args.results_dir +f'/fold_{self.fold}'+ '/f_cluster.npy', np.array(f_cluster, dtype=object), allow_pickle=True)

        c_index = concordance_index_censored((1-all_censorships).astype(bool),
                                             all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(val_loss, c_index))
        if self.writer:
            self.writer.add_scalar('val/loss', val_loss, self.epoch)
            if self.args.use_rl_loss:
                self.writer.add_scalar('val/rl_loss', val_rl_loss, self.epoch)
            self.writer.add_scalar('val/c-index', c_index, self.epoch)


        return c_index


    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(self.results_dir,
                                          'fold_' + str(self.fold),
                                          'model_best_{score:.4f}_{epoch}.pth.tar'.format(score=state['best_score'], epoch=state['epoch']))
        print('save best model {filename}'.format(filename=self.filename_best))
        torch.save(state, self.filename_best)

    def save_prototype(self, model):
        wsi_val = model.wsi_bank.prototypes.data.cpu().numpy()
        omic_val = model.omic_bank.prototypes.data.cpu().numpy()

        save_path = os.path.join(
            self.args.results_dir, f'prototype_fold_{self.fold}'
        )
        os.makedirs(save_path, exist_ok=True) 

        np.savez(os.path.join(save_path, f'epoch_{self.epoch}_prototype.npz'), wsi=wsi_val, omic=omic_val)