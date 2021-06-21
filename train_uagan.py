# -*- coding: utf-8 -*-
from torchvision.utils import save_image
import torch
from torch import optim
import torch.nn.functional as F
from torch.backends import cudnn
import numpy as np
import os
import time
import datetime
import argparse
import random

from data.brain import get_loaders
from utils.util import dice_score, check_dirs, print_net
from net.uagan import Discriminator, UAGAN


class Solver:
    def __init__(self, data_files, opt):
        self.opt = opt

        # Data Loader.
        self.phase = self.opt.phase
        self.selected_attr = self.opt.selected_attr
        self.image_size = self.opt.image_size
        self.batch_size = self.opt.batch_size
        self.num_workers = self.opt.num_workers
        loaders = get_loaders(data_files, self.selected_attr, self.batch_size,
                              self.num_workers, self.image_size)
        self.loaders = {x: loaders[x] for x in ('train', 'test')}

        # Model Configurations.
        self.c_dim = len(self.selected_attr)
        self.in_channels = self.c_dim + 1
        self.out_channels = self.opt.out_channels
        self.feature_maps = self.opt.feature_maps
        self.levels = self.opt.levels
        self.norm_type = self.opt.norm_type
        self.use_dropout = self.opt.use_dropout
        self.d_conv_dim = self.opt.d_conv_dim
        self.d_repeat_num = self.opt.d_repeat_num

        self.lambda_cls = self.opt.lambda_cls
        self.lambda_rec = self.opt.lambda_rec
        self.lambda_gp = self.opt.lambda_gp
        self.lambda_seg = self.opt.lambda_seg
        self.lambda_shape = self.opt.lambda_shape

        # Train Configurations.
        self.max_epoch = self.opt.max_epoch
        self.decay_epoch = self.opt.decay_epoch
        self.g_lr = self.opt.g_lr
        self.min_g_lr = self.opt.min_g_lr
        self.d_lr = self.opt.d_lr
        self.min_d_lr = self.opt.min_d_lr
        self.beta1 = self.opt.beta1
        self.beta2 = self.opt.beta2
        self.ignore_index = self.opt.ignore_index
        self.seg_loss_type = self.opt.seg_loss_type
        self.n_critic = self.opt.n_critic

        # Test Configurations.
        self.test_epoch = self.opt.test_epoch

        # Miscellaneous
        self.use_tensorboard = self.opt.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.checkpoint_dir = self.opt.checkpoint_dir
        self.log_dir = os.path.join(self.checkpoint_dir, 'logs')
        self.sample_dir = os.path.join(self.checkpoint_dir, 'sample_dir')
        self.model_save_dir = os.path.join(self.checkpoint_dir, 'model_save_dir')
        self.result_dir = os.path.join(self.checkpoint_dir, 'result_dir')
        check_dirs([self.log_dir, self.sample_dir, self.model_save_dir, self.result_dir])

        # Step Size.
        self.log_step = self.opt.log_step
        self.lr_update_epoch = self.opt.lr_update_epoch

        # Build Model and Tensorboard.
        self.G = None
        self.D = None
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        self.G = UAGAN(1, self.out_channels, self.in_channels, 1, self.feature_maps, self.levels,
                      self.norm_type, self.use_dropout)
        if self.phase == 'train':
            print_net(self.G)
        self.G.to(self.device)

        if self.phase == 'train':
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
            print_net(self.D)
            self.D.to(self.device)

            self.g_optimizer = optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
            self.d_optimizer = optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

    def build_tensorboard(self):
        from tensorboardX import SummaryWriter
        self.writer = SummaryWriter(self.log_dir)

    def restore_model(self, resume_epoch):
        print('Loading the trained models from step {}...'.format(resume_epoch))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_epoch))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

    def save_model(self, save_iters):
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(save_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(save_iters))
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)
        print('Saved model checkpoints into {}...'.format(self.model_save_dir))

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onenot(self, labels, dim):
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    @staticmethod
    def classification_loss(logit, target):
        return F.cross_entropy(logit, target)

    def seg_loss(self, outputs, targets, ignore_idx, loss_type='cross-entropy', weight=None):
        if loss_type == 'cross-entropy':
            if weight is not None:
                weight = weight.to(self.device)
                if ignore_idx:
                    return F.cross_entropy(outputs, targets, ignore_index=ignore_idx, weight=weight)
                return F.cross_entropy(outputs, targets, weight=weight)
            if ignore_idx:
                return F.cross_entropy(outputs, targets, ignore_index=ignore_idx)
            return F.cross_entropy(outputs, targets)
        else:
            exit('[Error] Loss type not found!')

    def train(self):

        g_lr = self.g_lr
        d_lr = self.d_lr
        loader = self.loaders['train']
        lambda_shape = 0

        if self.opt.use_weight:
            weight = torch.Tensor([0.02, 0.98])
        else:
            weight = None

        print('\nStart training...')
        start_time = time.time()
        for epoch in range(self.max_epoch):
            self.G.train()
            self.D.train()
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
            for i, (x_real, y_real, vec_org, label_org, name) in enumerate(loader):
                cur_step = epoch * len(loader) + i

                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]
                vec_trg = self.label2onenot(label_trg, self.c_dim)

                x_real = x_real.to(self.device)
                y_real = y_real.to(self.device)
                label_org = label_org.to(self.device)
                label_trg = label_trg.to(self.device)
                vec_org = vec_org.to(self.device)
                vec_trg = vec_trg.to(self.device)

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Compute loss with real images.
                out_src, out_cls = self.D(x_real)
                d_loss_real = - torch.mean(out_src)
                d_loss_cls = self.classification_loss(out_cls, label_org)

                # Compute loss with fake images.
                _, x_fake = self.G(x_real, vec_trg)
                out_src, _ = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_cls'] = d_loss_cls.item()
                loss['D/loss_gp'] = d_loss_gp.item()

                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #

                # Calc shape loss weight.
                if (cur_step + 1) < self.opt.shape_epoch * len(loader) + 1:
                    td = self.opt.shape_epoch * len(loader)
                    lambda_shape += self.lambda_shape / td

                if (i + 1) % self.n_critic == 0:
                    # Original-to-target domain.
                    y_seg, x_fake = self.G(x_real, vec_trg)

                    _, pred = torch.max(y_seg, 1)

                    cur_dices = dice_score(pred, y_real.detach(), 1, reduce=True) * y_real.size(0)
                    cur_samples = y_real.size(0)

                    out_src, out_cls = self.D(x_fake)
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_cls = self.classification_loss(out_cls, label_trg)
                    g_loss_seg = self.seg_loss(y_seg, y_real.detach(), self.ignore_index, self.seg_loss_type,
                                               weight)
                    # Target-to-original domain.
                    y_rec, x_rec = self.G(x_fake, vec_org)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_rec))
                    g_loss_shape = self.seg_loss(y_rec, y_real.detach(), self.ignore_index, self.seg_loss_type,
                                                 weight)

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.lambda_cls * g_loss_cls + self.lambda_seg * g_loss_seg + \
                             self.lambda_rec * g_loss_rec + lambda_shape * g_loss_shape
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    loss['G/lambda_shape'] = lambda_shape
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_cls'] = g_loss_cls.item()
                    loss['G/loss_seg'] = g_loss_seg.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_shape'] = g_loss_shape.item()
                    loss['train/dps'] = cur_dices / cur_samples

                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #
                if (cur_step + 1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    line = "Elapsed [{}], Epoch [{}/{}], Iterations [{}]".format(et, epoch + 1, self.max_epoch,
                                                                                 cur_step)
                    for k, v in loss.items():
                        line += ", {}: {:.4f}".format(k, v)
                        if self.use_tensorboard:
                            # self.writer.add_scalar(k, v, (cur_step + 1) // self.n_critic)
                            self.writer.add_scalar(k, v, (cur_step + 1))
                    print(line, flush=True)

            # Decay learning rates.
            if (epoch + 1) % self.lr_update_epoch == 0 and (epoch + 1) > (self.max_epoch - self.decay_epoch):
                g_dlr = self.g_lr - self.min_g_lr
                g_lr -= g_dlr / (self.decay_epoch / self.lr_update_epoch)
                d_dlr = self.d_lr - self.min_d_lr
                d_lr -= d_dlr / (self.decay_epoch / self.lr_update_epoch)
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
            if self.use_tensorboard:
                self.writer.add_scalar('G/g_lr', g_lr, epoch + 1)
                self.writer.add_scalar('D/d_lr', d_lr, epoch + 1)

        self.save_model(epoch + 1)

    def infer(self, epoch):
        from PIL import Image
        from process.utils import parse_image_name

        save_dir = os.path.join(self.result_dir, str(epoch))
        check_dirs(save_dir)
        self.restore_model(epoch)
        self.G.eval()

        print('Start Testing at iter {}...'.format(epoch))
        with torch.no_grad():
            for i, (x, _, vec, cls, names) in enumerate(self.loaders['test']):
                x = x.to(self.device)
                vec = vec.to(self.device)

                o, _ = self.G(x, vec)
                _, preds = torch.max(o, 1)

                preds = preds.cpu().numpy()
                for b in range(preds.shape[0]):
                    _, pid, index, _ = parse_image_name(names[b])
                    pred = preds[b, ...]
                    pred[pred == 1] = 255
                    img = Image.fromarray(pred.astype('uint8'))
                    img.save(os.path.join(save_dir, '{}_{}.png'.format(pid, index)))
        return


if __name__ == '__main__':
    cudnn.benchmark = True

    args = argparse.ArgumentParser()
    args.add_argument('--train_list', type=str)
    args.add_argument('--test_list', type=str)

    # Data Loader.
    args.add_argument('--phase', type=str, default='train')
    args.add_argument('--selected_attr', nargs='+', default=['FLAIR', 'T1', 'T1c', 'T2'])
    args.add_argument('--image_size', type=int, default=128)
    args.add_argument('--batch_size', type=int, default=8)
    args.add_argument('--num_workers', type=int, default=4)

    # Model configurations.
    args.add_argument('--out_channels', type=int, default=2)
    args.add_argument('--feature_maps', type=int, default=64)
    args.add_argument('--levels', type=int, default=4)
    args.add_argument('--norm_type', type=str, default='instance')
    args.add_argument('--use_dropout', type=int, default=1)
    args.add_argument('--d_conv_dim', type=int, default=64)
    args.add_argument('--d_repeat_num', type=int, default=6)

    # Lambda.
    args.add_argument('--lambda_cls', type=float, default=1)
    args.add_argument('--lambda_rec', type=float, default=10)
    args.add_argument('--lambda_gp', type=float, default=10)
    args.add_argument('--lambda_seg', type=float, default=100)
    args.add_argument('--lambda_shape', type=float, default=100)

    # Train configurations.
    args.add_argument('--max_epoch', type=int, default=100)
    args.add_argument('--decay_epoch', type=int, default=50)
    args.add_argument('--shape_epoch', type=int, default=100)
    args.add_argument('--g_lr', type=float, default=1e-4)
    args.add_argument('--min_g_lr', type=float, default=1e-6)
    args.add_argument('--d_lr', type=float, default=1e-4)
    args.add_argument('--min_d_lr', type=float, default=1e-6)
    args.add_argument('--beta1', type=float, default=0.9)
    args.add_argument('--beta2', type=float, default=0.999)
    args.add_argument('--ignore_index', type=int, default=None)
    args.add_argument('--seg_loss_type', type=str, default='cross-entropy')
    args.add_argument('--seed', type=int, default=1234)
    args.add_argument('--use_weight', type=int, default=1)
    args.add_argument('--n_critic', type=int, default=1)

    # Test configurations.
    args.add_argument('--test_epoch', nargs='+', default=None)

    # Miscellaneous.
    args.add_argument('--use_tensorboard', type=int, default=1)
    args.add_argument('--device', type=int, default=1)
    args.add_argument('--gpu_id', type=str, default='0')

    # Directories.
    args.add_argument('--checkpoint_dir', type=str, default='unet')

    # Step size.
    args.add_argument('--log_step', type=int, default=200)
    args.add_argument('--lr_update_epoch', type=int, default=1)
    args = args.parse_args()
    print('-----Config-----')
    for k, v in sorted(vars(args).items()):
        print('%s:\t%s' % (str(k), str(v)))
    print('-------End------\n')

    # Set Random Seed.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    data_files = dict(train=args.train_list, test=args.test_list)
    solver = Solver(data_files, args)
    if args.phase == 'train':
        solver.train()
    elif args.phase == 'test':
        for test_iter in args.test_epoch:
            test_iter = int(test_iter)
            solver.infer(test_iter)
            print()

    print('Done!')
