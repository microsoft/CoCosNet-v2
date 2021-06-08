# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import torch
from torchvision.utils import save_image
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.util import print_current_errors
from util.util import mkdir
from trainers.pix2pix_trainer import Pix2PixTrainer


if __name__ == '__main__':
    # parse options
    opt = TrainOptions().parse()
    # print options to help debugging
    print(' '.join(sys.argv))
    dataloader = data.create_dataloader(opt)
    len_dataloader = len(dataloader)
    # create tool for counting iterations
    iter_counter = IterationCounter(opt, len(dataloader))
    # create trainer for our model
    trainer = Pix2PixTrainer(opt, resume_epoch=iter_counter.first_epoch)
    save_root = os.path.join('checkpoints', opt.name, 'train')
    mkdir(save_root)

    for epoch in iter_counter.training_epochs():
        opt.epoch = epoch
        iter_counter.record_epoch_start(epoch)
        for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()
            # Training
            # train generator
            if i % opt.D_steps_per_G == 0:
                trainer.run_generator_one_step(data_i)
            # train discriminator
            trainer.run_discriminator_one_step(data_i)
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                try:
                    print_current_errors(opt, epoch, iter_counter.epoch_iter,
                                         iter_counter.epoch_iter_num, losses, iter_counter.time_per_iter)
                except OSError as err:
                    print(err)

            if iter_counter.needs_displaying():
                imgs_num = data_i['label'].shape[0]

                if opt.dataset_mode == 'deepfashionHD':
                    label = data_i['label'][:,:3,:,:]

                show_size = opt.display_winsize

                imgs = torch.cat((label.cpu(), data_i['ref'].cpu(), \
                                trainer.get_latest_generated().data.cpu(), \
                                data_i['image'].cpu()), 0)

                try:
                    save_name = '%08d_%08d.png' % (epoch, iter_counter.total_steps_so_far)
                    save_name = os.path.join(save_root, save_name)
                    save_image(imgs, save_name, nrow=imgs_num, padding=0, normalize=True)
                except OSError as err:
                    print(err)

            if iter_counter.needs_saving():
                print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, iter_counter.total_steps_so_far))
                try:
                    trainer.save('latest')
                    iter_counter.record_current_iter()
                except OSError as err:
                    import pdb; pdb.set_trace()
                    print(err)

        trainer.update_learning_rate(epoch)
        iter_counter.record_epoch_end()

        if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
            print('saving the model at the end of epoch %d, iters %d' %
                (epoch, iter_counter.total_steps_so_far))
            try:
                trainer.save('latest')
                trainer.save(epoch)
            except OSError as err:
                print(err)

    print('Training was successfully finished.')
