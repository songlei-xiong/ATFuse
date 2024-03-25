# -*- coding: utf-8 -*-
import argparse, random
from tqdm import tqdm

import torch

from visualization import get_local

get_local.activate()

import options.options as option
from utils import util
from solvers import create_solver
from data import create_dataloader
from data import create_dataset
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_options(option_file_path):
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    opt = option.parse(option_file_path)
    return opt


def set_random_seed(opt):
    seed = opt['solver']['manual_seed']
    if seed is None: seed = random.randint(1, 10000)  # 这里尽量直接把seed写死
    print(f"===> Random Seed: [{seed}]")
    random.seed(seed)
    torch.manual_seed(seed)


def create_Dataloader(opt):
    train_set, train_loader, val_set, val_loader = None, None, None, None
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt)
            print(f'===> Train Dataset: {train_set.name()}   Number of images: [{len(train_set)}]')
            if train_loader is None: raise ValueError("[Error] The training data does not exist")

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print(f'===> Val Dataset: {val_set.name()}   Number of images: [{len(val_set)}]')

        else:
            raise NotImplementedError(f"[Error] Dataset phase [{phase}] in *.json is not recognized.")
    return train_set, train_loader, val_set, val_loader


def train_step():
    pass


def main():
    option_file_path = 'options/train/train_ATFuse.json'
    opt = parse_options(option_file_path)
    # random seed
    set_random_seed(opt)

    # create train and val dataloader
    train_set, train_loader, val_set, val_loader = create_Dataloader(opt)

    solver = create_solver(opt)

    scale = opt['scale']
    model_name = opt['networks']['which_model'].upper()
    solver_log = solver.get_current_log()

    NUM_EPOCH = int(opt['solver']['num_epochs'])
    start_epoch = solver_log['epoch']

    print('===> Start Train')
    print("==================================================")
    print(f"Method: {model_name} || Scale: {scale} || Epoch Range: ({start_epoch} ~ {NUM_EPOCH})")
    solver_log['best_pred'] = 10000000000
    for epoch in range(start_epoch, NUM_EPOCH + 1):
        print(
            f'\n===> Training Epoch: [{epoch}/{NUM_EPOCH}]...  Learning Rate: {solver.get_current_learning_rate():.2e}')

        # Initialization
        solver_log['epoch'] = epoch

        # Train model
        get_local.clear()

        train_loss_list = []
        with tqdm(total=len(train_loader), desc=f'Epoch: [{epoch}/{NUM_EPOCH}]', miniters=1) as t:
            for iter, batch in enumerate(train_loader):
                solver.feed_data(batch)
                iter_loss = solver.train_step()
                batch_size = batch['Vi'].size(0)
                train_loss_list.append(iter_loss * batch_size)
                t.set_postfix_str("Batch Loss: %.4f" % iter_loss)
                t.update()
                cache = get_local.cache
                # print(cache)

        solver_log['records']['train_loss'].append(sum(train_loss_list) / len(train_set))
        solver_log['records']['lr'].append(solver.get_current_learning_rate())

        print(f'\nEpoch: [{epoch}/{NUM_EPOCH}]   Avg Train Loss: {sum(train_loss_list) / len(train_set):.6f}')

        print('===> Validating...', )

        psnr_list = []
        ssim_list = []
        val_loss_list = []

        for iter, batch in enumerate(val_loader):
            solver.feed_data(batch)
            iter_loss = solver.test()
            val_loss_list.append(iter_loss)

            # calculate evaluation metrics
            visuals = solver.get_current_visual()
            # 评估指标需要改，这是监督学习的
            psnr, ssim = util.calc_metrics(visuals['Fu'], visuals['Vi'], visuals['Ir'],crop_border=scale)
            # psnrIr, ssimIr = util.calc_metrics(visuals['Fu'], visuals['Ir'], crop_border=scale)
            # psnr_list.append((psnrVi+psnrIr)/2)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            if opt["save_image"] and epoch % 50 == 0:
                solver.save_current_visual(epoch, iter)

        solver_log['records']['val_loss'].append(sum(val_loss_list) / len(val_loss_list))
        solver_log['records']['psnr'].append(sum(psnr_list) / len(psnr_list))
        solver_log['records']['ssim'].append(sum(ssim_list) / len(ssim_list))

        # record the best epoch
        # epoch_is_best = False
        # if solver_log['best_pred'] < (sum(psnr_list) / len(psnr_list)):
        #     solver_log['best_pred'] = (sum(psnr_list) / len(psnr_list))
        #     epoch_is_best = True
        #     solver_log['best_epoch'] = epoch
        #best，损失函数最小
        epoch_is_best = False
        if solver_log['best_pred'] > (sum(val_loss_list) / len(val_loss_list)):
            solver_log['best_pred'] = (sum(val_loss_list) / len(val_loss_list))
            epoch_is_best = True
            solver_log['best_epoch'] = epoch

        # print(
        #     f"[{val_set.name()}] CC: {sum(psnr_list) / len(psnr_list):.4f}   RMSE: {sum(ssim_list) / len(ssim_list):.4f}"
        #     f"Loss: {sum(val_loss_list) / len(val_loss_list):.6f}   Best CC: {solver_log['best_pred']:.4f} in Epoch: "
        #     f"[{solver_log['best_epoch']:d}]"
        # )
        print(
            f"[{val_set.name()}] CC: {sum(psnr_list) / len(psnr_list):.4f}   Qabf: {sum(ssim_list) / len(ssim_list):.4f}"
            f"Loss: {sum(val_loss_list) / len(val_loss_list):.6f}   Best : {solver_log['best_pred']:.6f} in Epoch: "
            f"[{solver_log['best_epoch']:d}]"
        )

        solver.set_current_log(solver_log)
        solver.save_checkpoint(epoch, epoch_is_best)
        solver.save_current_log()
        # update lr
        solver.update_learning_rate(epoch)

    print('===> Finished !')


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    main()
