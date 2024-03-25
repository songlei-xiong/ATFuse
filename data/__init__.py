import torch.utils.data


def create_dataloader(dataset, dataset_opt):
    phase = dataset_opt['phase']
    if phase == 'train':
        batch_size = dataset_opt['batch_size']
        shuffle = True
        num_workers = dataset_opt['n_workers']                  #这是什么参数
    else:
        batch_size = 1
        shuffle = False
        num_workers = 1
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode'].upper()
    if mode == 'LRHR':
        from data.LRHR_dataset import LRHRDataset as D
    elif mode == 'IRVI':
        from data.IrVi_dataset import IrViDataset as D
    else:
        raise NotImplementedError("Dataset [%s] is not recognized." % mode)
    dataset = D(dataset_opt)
    print('===> [%s] Dataset is created.' % (mode))
    return dataset


# def create_dataset(dataset_opt):
#     mode = dataset_opt['mode'].upper()
#     if mode == 'IrVi':
#         from data.IrVi_dataset import IrViDataset as D
#     else:
#         raise NotImplementedError("Dataset [%s] is not recognized." % mode)
#     dataset = D(dataset_opt)
#     print('===> [%s] Dataset is created.' % (mode))
#     return dataset