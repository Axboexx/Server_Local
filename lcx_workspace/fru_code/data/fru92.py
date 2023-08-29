"""
@Time : 2023/7/12 17:47
@Author : Axboexx
@File : fru92.py
@Software: PyCharm
"""
import torch
import os
from torchvision import transforms
import PIL.Image

ImgLoader = lambda path: PIL.Image.open(path).convert('RGB')


def get_transform(mode, N, CROP, normalize):
    assert mode in ["train",
                    "test"], "specify your mode in `train` and `test`."
    pipeline = []
    if mode == "train":
        pipeline += [transforms.RandomHorizontalFlip(p=0.5)]
        crop_func = transforms.RandomCrop
    else:
        crop_func = transforms.CenterCrop
    pipeline += [
        transforms.Resize((N, N)),
        crop_func((CROP, CROP)),
        transforms.ToTensor(), normalize
    ]

    return transforms.Compose(pipeline)


def get_train_and_test_loader(metadata, k_fold, train_trans, test_trans,
                              train_batchsize, test_batchsize):
    """

    Args:
        metadata ([type]): [description]
        k_fold ([type]): 使用k折交叉验证的数据集，应在此处指定使用第几折。-1代表不适用交叉验证。
        train_trans ([type]): [description]
        test_trans ([type]): [description]
        train_batchsize ([type]): [description]
        test_batchsize ([type]): [description]
    """

    if k_fold >= 0:
        train_txt = metadata["DIR_TRAIN_IMAGES"] % (k_fold)
        test_txt = metadata["DIR_TEST_IMAGES"] % (k_fold)
    else:
        train_txt = metadata["DIR_TRAIN_IMAGES"]
        test_txt = metadata["DIR_TEST_IMAGES"]

    train_dataset = FruDataset(txt_dir=train_txt,
                               file_prefix=metadata["IMAGE_PREFIX"],
                               transform=train_trans)
    test_dataset = FruDataset(txt_dir=test_txt,
                              file_prefix=metadata["IMAGE_PREFIX"],
                              transform=test_trans)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=train_batchsize,
                                               shuffle=True,
                                               num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=test_batchsize,
                                              shuffle=False,
                                              num_workers=2)
    return train_loader, test_loader


class FruDataset(torch.utils.data.Dataset):

    def __init__(self,
                 txt_dir,
                 file_prefix,
                 transform=None,
                 target_transform=None,
                 loader=ImgLoader):
        data_txt = open(txt_dir, 'r')
        imgs = []
        for line in data_txt:
            line = line.strip()
            words = line.split(' ')
            imgs.append((" ".join(words[:-1]), int(words[-1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.file_prefix = file_prefix

    def __len__(self):

        return len(self.imgs)

    def __getitem__(self, index):
        img_name, label = self.imgs[index]
        img = self.loader(os.path.join(self.file_prefix, img_name))
        if self.transform is not None:
            img = self.transform(img)

        return img, label


fru92metainfo_full = dict(NUM_CLASSES=92,
                          DIR_TRAIN_IMAGES='/home/sheng/lcx_workspace/fru92_images/fru92_txt/fru_train.txt',
                          DIR_TEST_IMAGES='/home/sheng/lcx_workspace/fru92_images/fru92_txt/fru_test.txt',
                          IMAGE_PREFIX='/home/sheng/lcx_workspace/fru92_images/fru92_images')

fru92metainfo_kfold = dict(NUM_CLASSES=92,
                           DIR_TRAIN_IMAGES=
                           '/home/lcl/fru/fru92_lists/fru92_fold/fru92_split_split_%d_train.txt',
                           DIR_TEST_IMAGES=
                           '/home/lcl/fru/fru92_lists/fru92_fold/fru92_split_split_%d_test.txt',
                           IMAGE_PREFIX='/home/lcl/fru/fru92_images')
N = 256
CROP = 224
k_fold = -1
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def construct_fru92_data(train_batchsize, test_batchsize):
    train_transforms = get_transform(mode="train", N=N, CROP=CROP, normalize=normalize)
    test_transforms = get_transform(mode="test", N=N, CROP=CROP, normalize=normalize)
    train_loader, test_loader = get_train_and_test_loader(fru92metainfo_full, k_fold, train_transforms,
                                                          test_transforms,
                                                          train_batchsize, test_batchsize)
    return train_transforms, test_transforms, train_loader, test_loader;
