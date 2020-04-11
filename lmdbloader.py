# dataloader respecting the PyTorch conventions, but using tensorpack to load and process
# includes typical augmentations for ImageNet training

import os

import cv2
import torch

import numpy as np
import tensorpack.dataflow as td
from tensorpack import imgaug
from tensorpack.dataflow import (AugmentImageComponent, MultiProcessRunnerZMQ,
                                BatchData, MultiThreadMapData)
import pdb
import torchvision.transforms as transforms

#####################################################################################################
# copied from: https://github.com/ppwwyyxx/tensorpack/blob/master/examples/ResNet/imagenet_utils.py #
#####################################################################################################
class GoogleNetResize(imgaug.ImageAugmentor):
    """
    crop 8%~100% of the original image
    See `Going Deeper with Convolutions` by Google.
    """
    def __init__(self, crop_area_fraction=0.08,
                 aspect_ratio_low=0.75, aspect_ratio_high=1.333,
                 target_shape=224):
        self._init(locals())

    def _augment(self, img, _):
        h, w = img.shape[:2]
        area = h * w
        for _ in range(10):
            targetArea = self.rng.uniform(self.crop_area_fraction, 1.0) * area
            aspectR = self.rng.uniform(self.aspect_ratio_low, self.aspect_ratio_high)
            ww = int(np.sqrt(targetArea * aspectR) + 0.5)
            hh = int(np.sqrt(targetArea / aspectR) + 0.5)
            if self.rng.uniform() < 0.5:
                ww, hh = hh, ww
            if hh <= h and ww <= w:
                x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                out = img[y1:y1 + hh, x1:x1 + ww]
                out = cv2.resize(out, (self.target_shape, self.target_shape), interpolation=cv2.INTER_CUBIC)
                return out
        out = imgaug.ResizeShortestEdge(self.target_shape, interp=cv2.INTER_CUBIC).augment(img)
        out = imgaug.CenterCrop(self.target_shape).augment(out)
        return out


def fbresnet_augmentor(isTrain):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """

    if isTrain:
        augmentors = [
            GoogleNetResize(),
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # rgb-bgr conversion for the constants copied from fb.resnet.torch
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            # flip with prob default = 0.5
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((224, 224)),
        ]
    return augmentors
#####################################################################################################
#####################################################################################################

def lazy_resize(im):
    h, w = im.shape[:2]
    scale = 228.0 / min(h, w)
    desSize = map(int, [scale * w, scale * h])
    if scale > 1:
        im = cv2.resize(im, tuple(desSize), interpolation=cv2.INTER_CUBIC)
    return im

def simple_augmentor():
    augmentors = [
        imgaug.MapImage(lazy_resize),
        imgaug.CenterCrop((224, 224)),
    ]
    return augmentors

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def default_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


class LMDBLoader(object):
    """
    LMDB Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        do_aug (bool): set to ``True`` to do augmentation (regardless of mode)
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the 
            GPU, which is faster).
    """

    def __init__(self, mode, do_aug, batch_size=256, shuffle=False, num_workers=25, cache=50000,
                 cuda=False, out_tensor=True, data_transforms=None):
        # enumerate standard imagenet augmentors
        imagenet_augmentors = fbresnet_augmentor(do_aug)

        # load the lmdb if we can find it
        lmdb_loc = os.path.join(os.environ['IMAGENET'],'ILSVRC-%s.lmdb'%mode)
        ds = td.LMDBSerializer.load(lmdb_loc, shuffle=shuffle)
        #ds = td.LMDBData(lmdb_loc, shuffle=False)
        #ds = td.LocallyShuffleData(ds, cache)
        #ds = td.PrefetchData(ds, 5000, 1)
        #ds = td.LMDBDataPoint(ds)
        ds = td.MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR)[:,:,::-1], 0)
        ds = td.AugmentImageComponent(ds, imagenet_augmentors)
        ds = td.MultiProcessRunnerZMQ(ds, num_workers)
        self.ds = td.BatchData(ds, batch_size)
        self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cuda = cuda
        self.out_tensor = out_tensor
        # data_transforms should be present only when out_tensor=True
        # data_transforms typically consists of 
        # PIL Image transforms, ToTensor(), Normalize():
        #    normalize = transforms.Compose( [
        #          transforms.ToTensor(),
        #          transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                std=[0.229, 0.224, 0.225]) ] )
        self.data_transforms = data_transforms
        print("Loaded '%s'." %lmdb_loc)
                 
    def __iter__(self):
        # x is always 224*224, after being transformed by fbresnet_augmentor
        for x, y in self.ds.get_data():
            if self.out_tensor:
                x = nparray2tensor(x, self.data_transforms, self.cuda)
                y = torch.LongTensor(y)
                if self.cuda:
                    yield x, y.cuda()
                else:
                    yield x, y
            else:
                yield x, y
                
    def __len__(self):
        return self.ds.size()

def nparray2tensor(x, data_transforms, cuda):
    # data_transforms can only be applied to individual images
    if data_transforms:
        # ToTensor() swaps axes b, h, w, 3 -> b, 3, h, w
        x2_shape = list( np.array(x.shape)[[0, 3, 1, 2]] )
        if cuda:
            x2 = torch.zeros(x2_shape).cuda()
        else:
            x2 = torch.zeros(x2_shape)

        for i, xi in enumerate(x):
            # although Normalize() in data_transforms is done in-place, 
            # it's on a temporary tensor whose ref is returned
            x2[i] = data_transforms(xi)
    else:
        # swap axes b, h, w, 3 -> b, 3, h, w
        x2 = torch.from_numpy(np.moveaxis(x, 3, 1))
        if cuda:
            x2 = x2.cuda()
                   
    return x2

if __name__ == '__main__':
    from tqdm import tqdm
    dl = LMDBLoader('train', batch_size=30, cuda=True)
    pbar = tqdm(dl, total=len(dl))
    total_chan_sums = torch.zeros(3).cuda()
    total_pix_count = 0
    for x in pbar:
        chan_sums = x[0].sum(dim=3).sum(dim=2).sum(dim=0)
        total_pix_count += x[0].numel() / 3
        total_chan_sums += chan_sums
        avg_r, avg_g, avg_b = total_chan_sums / total_pix_count
        pbar.set_description("%.3f, %.3f, %.3f" %(avg_r, avg_g, avg_b))
