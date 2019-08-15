import pretrainedmodels
from PIL import Image
import numpy as np
import torchvision.transforms as trans
from torch.nn.functional import interpolate
import torch

def pearson(t1, t2):
    t1flat = t1.view(-1)
    t2flat = t2.view(-1)
    t1flatz = t1flat - t1flat.mean()
    t2flatz = t2flat - t2flat.mean()
    norm1 = (t1flatz**2).sum().sqrt()
    norm2 = (t2flatz**2).sum().sqrt()
    corr = (t1flatz * t2flatz).sum() / (norm1 * norm2)
    return corr.data.cpu().numpy()

resnet = pretrainedmodels.__dict__['resnet101']()
resnet.cuda()
for angle in range(0, 360, 30):
    # raw_data = np.random.randint(0, 256, (224, 224, 3)).astype(np.uint8)
    # im = Image.fromarray(raw_data)
    im = Image.open("d:/stylegan.png")
    raw_data = np.array(im)[:, :, :3]
    im = Image.fromarray(raw_data)    
    im_ten = trans.ToTensor()(im).unsqueeze(0).cuda()
    # [1, 2048, 7, 7]
    with torch.no_grad():
        orig_feat = resnet.features(im_ten)
    im_rot = im.rotate(angle, resample=Image.BILINEAR, expand=True)
    im_rot_ten = trans.ToTensor()(im_rot).unsqueeze(0).cuda()
    # [1, 2048, 10, 10]
    with torch.no_grad():
        rot_feat = resnet.features(im_rot_ten)
    orig_feat_np = orig_feat.data.cpu().numpy()
    max_featval = orig_feat_np.max()
    norm_orig_feat = (orig_feat_np / max_featval * 256).astype(np.uint8)

    for i in range(norm_orig_feat.shape[1]):
        orig_feat_im = Image.fromarray(norm_orig_feat[0, i])
        orig_feat_im_rot = orig_feat_im.rotate(angle, resample=Image.BILINEAR, expand=True)
        if i == 0:
            feat_rot_shape = list(orig_feat.shape)
            feat_rot_shape[2:] = np.array(orig_feat_im_rot).shape[:2]
            feat_rot = torch.zeros(feat_rot_shape).cuda()
        feat_rot[0, i].copy_( torch.tensor(np.array(orig_feat_im_rot)) )
    feat_rot = feat_rot * max_featval / 256
    # remove border zeros. Now 9*9
    if feat_rot.shape != rot_feat.shape:
        feat_rot = feat_rot[:,:, 1:-1, 1:-1]
        feat_rot = interpolate(feat_rot, rot_feat.shape[2:], mode='bilinear', align_corners=True)
    print( "%d: %.3f" %(angle, pearson(feat_rot, rot_feat)) )
