import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as trn
import torchvision.transforms.functional as trn_F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
'''
from resnext_50_32x4d import resnext_50_32x4d
from resnext_101_32x4d import resnext_101_32x4d
from resnext_101_64x4d import resnext_101_64x4d
'''
from scipy.stats import rankdata
import torchvision.models as models


import cv2
# from skvideo.io import VideoCapture
# import skvideo.io
import torch.utils.data as data
from torchvision.datasets.folder import DatasetFolder
from PIL import Image

import os
import os.path
import sys


class VideoFolder(DatasetFolder):

    def __init__(self, root, transform=None, target_transform=None, loader=None):
        super(VideoFolder, self).__init__(
            root, loader, ['.mp4'], transform=transform, target_transform=target_transform)

        self.vids = self.samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        # cap = VideoCapture(path)
        cap = cv2.VideoCapture(path)

        frames = []

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret: break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(Image.fromarray(frame)).unsqueeze(0))

        cap.release()

        return torch.cat(frames, 0), target

parser = argparse.ArgumentParser(description='Evaluates robustness of various nets on ImageNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Architecture

parser.add_argument('--perturbation', '-p', default='brightness', type=str,
                    choices=['gaussian_noise', 'shot_noise', 'motion_blur', 'zoom_blur',
                             'spatter', 'brightness', 'translate', 'rotate', 'tilt', 'scale',
                             'speckle_noise', 'gaussian_blur', 'snow', 'shear'])
parser.add_argument('--difficulty', '-d', type=int, default=1, choices=[1, 2, 3])
# Acceleration
parser.add_argument(
    '--resume',
    '-r',
    type=str,
    default='',
    help='Checkpoint path for resume / test.')
args = parser.parse_args()
print(args)

# /////////////// Model Setup ///////////////
net = models.resnet50(pretrained=False)

net = torch.nn.DataParallel(net).cuda()
cudnn.benchmark = True
if args.resume:
    if os.path.isfile(args.resume):
      checkpoint = torch.load(args.resume)
      start_epoch = checkpoint['epoch'] + 1
      #best_acc = checkpoint['best_acc']
      net.load_state_dict(checkpoint['state_dict'])
      #optimizer.load_state_dict(checkpoint['optimizer'])
      #net.load_state_dict(checkpoint)
      print('Model restored from epoch:', start_epoch)
args.prefetch = 4

args.test_bs = 4
torch.manual_seed(1)
np.random.seed(1)



net.eval()
cudnn.benchmark = True  # fire on all cylinders

print('Model Loaded\n')

# /////////////// Data Loader ///////////////
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
#mean = [0.5]*3
#std = [0.5]*3
if args.difficulty > 1 and 'noise' in args.perturbation:
    loader = torch.utils.data.DataLoader(
        VideoFolder(root="/DATA/imagenet-p/" +
                         args.perturbation + '_' + str(args.difficulty),
                    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])),
        batch_size=args.test_bs, shuffle=False, num_workers=5, pin_memory=True)
else:
    loader = torch.utils.data.DataLoader(
        VideoFolder(root="/DATA/imagenet-p/" + args.perturbation,
                    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])),
        batch_size=args.test_bs, shuffle=False, num_workers=5, pin_memory=True)

print('Data Loaded\n')


# /////////////// Stability Measurements ///////////////

identity = np.asarray(range(1, 1001))
cum_sum_top5 = np.cumsum(np.asarray([0] + [1] * 5 + [0] * (999 - 5)))
recip = 1./identity

# def top5_dist(sigma):
#     result = 0
#     for i in range(1,6):
#         for j in range(min(sigma[i-1], i) + 1, max(sigma[i-1], i) + 1):
#             if 1 <= j - 1 <= 5:
#                 result += 1
#     return result

def dist(sigma, mode='top5'):
    if mode == 'top5':
        return np.sum(np.abs(cum_sum_top5[:5] - cum_sum_top5[sigma-1][:5]))
    elif mode == 'zipf':
        return np.sum(np.abs(recip - recip[sigma-1])*recip)


def ranking_dist(ranks, noise_perturbation=True if 'noise' in args.perturbation else False, mode='top5'):
    result = 0
    step_size = 1 if noise_perturbation else args.difficulty

    for vid_ranks in ranks:
        result_for_vid = []

        for i in range(step_size):
            perm1 = vid_ranks[i]
            perm1_inv = np.argsort(perm1)

            for rank in vid_ranks[i::step_size][1:]:
                perm2 = rank
                result_for_vid.append(dist(perm2[perm1_inv], mode))
                if not noise_perturbation:
                    perm1 = perm2
                    perm1_inv = np.argsort(perm1)

        result += np.mean(result_for_vid) / len(ranks)

    return result


def flip_prob(predictions, noise_perturbation=True if 'noise' in args.perturbation else False):
    result = 0
    step_size = 1 if noise_perturbation else args.difficulty

    for vid_preds in predictions:
        result_for_vid = []

        for i in range(step_size):
            prev_pred = vid_preds[i]

            for pred in vid_preds[i::step_size][1:]:
                result_for_vid.append(int(prev_pred != pred))
                if not noise_perturbation: prev_pred = pred

        result += np.mean(result_for_vid) / len(predictions)

    return result


# /////////////// Get Results ///////////////

from tqdm import tqdm

predictions, ranks = [], []

with torch.no_grad():

    for data, target in loader:
        num_vids = data.size(0)
        data = data.view(-1,3,224,224).cuda()

        output = net(data)

        for vid in output.view(num_vids, -1, 1000):
            predictions.append(vid.argmax(1).to('cpu').numpy())
            ranks.append([np.uint16(rankdata(-frame, method='ordinal')) for frame in vid.to('cpu').numpy()])


ranks = np.asarray(ranks)

print('Computing Metrics\n')
FR, T5D, Zf= flip_prob(predictions),ranking_dist(ranks, mode='top5'),ranking_dist(ranks, mode='zipf')
print('Flipping Prob\t{:.5f}'.format(FR))
print('Top5 Distance\t{:.5f}'.format(T5D))
print('Zipf Distance\t{:.5f}'.format(Zf))
print(FR,T5D,Zf)
