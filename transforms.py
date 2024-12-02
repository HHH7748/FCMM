from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image
import random
import utils
import numpy as np
import torch
import matplotlib.pyplot as plt



class FCM(object):

      def __init__(self, probability = 1.0,no_zo=False,A=5,B=0.5):
        self.probability = probability
        self.no_zo=no_zo
        self.A=A
        self.B=B
      def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x 
        x = np.array(x).astype(np.float32)
        H,W,C=x.shape
        fft_1 = np.fft.fftn(x)

        x_min=np.random.randint(W//32,W//2)
        x_max=np.random.randint(W//2,W-W//32)
        y_min=np.random.randint(H//32,H//2)
        y_max=np.random.randint(H//2,H-H//32)

        a=np.random.uniform(0,self.A)
        mask=np.ones((H,W,C))
        mask[y_min:y_max,x_min:x_max,:]=(np.random.uniform(-a,a,size=(y_max-y_min,x_max-x_min,3)))
        if not self.no_zo:
            b=np.random.uniform(0,self.B)
            indices = np.where(mask == 1.)
            num_indices = len(indices[0])
            random_values = np.random.uniform(1-b, 1+b, size=num_indices)
            mask[indices] = random_values
        fft_1*=mask
        final = np.fft.ifftn(fft_1)
        x=final.astype(np.uint8)
        x = Image.fromarray(x)

        return x
      
class FCMM(object):
      def __init__(self, img_size=32, aug_all=False, sev=3, mixing_iter=4, beta=3, FCMProb=0.5, no_zo=False, A=5, B=0.5):
        if aug_all is False:
            utils.IMAGE_SIZE = img_size
            self.aug_list = utils.augmentations
        else:
            utils.IMAGE_SIZE = img_size
            self.aug_list = utils.augmentations_all
            print('aug_all')
        
        self.sev=sev
        self.beta=beta
        self.mixing_iter=mixing_iter
        self.FCMProb=FCMProb
        self.no_zo=no_zo
        self.mixings=utils.mixings
        self.A=A
        self.B=B
      def _apply_op(self,img,op,severity):
        img = np.clip(img, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img)  # Convert to PIL.Image
        pil_img = op(pil_img, severity)
        return np.asarray(pil_img).astype(np.float32)
      
      def _FCM(self,img):

        x=img.astype(np.float32)
        H,W,C=x.shape
        fft_1 = np.fft.fftn(x)
        x_min=np.random.randint(W//32,W//2)
        x_max=np.random.randint(W//2,W-W//32)
        y_min=np.random.randint(H//32,H//2)
        y_max=np.random.randint(H//2,H-H//32) 
        mask=np.ones((H,W,C))
        a=np.random.uniform(0,self.A)
        mask[y_min:y_max,x_min:x_max,:]=np.random.uniform(-a,a,size=(y_max-y_min,x_max-x_min,3))       
        if not self.no_zo:
            indices = np.where(mask == 1.)
            num_indices = len(indices[0])
            b=np.random.uniform(0,self.B)
            random_values = np.random.uniform(1-b, 1+b, size=num_indices)
            mask[indices] = random_values
        fft_1*=mask
        final=np.fft.ifftn(fft_1).real
        final=final.astype(np.float32)
        return final  
      
      def __call__(self, x):
        img=np.array(x).astype(np.float32)
        if np.random.random() < 0.5:
            img_aug=img.copy()
            op = np.random.choice(self.aug_list)
            mixed = self._apply_op(img_aug,op,self.sev)
        else:
            mixed=img.copy()
        if np.random.random() < self.FCMProb:
            mcp=mixed.copy()
            mixed = self._FCM(mcp)
        for _ in range(np.random.randint(self.mixing_iter+1)):
            if np.random.random() <= 1.:
                img_aug=img.copy()
                op = np.random.choice(self.aug_list)
                aug_img_copy=self._apply_op(img_aug,op,self.sev)
            else:
                aug_img_copy = img.copy()            
            if np.random.random() < self.FCMProb:
                aic=aug_img_copy.copy()
                aug_img_copy = self._FCM(aic)
            mixed_op = np.random.choice(self.mixings)   
            mixed = mixed_op(torch.from_numpy(mixed/255.),torch.from_numpy(aug_img_copy/255.), self.beta)
            mixed=mixed.clip(0,1)
            mixed=mixed.numpy()*255.
        x = mixed.astype(np.uint8)
        x = Image.fromarray(x)

        return x
class StandardAug(object):
    def __init__(self, img_size=32, aug_all=False, sev=3):
        if aug_all is False:
            utils.IMAGE_SIZE = img_size
            self.aug_list = utils.augmentations
        else:
            utils.IMAGE_SIZE = img_size
            self.aug_list = utils.augmentations_all
            print('aug_all')
        self.sev=sev

    def __call__(self, x):
        '''
        :param img: (PIL Image): Image
        :return: code img (PIL Image): Image
        '''
        op = np.random.choice(self.aug_list)
        x = op(x, sev)

        return x


