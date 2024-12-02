# FCM/FCMM

This is the repository for "Enhancing Safety Measures via Frequency Component Modification and Mixing"  

The proposed method has been implemented based on the official code of PixMix.


## Contents

`transforms.py` includes implementation FCM/FCMM

## Usage

Training FCMM :

CIFAR10/100: 
  ```
  python cifar.py \
    --dataset <cifar10 or cifar100> \
    --data-path <path/to/cifar and cifar-c> 
    --save <path/to/save/checkpoint and log>
    --jsd (Turn on JSD loss)
  ```

  For the implementation of FCM/StdAug+FCM, please change train_transform in the main function of cifar.py


ImageNet: 
  ```
  python imagenet.py \
    --data-standard <path/to/imagenet/train>
    --data-val <path/to/imagenet/val>
    --imagenet-r-dir <path/to/imagenet-r>
    --imagenet-c-dir <path/to/imagenet-c>
    --num-classes 1000
    --save <path/to/save/checkpoint and log>
    --batch-size 512
    --pretrained (Fine-tune pretrained model)
    --jsd (Turn on JSD loss)
  ```

