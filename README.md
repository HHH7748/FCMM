# Improving Model Robustness with Frequency Component Modification and Mixing (IEEE Access)

This is the repository for "Enhancing Safety Measures via Frequency Component Modification and Mixing"  

The proposed method has been implemented based on the official code of [PixMix](https://github.com/andyzoujm/pixmix/tree/main).


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

## Pretrained Model
### CIFAR10
[FCMM](https://drive.google.com/drive/folders/1yZtj0qJO-R5z_51kp-_D9i5gi6BaY-8U?usp=drive_link)

[FCMM+JSD](https://drive.google.com/drive/folders/1-4qt1k7-ttVxxx0MWtV_zqbyWob1cJfE?usp=drive_link)
### CIFAR100
[FCMM](https://drive.google.com/drive/folders/1IBEs8CIAfCuNxbdR0d8SgNSEGE6Kenqi?usp=drive_link)

[FCMM+JSD](https://drive.google.com/drive/folders/16O8eeWqJaC1_JGC6-VTGA0sXiQ0k9F6a?usp=drive_link)
### ImageNet
[FCMM](https://drive.google.com/drive/folders/1NBIf2nhQUk3YtoND6EWfgVksj56zTWnh?usp=drive_link)

[FCMM+JSD](https://drive.google.com/drive/folders/1lURevGgfIsYhimSrujDYbYEx82fvQFfK?usp=drive_link)
