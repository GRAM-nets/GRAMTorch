# GRAMTorch: PyTorch implementation of GRAM-nets

This example implements the paper [Generative Ratio Matching Networks](https://openreview.net/forum?id=SJg7spEYDS).

After every 100 training iterations, the files `$DATASET-$MODEL-data.png` and `$DATASET-$MODEL-samples-epoch=$EPOCH.png` are written to disk with the samples from the generative model.

After every epoch, models are saved to: `$DATASET-$MODEL-netG-epoch=$EPOCH.pth` and `$DATASET-$MODEL-netD-epoch=$EPOCH.pth`

## Downloading the dataset
You can download the LSUN dataset by cloning [this repo](https://github.com/fyu/lsun) and running
```
python download.py -c bedroom
```

## Usage
```
usage: main.py [-h] --dataset DATASET --model MODEL --dataroot DATAROOT
               [--workers WORKERS] [--batchSize BATCHSIZE]
               [--showSize SHOWSIZE] [--imageSize IMAGESIZE] [--nz NZ]
               [--nk NK] [--ngf NGF] [--ncf NCF] [--n_epochs N_EPOCHS]
               [--lr LR] [--beta1 BETA1] [--clip_ratio]
               [--eps_ratio EPS_RATIO] [--cuda] [--gpu_id GPU_ID]
               [--ngpu NGPU] [--netG NETG] [--netD NETD] [--netF NETF]
               [--outf OUTF] [--monitor_heuristic] [--manualSeed MANUALSEED]
               [--nowandb] [--classes CLASSES]

optional arguments:
  -h, --help              show this help message and exit
  --dataset DATASET       cifar10 | lsun | mnist |imagenet | folder | lfw | fake
  --model MODEL           gramnet | gan
  --dataroot DATAROOT     path to dataset
  --workers WORKERS       number of data loading workers
  --batchSize BATCHSIZE   input batch size
  --showSize SHOWSIZE     size of display batch
  --imageSize IMAGESIZE   the height / width of the input image to network
  --nz NZ                 size of the latent z vector
  --nk NK                 size of the projected k vector
  --ngf NGF 
  --ncf NCF 
  --n_epochs N_EPOCHS     number of epochs to train for
  --lr LR                 learning rate, default=0.0001
  --beta1 BETA1           beta1 for adam. default=0.5
  --clip_ratio            apply ratio clipping as suggested by one of the reviewer
  --eps_ratio EPS_RATIO   add eps to the diagonal before solving
  --cuda                  enables cuda
  --gpu_id GPU_ID         default GPU ID to use
  --ngpu NGPU             number of GPUs to use
  --netG NETG             path to netG (to continue training)
  --netD NETD             path to netD (to continue training)
  --netF NETF             path to netF (to continue training)
  --outf OUTF             folder to output images and model checkpoints
  --monitor_heuristic     monitor heuristic σ
  --manualSeed MANUALSEED manual seed
  --nowandb               disables wandb
  --classes CLASSES       comma separated list of classes for the lsun data set
```

Note that by default [Weights & Biases](https://www.wandb.ai/) is used for logging, 
but you can disable it by `--nowandb`.

## Acknowledgement

The code is heavily based on [the DCGAN example of PyTorch](https://github.com/pytorch/examples/tree/master/dcgan).
