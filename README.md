# Feature Lenses: Plug-and-play Neural Modules for Transformation-Invariant Visual Representations
This is the code for the arxiv paper "Feature Lenses: Plug-and-play Neural Modules for Transformation-Invariant Visual Representations". 

1. **featlens.py** The library implementing Feature Lenses.

2. **train.py** The code to train and evaluate Lenses and Xlayer. Supports ImageNet, MNIST-rot and rotated CIFAR-10.

3. **train2.sh** The training script on two GPUs for Lenses or Xlayer.

4. **resnet.py** Modified code of ResNet, to accommodate with Lenses or Xlayer (controlled by initialization arguments).

5. **train-dataaug.py** Training and evaluation code of ResNet and DataAug. Supports ImageNet and MNIST.

6. **cifar10.py** Training and evaluation code of ResNet and DataAug on CIFAR-10.

7. **prepare\_data.py** The script to create ImageNet LMDB databases (slightly faster than loading from individual image files).

8. **lmdbloader.py** The library to load and augment ImageNet LMDB databases.
    