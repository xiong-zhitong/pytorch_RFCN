# Region-based FCN implemented with PyTorch
This is a [PyTorch](https://github.com/pytorch/pytorch)
implementation of R-FCN for object detection. 
This project is mainly based on [faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch)

For details about R-FCN please refer to the [paper](https://arxiv.org/abs/1605.06409)
R-FCN: Object Detection via Region-based Fully Convolutional Networks
by Jifeng Dai, Yi Li, Kaiming He, Jian Sun.

### Installation
1. Clone the this repository
    ```bash
    git clone https://github.com/PureDiors/pytorch_RFCN.git
    ```

2. Build the Cython modules for nms and the psroi_pooling layer
    ```bash
    cd faster_rcnn_pytorch/faster_rcnn
    ./make.sh
    ```

The psroi pooling layer is defined in the folder faster_rcnn/psroi_pooling,
and the whole detection network is implemented in faster_rcnn/rfcn.py.


### Training on Pascal VOC 2007

Follow [this project (TFFRCNN)](https://github.com/CharlesShang/TFFRCNN)
to download and prepare the training, validation, test data 
and the VGG16 model pre-trained on ImageNet. 

Since the program loading the data in `faster_rcnn_pytorch/data` by default,
you can set the data path as following.
```bash
cd faster_rcnn_pytorch
mkdir data
cd data
ln -s $VOCdevkit VOCdevkit2007
```


The speed for training the rfcn with VGG16 on a Nvidia Titan X(Pascal) is 4.9 fps, and 12 fps for testing.

You can set some hyper-parameters in `train.py` and training parameters in the `.yml` file.


### Training with TensorBoard
With the aid of [Crayon](https://github.com/torrvision/crayon),
we can access the visualisation power of TensorBoard for any 
deep learning framework.

To use the TensorBoard, install Crayon (https://github.com/torrvision/crayon)
and set `use_tensorboard = True` in `faster_rcnn/train.py`.

### Evaluation
Set the path of the trained model in `test.py`.
```bash
cd faster_rcnn_pytorch
mkdir output
python test.py
```
