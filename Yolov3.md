## Quick Glance ##
```
1. The "stride" is totally decided by the model struct, as the output receptive field of the model. E.g.,
with the official yolov3 struct, the strides, also known as the output receptive fields are: 8, 16 and 32.
It has nothing to do with input image or anchor box size, it's an nature/essential property of the model.
----------------------------------------------------------------------------------------------------------
Thus, one can input images with different kinds of resolutions into the model, and get the output objects.
Pay attention here, the min input size should be large than the max stride, e.g. 32 pixel for official yolo.
The only difference is there would be fewer large/middle objects, and the large objects in the origin big
images would be detected as middle or small objects. For example, with official yolov3 model,
==========================================================================================================
 resolutions | feature map shape(stride: 8, 16, 32) | comments
----------------------------------------------------------------------------------------------------------
  640x640x3  |  3x85x80x80, 3x85x40x40, 3x85x20x20  | 1) take time;
  320x320x3  |  3x85x40x40, 3x85x20x20, 3x85x10x10  | 1) much faster since feature map shape is 1/4;
==========================================================================================================
So the feature map shape is actually the grid cell's count, e.g. 3x85x80x80 => total 80x80 grid cells.
If we need to handle rather small images, then we'd better update the model struct, e.g. add new layer/block
struct for extracting feature maps of stride: 4.

2. The "anchor box" is totally decided by the input image datasets. It has nothing to do with the model struct
except the model performance. It's an nature/essential property of the dataset. Once a model is under training,
the related "anchor box" is fixed. For yolov3, each feature point has 3 different shapes of anchor box(based on
image size 640x640 with COCO dataset).

3. For yolov3(pytorch), the predicted offsets for bbox are scaled by 1/2, and shifted by +0.5 .
(0,0) +-------+ (0, 1)                   (dx, dy)
      | _____ |                 predicted       origin
      | |   | |                 (0.50, 0.50) -> (0.50, 0.50)
      | | . | | <- (0.5, 0.5)   (0.25, 0.25) -> (0.00, 0.00)
      | |   | |                 (0.75, 0.75) -> (1.00, 1.00)
      | ¯¯¯¯¯ |                 (0.00, 0.00) -> (-0.5, -0.5)
(1,0) +-------+ (1, 1)          (1.00, 1.00) -> (1.50, 1.50)
It seems that the predicted center point can be outside the grid cell.
```

## Quick Env ##
```
## create new env(e.g. ubuntu>=16.04)
$ conda create -n yolov3 python=3.8 -y
## install requirements
$ pip install -r requirements.txt -i http://pypi.douban.com/simple --trusted-host pypi.douban.com --no-deps
## install extra packages
$ conda install typing-extensions pytz requests -y
$ conda install python-dateutil pyparsing cycler kiwisolver grpcio cachetools -y
$ pip install tensorboard==2.5.0 tensorboard-data-server tensorboard-plugin-wit>=1.6.0 -i http://pypi.douban.com/simple --trusted-host pypi.douban.com --no-deps
$ pip install markdown absl-py werkzeug pyasn1_modules pyasn1 protobuf thop -i http://pypi.douban.com/simple --trusted-host pypi.douban.com --no-deps
$ pip install google-auth google-auth-oauthlib requests-oauthlib rsa oauthlib -i http://pypi.douban.com/simple --trusted-host pypi.douban.com --no-deps
## install pycocotools
$ tar xvzf pycocotools-2.0.2.tar.gz
$ cd pycocotools-2.0.2
$ python setup.py --help
$ python setup.py build
$ python setup.py install

* Note:
1) For ubuntu 14.04, one may need to do more works;
```

## Quick Start ##
```
## download pretrained weights(v9.5.0)
$ export MODELS="yolov3-tiny yolov3-spp yolov3 yolov5l"
$ for m in $MODELS; do wget https://github.com/ultralytics/yolov3/releases/download/v9.5.0/${m}.pt

## detect
$ export MODELS="yolov3-tiny yolov3-spp yolov3 yolov5l"
$ export OPTIONS="--source data/images --img-size 640 --save-txt --save-conf"
$ for m in $MODELS; do python detect.py --weights weights/${m}.pt --project runs/detect/${m} $OPTIONS

## test
$ export MODELS="yolov3-tiny yolov3-spp yolov3 yolov5l"
$ export OPTIONS="--data data/coco128.yaml --img-size 640 --verbose --save-txt --save-conf" # --single-cls
$ for m in $MODELS; do python test.py --weights weights/${m}.pt --project runs/test/${m} $OPTIONS > runs/test/${m}.txt; done

## train
$ wget https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip
$ export MODELS="yolov3-tiny yolov3-spp yolov3"
$ export OPTIONS="--data data/coco128.yaml --epochs 20 --batch-size 16 --img-size 640" # --rect --device --single-cls --adam
$ for m in $MODELS; do python train.py --weights weights/${m}.pt --cfg models/${m}.yaml --project runs/train/${m} $OPTIONS; done

* Note for Window:
> SET OPTIONS=--data data/coco128.yaml --img-size 640 --verbose --save-txt --save-conf
> echo %OPTIONS%
```

## Quick Models (Train) ##
```
* yolov3-tiny
================================================================================
                 from  n    params  module                                  arguments
  0                -1  1       464  models.common.Conv                      [3, 16, 3, 1]
  1                -1  1         0  torch.nn.modules.pooling.MaxPool2d      [2, 2, 0]
  2                -1  1      4672  models.common.Conv                      [16, 32, 3, 1]
  3                -1  1         0  torch.nn.modules.pooling.MaxPool2d      [2, 2, 0]
  4                -1  1     18560  models.common.Conv                      [32, 64, 3, 1]
  5                -1  1         0  torch.nn.modules.pooling.MaxPool2d      [2, 2, 0]
  6                -1  1     73984  models.common.Conv                      [64, 128, 3, 1]
  7                -1  1         0  torch.nn.modules.pooling.MaxPool2d      [2, 2, 0]
  8                -1  1    295424  models.common.Conv                      [128, 256, 3, 1]
  9                -1  1         0  torch.nn.modules.pooling.MaxPool2d      [2, 2, 0]
 10                -1  1   1180672  models.common.Conv                      [256, 512, 3, 1]
 11                -1  1         0  torch.nn.modules.padding.ZeroPad2d      [[0, 1, 0, 1]]
 12                -1  1         0  torch.nn.modules.pooling.MaxPool2d      [2, 1, 0]
 13                -1  1   4720640  models.common.Conv                      [512, 1024, 3, 1]
 14                -1  1    262656  models.common.Conv                      [1024, 256, 1, 1]
 15                -1  1   1180672  models.common.Conv                      [256, 512, 3, 1]
 16                -2  1     33024  models.common.Conv                      [256, 128, 1, 1]
 17                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']
 18           [-1, 8]  1         0  models.common.Concat                    [1]
 19                -1  1    885248  models.common.Conv                      [384, 256, 3, 1]
--------------------------------------------------------------------------------
 20          [19, 15]  1    196350  models.yolo.Detect                      [80, [[10, 14, 23, 27, 37, 58], [81, 82, 135, 169, 344, 319]], [256, 512]]
Model Summary: 59 layers, 8852366 parameters, 8852366 gradients, 13.3 GFLOPS
--------------------------------------------------------------------------------
 20          [19, 15]  1     13860  models.yolo.Detect                      [1, [[10, 14, 23, 27, 37, 58], [81, 82, 135, 169, 344, 319]], [256, 512]]
Model Summary: 59 layers, 8669876 parameters, 8669876 gradients, 13.0 GFLOPS
================================================================================
Transferred 66/72 items from weights/yolov3-tiny.pt
Scaled weight_decay = 0.0005625000000000001
Optimizer groups: 13 .bias, 13 conv.weight, 11 other

* yolov3
================================================================================
                 from  n    params  module                                  arguments                     
  0                -1  1       928  models.common.Conv                      [3, 32, 3, 1]                 
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     20672  models.common.Bottleneck                [64, 64]                      
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  2    164608  models.common.Bottleneck                [128, 128]                    
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  8   2627584  models.common.Bottleneck                [256, 256]                    
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  8  10498048  models.common.Bottleneck                [512, 512]                    
  9                -1  1   4720640  models.common.Conv                      [512, 1024, 3, 2]             
 10                -1  4  20983808  models.common.Bottleneck                [1024, 1024]                  
 11                -1  1   5245952  models.common.Bottleneck                [1024, 1024, False]           
 12                -1  1    525312  models.common.Conv                      [1024, 512, [1, 1]]           
 13                -1  1   4720640  models.common.Conv                      [512, 1024, 3, 1]             
 14                -1  1    525312  models.common.Conv                      [1024, 512, 1, 1]             
 15                -1  1   4720640  models.common.Conv                      [512, 1024, 3, 1]             
 16                -2  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 17                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 18           [-1, 8]  1         0  models.common.Concat                    [1]                           
 19                -1  1   1377792  models.common.Bottleneck                [768, 512, False]             
 20                -1  1   1312256  models.common.Bottleneck                [512, 512, False]             
 21                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 22                -1  1   1180672  models.common.Conv                      [256, 512, 3, 1]              
 23                -2  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 24                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 25           [-1, 6]  1         0  models.common.Concat                    [1]                           
 26                -1  1    344832  models.common.Bottleneck                [384, 256, False]             
 27                -1  2    656896  models.common.Bottleneck                [256, 256, False]             
--------------------------------------------------------------------------------
 28      [27, 22, 15]  1    457725  models.yolo.Detect                      [80, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [256, 512, 1024]]
Model Summary: 333 layers, 61949149 parameters, 61949149 gradients, 156.4 GFLOPS
--------------------------------------------------------------------------------
 28      [27, 22, 15]  1     32310  models.yolo.Detect                      [1, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [256, 512, 1024]]
Model Summary: 333 layers, 61523734 parameters, 61523734 gradients, 155.1 GFLOPS
================================================================================
Transferred 438/440 items from weights/yolov3.pt
Scaled weight_decay = 0.0005
Optimizer groups: 75 .bias, 75 conv.weight, 72 other

* yolov3-spp
================================================================================
                 from  n    params  module                                  arguments                     
  0                -1  1       928  models.common.Conv                      [3, 32, 3, 1]                 
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     20672  models.common.Bottleneck                [64, 64]                      
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  2    164608  models.common.Bottleneck                [128, 128]                    
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  8   2627584  models.common.Bottleneck                [256, 256]                    
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  8  10498048  models.common.Bottleneck                [512, 512]                    
  9                -1  1   4720640  models.common.Conv                      [512, 1024, 3, 2]             
 10                -1  4  20983808  models.common.Bottleneck                [1024, 1024]                  
 11                -1  1   5245952  models.common.Bottleneck                [1024, 1024, False]           
 12                -1  1   1574912  models.common.SPP                       [1024, 512, [5, 9, 13]]       
 13                -1  1   4720640  models.common.Conv                      [512, 1024, 3, 1]             
 14                -1  1    525312  models.common.Conv                      [1024, 512, 1, 1]             
 15                -1  1   4720640  models.common.Conv                      [512, 1024, 3, 1]             
 16                -2  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 17                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 18           [-1, 8]  1         0  models.common.Concat                    [1]                           
 19                -1  1   1377792  models.common.Bottleneck                [768, 512, False]             
 20                -1  1   1312256  models.common.Bottleneck                [512, 512, False]             
 21                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 22                -1  1   1180672  models.common.Conv                      [256, 512, 3, 1]              
 23                -2  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 24                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 25           [-1, 6]  1         0  models.common.Concat                    [1]                           
 26                -1  1    344832  models.common.Bottleneck                [384, 256, False]             
 27                -1  2    656896  models.common.Bottleneck                [256, 256, False]             
--------------------------------------------------------------------------------
 28      [27, 22, 15]  1    457725  models.yolo.Detect                      [80, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [256, 512, 1024]]
Model Summary: 342 layers, 62998749 parameters, 62998749 gradients, 157.3 GFLOPS
--------------------------------------------------------------------------------
 28      [27, 22, 15]  1     32310  models.yolo.Detect                      [1, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [256, 512, 1024]]
Model Summary: 342 layers, 62573334 parameters, 62573334 gradients, 155.9 GFLOPS
================================================================================
Transferred 444/446 items from weights/yolov3-spp.pt
Scaled weight_decay = 0.0005
Optimizer groups: 76 .bias, 76 conv.weight, 73 other
```

## Quick FAQ ##
```
1. ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.22' not found (required by /home/qinhj/.conda/envs/yolov3/lib/python3.9/site-packages/scipy/fft/_pocketfft/pypocketfft.cpython-39-x86_64-linux-gnu.so)
## install libstdc++6
$ sudo apt-get install libstdc++6
## check 'GLIBCXX_3.4.22'
$ strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
## if still not found, try:
$ sudo add-apt-repository ppa:ubuntu-toolchain-r/test
$ sudo apt-get update
$ sudo apt-get upgrade
$ sudo apt-get dist-upgrade
## check 'GLIBCXX_3.4.22' once more
$ strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX

2. higher mAP with poorly detect results
See https://github.com/ultralytics/yolov3/issues/1252
mAP is the total result to show how model performs. In fact, behind mAP are three losses: iou, cls, obj.
obj is to determine: is there a target?
cls is to classify: What the target is?
iou: How far the predicted bounding box is between ground truth
iou commonly drop slower than others, and cls and obj drop much more fast than it.
SO there is indeed a case like yours: higher mAP but poorly detect result. Just train more epoch and model will be more stable.

3. How to understand the result?
========================================================================================================================
epoch  mem  lossbox  lossobj losscls losstotal  targets img_size    P      R   mAP@.5 mAP@.5-.95 lossbox lossobj losscls
========================================================================================================================
0/29 10.9G  0.03062 0.008517       0   0.03914       79      640 0.8286 0.7928 0.8634   0.6108   0.02334 0.004774   0
...

4. How to debug model?
>>> ## load pytorch state dict
>>> import torch
>>> device = "cpu"
>>> ckpt = torch.load("weights/yolov3.pt", map_location=device)
>>> ckpt.keys()
dict_keys(['epoch', 'best_fitness', 'training_results', 'model', 'ema', 'updates', 'optimizer', 'wandb_id'])
>>> ## load models.yolo.Model(without model state dict)
>>> from models.yolo import Model
>>> model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
>>> ## load model state dict
>>> exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
>>> state_dict = ckpt['model'].float().state_dict()  # to FP32
>>> state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
>>> model.load_state_dict(state_dict, strict=False)  # load

5. How to check module?
>>> ## load models.yolo.Model(without model state dict)
>>> m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
>>> m
Detect(
  (m): ModuleList(
    (0): Conv2d(128, 18, kernel_size=(1, 1), stride=(1, 1))
    (1): Conv2d(256, 18, kernel_size=(1, 1), stride=(1, 1))
  )
)
>>> m.anchor_grid[:]
tensor([[[[[[  8.33594,  16.42188]]],
          [[[ 17.78125,  35.81250]]],
          [[[ 34.93750,  64.00000]]]]],
        [[[[[ 60.81250, 124.00000]]],
          [[[112.43750, 230.25000]]],
          [[[257.50000, 348.75000]]]]]])
```
