# Person Detection: Yolov3 #
[SINGLE-CLASS TRAINING EXAMPLE](https://github.com/ultralytics/yolov3/issues/102)  
```
==============================================================================
    model   | device | GPU Memory |    VmPeak   | Threads | Cpus_allowed_list
==============================================================================
   yolov3   | CUDA:0 |  13427MiB  | 12509484 kB |    3    |        0-3
------------------------------------------------------------------------------
yolov3-tiny |  CPU   |  --------  |  9389260 kB |    34   |        0-7
==============================================================================
```

## Test Benchmark ##
```
## quick commands
$ export MODELS="yolov3-tiny yolov3-spp yolov3 yolov5l"
$ export OPTIONS="--data data/coco5k.yaml --img-size 640 --verbose --save-txt --save-conf"
$ for m in $MODELS; do python test.py --weights weights/${m}.pt --project runs/test/${m} $OPTIONS > runs/test/${m}.txt; done

## runtime Namespace(others)
batch_size=32, conf_thres=0.001, iou_thres=0.6, task='val', single_cls=False, augment=False,
verbose=True, save_txt=True, save_hybrid=False, save_conf=True, save_json=False, exist_ok=False

## output results
===========================================================================================================================================
    model   | GPU Memory | Layers | Parameters | Gradients |  FLOPS  | Precision    Recall  mAP@0.5 mAP@0.5:0.95
===========================================================================================================================================
yolov3-tiny | 3965MiB    |  48    | 8,849,182  |    0      |  13.2G  |     0.624    0.543    0.561      0.261
-------------------------------------------------------------------------------------------------------------------------------------------
yolov3-spp  | 12499MiB   |  269   | 62,971,933 |    0      | 157.1G  |     0.8      0.734    0.799      0.529
-------------------------------------------------------------------------------------------------------------------------------------------
yolov3      | 7703MiB    |  261   | 61,922,845 |    0      | 156.3G  |     0.776    0.741    0.791      0.524
-------------------------------------------------------------------------------------------------------------------------------------------
yolov5l     | 7275MiB    |  392   | 47,025,981 |    0      | 115.4G  |     0.832    0.727    0.809      0.565
===========================================================================================================================================
```

## Test Case 0: scratch without any pretrained weights (?: maybe need more epochs, e.g. 50) ##
```
===========================================================================================================================================
      model     | --device |        --weights       |          --cfg          |         --hyp         | Precision Recall mAP@0.5 mAP@0.5:0.95
===========================================================================================================================================
(?) yolov3-tiny |  CUDA:0  | weights/yolov3-tiny.pt | models/yolov3-tiny.yaml | data/hyp.scratch.yaml |   0.5907  0.4692  0.4989  0.2195<-- ^C (↓  7/299)
-------------------------------------------------------------------------------------------------------------------------------------------
(?) yolov3      |  CUDA:0  | weights/yolov3.pt      | models/yolov3.yaml      | data/hyp.scratch.yaml |   0.7054  0.6422  0.6891  0.4105<-- ^C (↓ 14/299)
===========================================================================================================================================

## train model
$ MODEL=yolov3 # yolov3-tiny
$ OPTIONS="--data data/coco_person.yaml --single-cls --epochs 300 --batch-size 24"
$ python train.py --weights '' --cfg models/$MODEL.yaml --project runs/train/$MODEL $OPTIONS
```

## Test Case 1: scratch/finetune with official weights (X: optimize too slow) ##
```
$ MODEL=yolov3-tiny
===========================================================================================================================================
yolov3-tiny | --device |    --weights           |       --cfg             |         --hyp           | Precision Recall mAP@0.5 mAP@0.5:0.95
===========================================================================================================================================
(↑) case1   |   CPU    | weights/yolov3-tiny.pt | models/yolov3-tiny.yaml | data/hyp.scratch.yaml   |   0.6656  0.5055  0.5637  0.2645  <-- ^C (↑  5/299)
-------------------------------------------------------------------------------------------------------------------------------------------
(↑) case2   |  CUDA:0  | weights/yolov3-tiny.pt | models/yolov3-tiny.yaml | data/hyp.finetune.yaml  |   0.6428  0.5311  0.5671  0.2683  <-- ^C (↑ 10/299)
===========================================================================================================================================

$ MODEL=yolov3 # degeneration ??
===========================================================================================================================================
    yolov3  | --device |    --weights           |       --cfg             |         --hyp           | Precision Recall mAP@0.5 mAP@0.5:0.95
===========================================================================================================================================
(↓) case1   |  CUDA:0  | weights/yolov3.pt      | models/yolov3.yaml      | data/hyp.scratch.yaml   |   0.767   0.7066  0.7728  0.505 ??<-- ^C (↓ 14/299)
-------------------------------------------------------------------------------------------------------------------------------------------
(↑) case2   |  CUDA:0  | weights/yolov3.pt      | models/yolov3.yaml      | data/hyp.finetune.yaml  |   0.7896  0.7245  0.7912  0.5256  <-- ^C (↑ 10/299)
===========================================================================================================================================

## train model
$ OPTIONS="--data data/coco_person.yaml --single-cls --epochs 300 --batch-size 24"
$ python train.py --weights weights/$MODEL.pt --cfg models/$MODEL.yaml --project runs/train/$MODEL $OPTIONS --hyp data/hyp.scratch.yaml
$ python train.py --weights weights/$MODEL.pt --cfg models/$MODEL.yaml --project runs/train/$MODEL $OPTIONS --hyp data/hyp.finetune.yaml

* Common Options
data: data/coco_person.yaml
epochs: 300
batch_size: 24
img_size: [640, 640]
multi_scale: false
single_cls: true
adam: false
```

## Test Case 2: scratch with warmup weights on val data (Y: seems overfit a lot) ##
```
$ MODEL=yolov3-tiny
===========================================================================================================================================
yolov3-tiny | --epochs | --device |       --weights         |    --cfg    |         --data          | Precision Recall mAP@0.5 mAP@0.5:0.95
===========================================================================================================================================
(↑) step1   |   20     |   CPU    | weights/yolov3-tiny.pt  |             | data/coco_person5k.yaml |   0.645   0.6138  0.6356  0.3018  <-- Overfit(maybe)
------------------------------------------------------------| models/     |----------------------------------------------------------------
(↑) step2   |   30     |   CPU    | warmup/weights/best.pt  | yolov3-tiny | data/coco_person.yaml   |   0.6476  0.5703  0.6005  0.297   <-- Fin(↓ 29/29) 
------------------------------------------------------------| .yaml       |----------------------------------------------------------------
(x) step3   |   20     |   CPU    | epoch30/weights/last.pt |             | data/coco_person.yaml   |   0.6457  0.5606  0.5953  0.2921  <-- ^C (↓  0/19)
===========================================================================================================================================

$ MODEL=yolov3
===========================================================================================================================================
    yolov3  | --epochs | --device |       --weights         |    --cfg    |         --data          | Precision Recall mAP@0.5 mAP@0.5:0.95
===========================================================================================================================================
(↑) step1   |   20     |  CUDA:0  | weights/yolov3.pt       |             | data/coco_person5k.yaml |   0.8455  0.8056  0.8777  0.6187  <-- Overfit(maybe)
------------------------------------------------------------| models/     |----------------------------------------------------------------
(↑) step2   |   30     |  CUDA:0  | warmup/weights/best.pt  | yolov3.yaml | data/coco_person.yaml   |   0.7958  0.7223  0.799   0.5361  <-- Fin(↓ 29/29)
------------------------------------------------------------|             |----------------------------------------------------------------
(x) step3   |   20     |  CUDA:0  | epoch30/weights/last.pt |             | data/coco_person.yaml   |   0.7685  0.682   0.7573  0.4815  <-- ^C (↓  2/19)
===========================================================================================================================================

* Common Options
hyp: data/hyp.scratch.yaml  # warmup_epochs: 3.0
batch_size: 24
img_size: [640, 640]
multi_scale: false
single_cls: true
adam: false

* Results:
1) step1 may overfit even only with "--epochs 20", one may try interrupt and "--resume";
2) step2 may have some worse results on the first a few epochs(0~4), maybe because step1's overfit;
```

## Test Case 3: finetune with warmup weights on val data (X: finetune has only few gain) ##
```
$ MODEL=yolov3-tiny
===========================================================================================================================================
yolov3-tiny | --epochs |       --weights         |         --hyp          |         --data          | Precision Recall mAP@0.5 mAP@0.5:0.95
===========================================================================================================================================
(↑) step1   |   20     | weights/yolov3-tiny.pt  | data/hyp.scratch.yaml  | data/coco_person5k.yaml |   0.6733  0.5947  0.6351  0.3014  <-- Overfit(maybe)
-------------------------------------------------------------------------------------------------------------------------------------------
(↑) step2   |   30     | warmup/weights/best.pt  | data/hyp.finetune.yaml | data/coco_person.yaml   |   0.6564  0.5771  0.6138  0.3042  <-- Fin(↑ 28/29)
===========================================================================================================================================

## train model
$ OPTIONS="--cfg models/$MODEL.yaml --single-cls --batch-size 24 --project runs/train/$MODEL"
$ python train.py --weights weights/${MODEL}.pt --hyp data/hyp.scratch.yaml --data data/coco_person5k.yaml --epochs 20 $OPTIONS
$ python train.py --weights warmup/weights/best.pt --hyp data/hyp.finetune.yaml --data data/coco_person.yaml --epochs 30 $OPTIONS

* Common Options
cfg: models/yolov3-tiny.yaml
batch_size: 24
img_size: [640, 640]
device: ''  # as CUDA:0
multi_scale: false
single_cls: true
adam: false

* Results
1) step1 may overfit even only with "--epochs 20", one may try interrupt and "--resume";
2) step2 may have only few gain;
```

## Test Case 4: frozen some lower layers(e.g. bottleneck) ##
```
TODO
```

## Test Results ##
```
1) scratch on val data => scratch on train data; ???
2) finetune with pretrained weights directly;
```