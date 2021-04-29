# Imilab: smoke test with Yolov3 #

## Quick Note ##
```
1) The options can be overwritten by the last same one;
2) For Window, try:
> SET OPTIONS=--data data/coco128.yaml --img-size 640 --verbose --save-txt --save-conf
> echo %OPTIONS%
3) It seems that training on the val data as warmup really helpful, but one must pay attention to the overfit;
4) One may always try "--epoch 300", and "--resume ${WEIGHTS}/last.pt" with Ctrl + C;
```

## Quick Convert ##
[Convert Toolkit](https://convertmodel.com/#outputFormat=tengine)  

## Quick Train (as warmup) ##
```
* Note:
1) It seems that with "--multi-scale", the "--batch-size" can only be 8 under only one Tesla T4(15109.75MB);
2) After doing some smoke test, it seems that "--adam" isn't a good choise to work with official pretrained weights;
(i.e. to use pretrained weights, one must use the same opt.yaml as pretrained weights used)
3) Maybe we shouldn't use the vol2017.txt as warmup training dataset, which could result in
the warmupped weights is always the best one;

$ export OPTIONS="--data data/coco_person5k.yaml --epochs 20 --batch-size 24 --img-size 640 --single-cls"
$ export OPTIONS="--data data/coco_person5k.yaml --epochs 20 --batch-size 8 --img-size 640 --multi-scale --single-cls"

================================================================================
$ python train.py --weights weights/yolov3-tiny.pt --cfg models/yolov3-tiny.yaml --project runs/train/yolov3-tiny $OPTIONS
--------------------------------------------------------------------------------
Namespace(weights='weights/yolov3-tiny.pt', cfg='models/yolov3-tiny.yaml', data='data/coco_person5k.yaml', hyp='data/hyp.scratch.yaml', epochs=20, batch_size=24, img_size=[640, 640], rect=False, resume=False, nosave=False, notest=False, noautoanchor=False, evolve=False, bucket='', cache_images=False, image_weights=False, device='', multi_scale=False, single_cls=False, adam=False, sync_bn=False, local_rank=-1, workers=8, project='runs/train/yolov3-tiny', entity=None, name='exp', exist_ok=False, quad=False, linear_lr=False, label_smoothing=0.0, upload_dataset=False, bbox_interval=-1, save_period=-1, artifact_alias='latest', world_size=1, global_rank=-1, save_dir='runs/train/yolov3-tiny/exp3', total_batch_size=24)
tensorboard: Start with 'tensorboard --logdir runs/train/yolov3-tiny', view at http://localhost:6006/
hyperparameters: lr0=0.01, lrf=0.2, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0
--------------------------------------------------------------------------------
autoanchor: Analyzing anchors... anchors/target = 2.97, Best Possible Recall (BPR) = 0.9953
Image sizes 640 train, 640 test                                     -----------+
Using 4 dataloader workers                                          GPU Memory |
Logging results to runs/train/yolov3-tiny/exp3                         ========|
Starting training for 20 epochs...                                     3649MiB |
-------------------------------------------------------------------------------+
     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
      0/19     2.47G    0.0811    0.0394         0    0.1205        18       640: 100%|¨€| 209/209 [01:00<00:00,  3.46it/s]
...

================================================================================
$ python train.py --weights weights/yolov3.pt --cfg models/yolov3.yaml --project runs/train/yolov3 $OPTIONS
--------------------------------------------------------------------------------
Namespace(weights='weights/yolov3.pt', cfg='models/yolov3.yaml', data='data/coco_person5k.yaml', hyp='data/hyp.scratch.yaml', epochs=20, batch_size=24, img_size=[640, 640], rect=False, resume=False, nosave=False, notest=False, noautoanchor=False, evolve=False, bucket='', cache_images=False, image_weights=False, device='', multi_scale=False, single_cls=True, adam=False, sync_bn=False, local_rank=-1, workers=8, project='runs/train/yolov3', entity=None, name='exp', exist_ok=False, quad=False, linear_lr=False, label_smoothing=0.0, upload_dataset=False, bbox_interval=-1, save_period=-1, artifact_alias='latest', world_size=1, global_rank=-1, save_dir='runs/train/yolov3/exp2', total_batch_size=24)
tensorboard: Start with 'tensorboard --logdir runs/train/yolov3', view at http://localhost:6006/
hyperparameters: lr0=0.01, lrf=0.2, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0
--------------------------------------------------------------------------------
autoanchor: Analyzing anchors... anchors/target = 4.45, Best Possible Recall (BPR) = 0.9962
Image sizes 640 train, 640 test                                     -----------+
Using 4 dataloader workers                                          GPU Memory |
Logging results to runs/train/yolov3/exp2                              ========|
Starting training for 20 epochs...                                    13283MiB |
-------------------------------------------------------------------------------+
     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
      0/19     12.8G   0.07366   0.01961         0   0.09327        56       640:  60%|¨€¨‡| 125/209 [02:39<01:42,  1.22s/it]
...

================================================================================
$ python train.py --weights weights/yolov3-spp.pt --cfg models/yolov3-spp.yaml --project runs/train/yolov3-spp $OPTIONS --batch-size 16 // maybe we should try "--batch-size 20"
--------------------------------------------------------------------------------
Namespace(weights='weights/yolov3-spp.pt', cfg='models/yolov3-spp.yaml', data='data/coco_person5k.yaml', hyp='data/hyp.scratch.yaml', epochs=20, batch_size=16, img_size=[640, 640], rect=False, resume=False, nosave=False, notest=False, noautoanchor=False, evolve=False, bucket='', cache_images=False, image_weights=False, device='', multi_scale=False, single_cls=True, adam=False, sync_bn=False, local_rank=-1, workers=8, project='runs/train/yolov3-spp', entity=None, name='exp', exist_ok=False, quad=False, linear_lr=False, label_smoothing=0.0, upload_dataset=False, bbox_interval=-1, save_period=-1, artifact_alias='latest', world_size=1, global_rank=-1, save_dir='runs/train/yolov3-spp/exp2', total_batch_size=16)
tensorboard: Start with 'tensorboard --logdir runs/train/yolov3-spp', view at http://localhost:6006/
hyperparameters: lr0=0.01, lrf=0.2, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0
--------------------------------------------------------------------------------
autoanchor: Analyzing anchors... anchors/target = 4.45, Best Possible Recall (BPR) = 0.9962
Image sizes 640 train, 640 test                                     -----------+
Using 4 dataloader workers                                          GPU Memory |
Logging results to runs/train/yolov3-spp/exp2                          ========|
Starting training for 20 epochs...                                     9659MiB |
-------------------------------------------------------------------------------+
     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
      0/19     9.02G    0.0753   0.01923         0   0.09453        67       640:  47%|¨ | 147/313 [02:09<02:20,  1.18it/s]
...
```

## Quick Resume (from an interrupted run) ##
```
$ python train.py --cfg models/yolov3.yaml --project runs/train/yolov3 --resume runs/train/yolov3/imilab/epoch300/weights/last.pt
```

## Quick Detect ##
```
$ export OPTIONS="--source data/images --img-size 640 --save-txt --save-conf"

================================================================================
$ python detect.py --weights runs/train/yolov3-tiny/exp3/weights/best.pt --project runs/detect/yolov3-tiny $OPTIONS
--------------------------------------------------------------------------------
Namespace(weights=['runs/train/yolov3-tiny/exp3/weights/best.pt'], source='data/images', img_size=640, conf_thres=0.25, iou_thres=0.45, device='', view_img=False, save_txt=True, save_conf=True, nosave=False, classes=None, agnostic_nms=False, augment=False, update=False, project='runs/detect/yolov3-tiny', name='exp', exist_ok=False)
YOLOv3 ?? v9.5.0-7-gc64b7f1 torch 1.8.1+cu102 CUDA:0 (Tesla T4, 15109.75MB)
--------------------------------------------------------------------------------
Fusing layers...
Model Summary: 48 layers, 8666692 parameters, 0 gradients, 12.9 GFLOPS
image 1/2 /opt/workshop/qinhj/yolov3/data/images/bus.jpg: 640x480 3 persons, Done. (0.006s)
image 2/2 /opt/workshop/qinhj/yolov3/data/images/zidane.jpg: 384x640 2 persons, Done. (0.005s)
Results saved to runs/detect/yolov3-tiny/exp2
2 labels saved to runs/detect/yolov3-tiny/exp2/labels
Done. (0.093s)

================================================================================
$ python detect.py --weights runs/train/yolov3/exp2/weights/best.pt --project runs/detect/yolov3 $OPTIONS
--------------------------------------------------------------------------------
Namespace(weights=['runs/train/yolov3/exp2/weights/best.pt'], source='data/images', img_size=640, conf_thres=0.25, iou_thres=0.45, device='', view_img=False, save_txt=True, save_conf=True, nosave=False, classes=None, agnostic_nms=False, augment=False, update=False, project='runs/detect/yolov3', name='exp', exist_ok=False)
YOLOv3 ?? v9.5.0-7-gc64b7f1 torch 1.8.1+cu102 CUDA:0 (Tesla T4, 15109.75MB)
--------------------------------------------------------------------------------
Fusing layers...
Model Summary: 261 layers, 61497430 parameters, 0 gradients, 154.9 GFLOPS
image 1/2 /opt/workshop/qinhj/yolov3/data/images/bus.jpg: 640x480 3 persons, Done. (0.027s)
image 2/2 /opt/workshop/qinhj/yolov3/data/images/zidane.jpg: 384x640 2 persons, Done. (0.024s)
Results saved to runs/detect/yolov3/exp2
2 labels saved to runs/detect/yolov3/exp2/labels
Done. (0.141s)

================================================================================
$ python detect.py --weights runs/train/yolov3-spp/exp2/weights/best.pt --project runs/detect/yolov3-spp $OPTIONS
--------------------------------------------------------------------------------
Namespace(weights=['runs/train/yolov3-spp/exp2/weights/best.pt'], source='data/images', img_size=640, conf_thres=0.25, iou_thres=0.45, device='', view_img=False, save_txt=True, save_conf=True, nosave=False, classes=None, agnostic_nms=False, augment=False, update=False, project='runs/detect/yolov3-spp', name='exp', exist_ok=False)
YOLOv3 ?? v9.5.0-6-g4d8fa83 torch 1.8.1+cu102 CUDA:0 (Tesla T4, 15109.75MB)
--------------------------------------------------------------------------------
Fusing layers...
Model Summary: 269 layers, 62546518 parameters, 0 gradients, 155.7 GFLOPS
image 1/2 /opt/workshop/qinhj/yolov3/data/images/bus.jpg: 640x480 4 persons, Done. (0.027s)
image 2/2 /opt/workshop/qinhj/yolov3/data/images/zidane.jpg: 384x640 2 persons, Done. (0.023s)
Results saved to runs/detect/yolov3-spp/exp
2 labels saved to runs/detect/yolov3-spp/exp/labels
Done. (0.136s)
```

## Quick Test ##
```
* Note:
1) the 'nc' in yaml file and the option "--single-cls" must be the same as the training config;

********************************************************************************
** official models (export OPTIONS="--data data/coco5k.yaml --verbose --save-txt --save-conf")   // benchmark
********************************************************************************

================================================================================
$ python test.py --weights weights/yolov3.pt --project runs/test/yolov3 $OPTIONS    // 1705MiB
--------------------------------------------------------------------------------
Namespace(weights=['weights/yolov3.pt'], data='data/coco5k.yaml', batch_size=32, img_size=640, conf_thres=0.001, iou_thres=0.6, task='val', device='', single_cls=False, augment=False, verbose=True, save_txt=True, save_hybrid=False, save_conf=True, save_json=False, project='runs/test/yolov3', name='exp', exist_ok=False)
YOLOv3 ?? v9.5.0-6-g9c177b3 torch 1.8.1+cu102 CUDA:0 (Tesla T4, 15109.75MB)
--------------------------------------------------------------------------------
Fusing layers...
Model Summary: 261 layers, 61922845 parameters, 0 gradients, 156.3 GFLOPS
val: Scanning '../coco_person/val2017.cache' images and labels... 5000 found, 0 missing, 2307 empty, 0 corrupted: 100%|¨€| 5
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|¨€| 157/157 [15:30<00:00,
                 all        5000       11004       0.776       0.741       0.791       0.524
              person        5000       11004       0.776       0.741       0.791       0.524
Speed: 13.0/1.3/14.3 ms inference/NMS/total per 640x640 image at batch-size 32
Results saved to runs/test/yolov3/exp6
4999 labels saved to runs/test/yolov3/exp6/labels
================================================================================
$ python test.py --weights weights/yolov3-tiny.pt --project runs/test/yolov3-tiny $OPTIONS
--------------------------------------------------------------------------------
Namespace(augment=False, batch_size=32, conf_thres=0.001, data='data/coco5k.yaml', device='', exist_ok=False, img_size=640, iou_thres=0.6, name='exp', project='runs/test/yolov3-tiny', save_conf=True, save_hybrid=False, save_json=False, save_txt=True, single_cls=False, task='val', verbose=True, weights=['weights/yolov3-tiny.pt'])
YOLOv3 ?? v9.5.0-7-gc64b7f1 torch 1.8.1+cu102 CPU
--------------------------------------------------------------------------------
Fusing layers... 
Model Summary: 48 layers, 8849182 parameters, 0 gradients, 13.2 GFLOPS
val: Scanning '../coco_person/val2017.cache' images and labels... 5000 found, 0 missing, 2307 empty, 0 corrupted: 100%|¨€| 5000/5000 [00:00<?, ?it/s]
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|¨€| 157/157 [20:04<00:00,  7.67s/it]
                 all        5000       11004       0.625       0.541       0.561       0.261
                   0        5000       11004       0.625       0.541       0.561       0.261
Speed: 33.2/1.8/35.0 ms inference/NMS/total per 640x640 image at batch-size 32
Results saved to runs/test/yolov3-tiny/exp2
5000 labels saved to runs/test/yolov3-tiny/exp2/labels
================================================================================
$ python test.py --weights weights/yolov3-spp.pt --project runs/test/yolov3-spp $OPTIONS    // the best one in pretrained weights
--------------------------------------------------------------------------------
Namespace(weights=['weights/yolov3-spp.pt'], data='data/coco5k.yaml', batch_size=32, img_size=640, conf_thres=0.001, iou_thres=0.6, task='val', device='', single_cls=False, augment=False, verbose=True, save_txt=True, save_hybrid=False, save_conf=True, save_json=False, project='runs/test/yolov3-spp', name='exp', exist_ok=False)
YOLOv3 ?? v9.5.0-6-g4d8fa83 torch 1.8.1+cu102 CUDA:0 (Tesla T4, 15109.75MB)
--------------------------------------------------------------------------------
Fusing layers...
Model Summary: 269 layers, 62971933 parameters, 0 gradients, 157.1 GFLOPS
val: Scanning '../coco_person/val2017.cache' images and labels... 5000 found, 0 missing, 2307 empty, 0 corrupted: 100%|¨€| 5
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|¨€| 157/157 [15:31<00:00,
                 all        5000       11004         0.8       0.734       0.799       0.529
              person        5000       11004         0.8       0.734       0.799       0.529
Speed: 13.2/1.3/14.5 ms inference/NMS/total per 640x640 image at batch-size 32
Results saved to runs/test/yolov3-spp/exp2
5000 labels saved to runs/test/yolov3-spp/exp2/labels


********************************************************************************
** imilab warmup models (export OPTIONS="--data data/coco_person5k.yaml --single-cls --verbose --save-txt --save-conf")
********************************************************************************

================================================================================
$ python test.py --weights runs/train/yolov3-tiny/exp3/weights/best.pt --project runs/test/yolov3-tiny $OPTIONS
--------------------------------------------------------------------------------
Namespace(weights=['runs/train/yolov3-tiny/exp3/weights/best.pt'], data='data/coco_person5k.yaml', batch_size=32, img_size=640, conf_thres=0.001, iou_thres=0.6, task='val', device='', single_cls=True, augment=False, verbose=True, save_txt=True, save_hybrid=False, save_conf=True, save_json=False, project='runs/test/yolov3-tiny', name='exp', exist_ok=False)
YOLOv3 ?? v9.5.0-7-gc64b7f1 torch 1.8.1+cu102 CUDA:0 (Tesla T4, 15109.75MB)
--------------------------------------------------------------------------------
Fusing layers...
Model Summary: 48 layers, 8666692 parameters, 0 gradients, 12.9 GFLOPS
val: Scanning '../coco_person/val2017.cache' images and labels... 5000 found, 0 missing, 2307 empty, 0 corrupted: 100%|¨€| 5
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|¨€| 157/157 [05:17<00:00,
                 all        5000       11004       0.673       0.597       0.636       0.302
Speed: 2.4/1.2/3.6 ms inference/NMS/total per 640x640 image at batch-size 32
Results saved to runs/test/yolov3-tiny/exp7
4735 labels saved to runs/test/yolov3-tiny/exp7/labels

================================================================================
$ python test.py --weights runs/train/yolov3/exp2/weights/best.pt --project runs/test/yolov3 $OPTIONS    // (X: nc=80 -> nc=1)
--------------------------------------------------------------------------------
Namespace(weights=['runs/train/yolov3/exp2/weights/best.pt'], data='data/coco_person5k.yaml', batch_size=32, img_size=640,conf_thres=0.001, iou_thres=0.6, task='val', device='', single_cls=True, augment=False, verbose=True, save_txt=True, save_hybrid=False, save_conf=True, save_json=False, project='runs/test/yolov3', name='exp', exist_ok=False)
YOLOv3 ?? v9.5.0-7-gc64b7f1 torch 1.8.1+cu102 CUDA:0 (Tesla T4, 15109.75MB)
--------------------------------------------------------------------------------
Fusing layers...
Model Summary: 261 layers, 61497430 parameters, 0 gradients, 154.9 GFLOPS
val: Scanning '../coco_person/val2017.cache' images and labels... 5000 found, 0 missing, 2307 empty, 0 corrupted: 100%|¨€| 5
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|¨€| 157/157 [03:42<00:00,
                 all        5000       11004       0.845       0.805       0.877       0.618
Speed: 10.5/0.7/11.2 ms inference/NMS/total per 640x640 image at batch-size 32
Results saved to runs/test/yolov3/exp8
4100 labels saved to runs/test/yolov3/exp8/labels

================================================================================
$ python test.py --weights runs/train/yolov3-spp/exp2/weights/best.pt --project runs/test/yolov3-spp $OPTIONS
--------------------------------------------------------------------------------
Namespace(weights=['runs/train/yolov3-spp/exp2/weights/best.pt'], data='data/coco_person5k.yaml', batch_size=32, img_size=640, conf_thres=0.001, iou_thres=0.6, task='val', device='', single_cls=True, augment=False, verbose=True, save_txt=True, save_hybrid=False, save_conf=True, save_json=False, project='runs/test/yolov3-spp', name='exp', exist_ok=False)
YOLOv3 ?? v9.5.0-6-g4d8fa83 torch 1.8.1+cu102 CUDA:0 (Tesla T4, 15109.75MB)
--------------------------------------------------------------------------------
Fusing layers...
Model Summary: 269 layers, 62546518 parameters, 0 gradients, 155.7 GFLOPS
val: Scanning '../coco_person/val2017.cache' images and labels... 5000 found, 0 missing, 2307 empty, 0 corrupted: 100%|¨€| 5
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|¨€| 157/157 [04:10<00:00,
                 all        5000       11004        0.85       0.808       0.882       0.623
Speed: 10.9/0.8/11.7 ms inference/NMS/total per 640x640 image at batch-size 32
Results saved to runs/test/yolov3-spp/exp
4140 labels saved to runs/test/yolov3-spp/exp/labels
```

## Quick FAQ ##
```
1. ValueError while training with test2017.txt
train: Scanning '../coco_person/test2017.cache' images and labels... 40670 found, 0 missing, 40670 empty, 0 corrupted: 100%
Traceback (most recent call last):
  File "train.py", line 543, in <module>
    train(hyp, opt, device, tb_writer)
  File "train.py", line 193, in train
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
  File "/home/qinhj/.conda/envs/yolov3/lib/python3.8/site-packages/numpy/core/_methods.py", line 39, in _amax
    return umr_maximum(a, axis, None, out, keepdims, initial, where)
ValueError: zero-size array to reduction operation maximum which has no identity

2. Convert Error: {ConstantOfShape} not supported for yolov3
A: add "--simplify" when export yolov3 or yolov5 models.
https://github.com/OAID/Tengine-Convert-Tools/issues/39

3. why the evaluates always decrease in the 2nd epoch with "--weights"??
Same issue as https://github.com/ultralytics/yolov3/issues/1347
```