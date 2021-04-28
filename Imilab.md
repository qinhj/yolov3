# Imilab: smoke test with Yolov3 #

## Quick Note ##
```
1) the options can be overwritten by the last same one;
2) for Window, try:
> SET OPTIONS=--data data/coco128.yaml --img-size 640 --verbose --save-txt --save-conf
> echo %OPTIONS%
```

## Quick Train ##
```
$ export OPTIONS="--data data/coco_person5k.yaml --epochs 20 --batch-size 16 --img-size 640 --single-cls --adam"
$ export OPTIONS="--data data/coco_person5k.yaml --epochs 20 --batch-size 8 --img-size 640 --multi-scale --single-cls --adam"
#$ export OPTIONS=$OPTIONS" --rect --resume --device"

================================================================================
$ python train.py --weights weights/yolov3-tiny.pt --cfg models/yolov3-tiny.yaml --project runs/train/yolov3-tiny $OPTIONS
--------------------------------------------------------------------------------
Namespace(weights='weights/yolov3-tiny.pt', cfg='models/yolov3-tiny.yaml', data='data/coco_person5k.yaml', hyp='data/hyp.scratch.yaml', epochs=20, batch_size=16, img_size=[640, 640], rect=False, resume=False, nosave=False, notest=False, noautoanchor=False, evolve=False, bucket='', cache_images=False, image_weights=False, device='', multi_scale=False, single_cls=True, adam=True, sync_bn=False, local_rank=-1, workers=8, project='runs/train/yolov3-tiny', entity=None, name='exp', exist_ok=False, quad=False, linear_lr=False, label_smoothing=0.0, upload_dataset=False, bbox_interval=-1, save_period=-1, artifact_alias='latest', world_size=1, global_rank=-1, save_dir='runs/train/yolov3-tiny/exp2', total_batch_size=16)
tensorboard: Start with 'tensorboard --logdir runs/train/yolov3-tiny', view at http://localhost:6006/
hyperparameters: lr0=0.01, lrf=0.2, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0
wandb: Install Weights & Biases for YOLOv5 logging with 'pip install wandb' (recommended)
Overriding model.yaml nc=80 with nc=1
--------------------------------------------------------------------------------
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
 20          [19, 15]  1     13860  models.yolo.Detect                      [1, [[10, 14, 23, 27, 37, 58], [81, 82, 135, 169, 344, 319]], [256, 512]]
Model Summary: 59 layers, 8669876 parameters, 8669876 gradients, 13.0 GFLOPS
--------------------------------------------------------------------------------
Transferred 66/72 items from weights/yolov3-tiny.pt
Scaled weight_decay = 0.0005
Optimizer groups: 13 .bias, 13 conv.weight, 11 other
train: Scanning '../coco_person/val2017' images and labels... 5000 found, 0 missing, 2307 empty, 0 corrupted: 100%|¨€| 5000/5000 [03:58<00:00, 20.94it/s]
train: New cache created: ../coco_person/val2017.cache
val: Scanning '../coco_person/val2017.cache' images and labels... 5000 found, 0 missing, 2307 empty, 0 corrupted: 100%|¨€| 5000/5000 [00:00<?, ?it/s]
Plotting labels... 
--------------------------------------------------------------------------------
autoanchor: Analyzing anchors... anchors/target = 2.97, Best Possible Recall (BPR) = 0.9953
Image sizes 640 train, 640 test
Using 4 dataloader workers
Logging results to runs/train/yolov3-tiny/exp2
Starting training for 20 epochs...
--------------------------------------------------------------------------------
     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
      0/19     2.75G    0.0736   0.03689         0    0.1105        18       640: 100%|¨€| 313/313 [01:02<00:00,  5.02it/s]
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|¨€| 157/157 [00:41<00:00,  3.82it/s]
                 all        5000       11004       0.226       0.289       0.144      0.0317
--------------------------------------------------------------------------------
     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
      1/19     1.93G   0.07021   0.03817         0    0.1084        48       640: 100%|¨€| 313/313 [00:55<00:00,  5.60it/s]
...

================================================================================
$ python train.py --weights weights/yolov3-spp.pt --cfg models/yolov3-spp.yaml --project runs/train/yolov3-spp $OPTIONS --epochs 30
--------------------------------------------------------------------------------
Namespace(weights='weights/yolov3-spp.pt', cfg='models/yolov3-spp.yaml', data='data/coco_person5k.yaml', hyp='data/hyp.scratch.yaml', epochs=30, batch_size=16, img_size=[640, 640], rect=False, resume=False, nosave=False, notest=False, noautoanchor=False, evolve=False, bucket='', cache_images=False, image_weights=False, device='', multi_scale=False, single_cls=True, adam=True, sync_bn=False, local_rank=-1, workers=8, project='runs/train/yolov3-spp', entity=None, name='exp', exist_ok=False, quad=False, linear_lr=False, label_smoothing=0.0, upload_dataset=False, bbox_interval=-1, save_period=-1, artifact_alias='latest', world_size=1, global_rank=-1, save_dir='runs/train/yolov3-spp/exp2', total_batch_size=16)
tensorboard: Start with 'tensorboard --logdir runs/train/yolov3-spp', view at http://localhost:6006/
hyperparameters: lr0=0.01, lrf=0.2, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0
wandb: Install Weights & Biases for YOLOv5 logging with 'pip install wandb' (recommended)
Overriding model.yaml nc=80 with nc=1
--------------------------------------------------------------------------------
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
 28      [27, 22, 15]  1     32310  models.yolo.Detect                      [1, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [256, 512, 1024]]
Model Summary: 342 layers, 62573334 parameters, 62573334 gradients, 155.9 GFLOPS
--------------------------------------------------------------------------------
Transferred 438/446 items from weights/yolov3-spp.pt
Scaled weight_decay = 0.0005
Optimizer groups: 76 .bias, 76 conv.weight, 73 other
train: Scanning '../coco_person/val2017.cache' images and labels... 5000 found, 0 missing, 2307 empty, 0 corrupted: 100%|¨€| 5000/5000 [00:00<?, ?it/s]
val: Scanning '../coco_person/val2017.cache' images and labels... 5000 found, 0 missing, 2307 empty, 0 corrupted: 100%|¨€| 5000/5000 [00:00<?, ?it/s]
Plotting labels... 
--------------------------------------------------------------------------------
autoanchor: Analyzing anchors... anchors/target = 4.45, Best Possible Recall (BPR) = 0.9962
Image sizes 640 train, 640 test
Using 4 dataloader workers
Logging results to runs/train/yolov3-spp/exp2
Starting training for 30 epochs...
--------------------------------------------------------------------------------
     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
      0/29     8.21G   0.07025    0.0146         0   0.08484        26       640: 100%|¨€| 313/313 [04:33<00:00,  1.15it/s]
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|¨€| 157/157 [02:21<00:00,  1.11it/s]
                 all        5000       11004      0.0312      0.0395     0.00848     0.00208
--------------------------------------------------------------------------------
     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
      1/29     9.32G   0.07011   0.01473         0   0.08483        27       640: 100%|¨€| 313/313 [04:18<00:00,  1.21it/s]
...

================================================================================
$ python train.py --weights weights/yolov3.pt --cfg models/yolov3.yaml --project runs/train/yolov3 $OPTIONS --batch-size 24
--------------------------------------------------------------------------------
Namespace(weights='weights/yolov3.pt', cfg='models/yolov3.yaml', data='data/coco_person5k.yaml', hyp='data/hyp.scratch.yaml', epochs=20, batch_size=24, img_size=[640, 640], rect=False, resume=False, nosave=False, notest=False, noautoanchor=False, evolve=False, bucket='', cache_images=False, image_weights=False, device='', multi_scale=False, single_cls=True, adam=True, sync_bn=False, local_rank=-1, workers=8, project='runs/train/yolov3', entity=None, name='exp', exist_ok=False, quad=False, linear_lr=False, label_smoothing=0.0, upload_dataset=False, bbox_interval=-1, save_period=-1, artifact_alias='latest', world_size=1, global_rank=-1, save_dir='runs/train/yolov3/exp', total_batch_size=24)
tensorboard: Start with 'tensorboard --logdir runs/train/yolov3', view at http://localhost:6006/
hyperparameters: lr0=0.01, lrf=0.2, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0
wandb: Install Weights & Biases for YOLOv5 logging with 'pip install wandb' (recommended)
Overriding model.yaml nc=80 with nc=1
--------------------------------------------------------------------------------
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
 28      [27, 22, 15]  1     32310  models.yolo.Detect                      [1, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [256, 512, 1024]]
Model Summary: 333 layers, 61523734 parameters, 61523734 gradients, 155.1 GFLOPS
--------------------------------------------------------------------------------
Transferred 432/440 items from weights/yolov3.pt
Scaled weight_decay = 0.0005625000000000001
Optimizer groups: 75 .bias, 75 conv.weight, 72 other
train: Scanning '../coco_person/val2017.cache' images and labels... 5000 found, 0 missing, 2307 empty, 0 corrupted: 100%|¨€| 5000/5000 [00:00<?, ?it/s]
val: Scanning '../coco_person/val2017.cache' images and labels... 5000 found, 0 missing, 2307 empty, 0 corrupted: 100%|¨€| 5000/5000 [00:00<?, ?it/s]
Plotting labels... 
--------------------------------------------------------------------------------
autoanchor: Analyzing anchors... anchors/target = 4.45, Best Possible Recall (BPR) = 0.9962
Image sizes 640 train, 640 test
Using 4 dataloader workers
Logging results to runs/train/yolov3/exp
Starting training for 20 epochs...
--------------------------------------------------------------------------------
     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
      0/19     6.83G   0.06479   0.01488         0   0.07967        46       640: 100%|¨€| 209/209 [04:32<00:00,  1.30s/it]
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|¨€| 105/105 [02:40<00:00,  1.53s/it]
                 all        5000       11004      0.0339      0.0922      0.0128     0.00222
--------------------------------------------------------------------------------
     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
      1/19     13.2G   0.06604   0.01448         0   0.08053        23       640: 100%|¨€| 209/209 [04:15<00:00,  1.22s/it]
...
```

## Quick Detect ##
```
$ export OPTIONS="--source data/images --img-size 640 --save-txt --save-conf"
$ python detect.py --weights weights/yolov3-tiny.pt --project runs/detect/yolov3-tiny $OPTIONS
$ python detect.py --weights weights/yolov3.pt --project runs/detect/yolov3 $OPTIONS
$ python detect.py --weights weights/yolov3-spp.pt --project runs/detect/yolov3-spp $OPTIONS
```

## Quick Test ##
```
$ export OPTIONS="--data data/coco128.yaml --img-size 640 --verbose --save-txt --save-conf"
$ python test.py --weights weights/yolov3-tiny.pt --project runs/test/yolov3-tiny $OPTIONS
$ python test.py --weights weights/yolov3.pt --project runs/test/yolov3 $OPTIONS
$ python test.py --weights weights/yolov3-spp.pt --project runs/test/yolov3-spp $OPTIONS
```

## FAQ ##
```
```
