## Quick Env ##
```
# clone repo
$ git clone https://github.com/ultralytics/yolov5

# base requirements
$ pip install -r requirements.txt

# export requirements
$ pip install coremltools>=4.1 onnx>=1.8.1 scikit-learn==0.19.2 # -i http://pypi.douban.com/simple --trusted-host pypi.douban.com --no-deps, or:
$ pip install onnx-1.9.0-cp39-cp39-manylinux2010_x86_64.whl
$ pip install scikit_learn-0.24.2-cp39-cp39-manylinux2010_x86_64.whl
$ pip install coremltools-4.1-cp38-none-manylinux1_x86_64.whl
$ pip install onnxoptimizer-0.2.6-cp39-cp39-manylinux2014_x86_64.whl
$ pip install onnxruntime-1.7.0-cp39-cp39-manylinux2014_x86_64.whl
```

## Quick Convert ##
```
$ export MODELS=/media/sf_Workshop/Models/coco_nc80/
$ export OPTIONS="--img 640 --batch 1"

* yolov3 (only)
$ cd yolov3
$ python models/export.py --weights $MODELS/yolov3-tiny.pt $OPTIONS

* yolov5/yolov3
$ cd yolov5
## export at 640x640 with batch size 1
$ export OPTIONS="--img 640 --batch 1 --simplify --device 0"
$ python models/export.py --weights $MODELS/yolov5s.pt $OPTIONS

## Tips:
0) default: cpu, for GPU try: "--device 0";
1) add "--simplify" while export yolov5 models to onnx if necessary;
```

## Reference ##
[onnx/optimizer](https://github.com/onnx/optimizer)  