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

* yolov3 (only)
$ cd yolov3
# export at 416x416 with batch size 1
$ export OPTIONS="--img-size 416 416 --batch 1 --simplify"
$ python models/export.py --weights $MODELS/yolov3-tiny.pt $OPTIONS

* yolov5/yolov3
$ cd yolov5
# export at 640x640 with batch size 1
$ export OPTIONS="--img-size 640 640 --batch 1 --device 0"
$ python models/export.py --weights $MODELS/yolov5s.pt $OPTIONS

## Tips:
0) default: cpu, for GPU try: "--device 0";
1) add "--simplify" while export yolov5 models to onnx if necessary;
```

## Quick Tutorial ##
```
>>> import onnx
>>> attr = {"kernel_shape":(2,2),"ceil_mode":0,"pads":(0,0,0,0),"strides":(1,1)}
>>> node = onnx.helper.make_node("MaxPool",["69"],["70"],"MaxPool_xxx", **attr)
>>> for attr in node.attribute:
...     if "pads" == attr.name:
...         node.attribute.remove(attr)
...         break
>>> node.attribute.extend(onnx.helper.make_attribute(key, value) for key, value in {"pads":(0,1,0,1)}.items())

# =========================
# PADs Order: nchw(above) nchw(below)
# =========================
* reflect_pad
# =========================
padding_above: [0, 0, 1, 1]
padding_below: [0, 0, 1, 1]
mode: reflect
# =========================
* edge_pad
# =========================
padding_above: [0, 0, 1, 1]
padding_below: [0, 0, 1, 1]
mode: edge
```

## Reference ##
[onnx/optimizer](https://github.com/onnx/optimizer)  