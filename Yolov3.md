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
## download pretrained weights
$ wget https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip
$ wget https://github.com/ultralytics/yolov3/releases/download/v9.5.0/yolov3-tiny.pt
$ wget https://github.com/ultralytics/yolov3/releases/download/v9.5.0/yolov3-spp.pt
$ wget https://github.com/ultralytics/yolov3/releases/download/v9.5.0/yolov3.pt
$ wget https://github.com/ultralytics/yolov3/releases/download/v9.5.0/yolov5l.pt

## detect
$ export OPTIONS="--source data/images --img-size 640 --save-txt --save-conf"
$ python detect.py --weights weights/yolov3-tiny.pt --project runs/detect/yolov3-tiny $OPTIONS
$ python detect.py --weights weights/yolov3.pt --project runs/detect/yolov3 $OPTIONS
$ python detect.py --weights weights/yolov3-spp.pt --project runs/detect/yolov3-spp $OPTIONS
$ python detect.py --weights weights/yolov5l.pt --project runs/detect/yolov5l $OPTIONS

## test
$ export OPTIONS="--data data/coco128.yaml --img-size 640 --verbose --save-txt --save-conf"
$ python test.py --weights weights/yolov3-tiny.pt --project runs/test/yolov3-tiny $OPTIONS
$ python test.py --weights weights/yolov3.pt --project runs/test/yolov3 $OPTIONS
$ python test.py --weights weights/yolov3-spp.pt --project runs/test/yolov3-spp $OPTIONS
$ python test.py --weights weights/yolov5l.pt --project runs/test/yolov5l $OPTIONS

## train
$ export OPTIONS="--data data/coco128.yaml --epochs 20 --batch-size 16 --img-size 640"
$ export OPTIONS=$OPTIONS" --rect --resume --device --single-cls --adam"
$ python train.py --weights weights/yolov3-tiny.pt --cfg models/yolov3-tiny.yaml --project runs/train/yolov3-tiny $OPTIONS
$ python train.py --weights weights/yolov3-spp.pt --cfg models/yolov3-spp.yaml --project runs/train/yolov3-spp $OPTIONS
$ python train.py --weights weights/yolov3.pt --cfg models/yolov3.yaml --project runs/train/yolov3 $OPTIONS

* Note for Window:
> SET OPTIONS=--data data/coco128.yaml --img-size 640 --verbose --save-txt --save-conf
> echo %OPTIONS%
```

## FAQ ##
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
```
