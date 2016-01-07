require('nn')
require('cudnn')
require('inn')
require('utils.caffeModelConverter')
require('nnf')
debugger = require 'fb.debugger'

caffeConverter = caffeModelConverter('./','./models/frcnn_alexnet.lua','./data/models/caffe_model_proto/CaffeNet/test.prototxt','./data/models/caffe_fast_rcnn_models/caffenet_fast_rcnn_iter_40000.caffemodel')
caffeConverter:convert()


