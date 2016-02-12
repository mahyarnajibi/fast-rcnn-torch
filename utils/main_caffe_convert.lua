require('nn')
require('cudnn')
require('inn')
require('detection')

caffeConverter = detection.CaffeModelConverter('./models/VGG16/VGG16_imgnet.lua','/mnt/mag5tb/data/detection/data/imagenet_models/VGG16.prototxt','/mnt/mag5tb/data/detection/data/imagenet_models/VGG16.v2.caffemodel','VGG16.v2','./')
caffeConverter:convert()

