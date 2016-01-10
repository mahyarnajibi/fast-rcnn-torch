-- This is a demo file for performing detection give some proposals

matio = require 'matio'
require 'image'
require 'cudnn'
require 'inn'
require 'nn'
require 'torch'
require 'detection'

config = dofile 'config.lua'
config = config.parse(arg)
utils = detection.GeneralUtils()

print(config)

image_path = './demo/test.jpg'
selective_search_path = './demo/proposals.mat'
param_path = 'data/models/torch_fast_rcnn_models/CaffeNet/FRCNN_CaffeNet.t7'
model_path = 'models/CaffeNet/FRCNN.lua'

network = detection.Net(model_path,param_path)
network:get_net():cuda()
network:get_net():evaluate()

image_transformer= detection.ImageTransformer{mean_pix={102.9801,115.9465,122.7717},
                                         raw_scale = 255,
                                         swap = {3,2,1}}

network_wrapper = detection.NetworkWrapper(network,image_transformer)
network_wrapper:evaluate()


-- Loading proposals

---NOTE that the code assumes that the minimum bounding box position is at 1
-- Also the order is like as the original selective search output 
-- In other words the code assumes that it is reading the original selective search mat file!

proposals = matio.load(selective_search_path)['boxes']
proposals = proposals:add(1)




-- Loading processed image for debug!
--im = matio.load('./demo/im.mat')['im']

-- Loading image

im = image.load(image_path)

-- detect !
scores, bboxes = network_wrapper:detect(im, proposals)


-- visualization
threshold = 0.5
-- classes from Pascal used for training the model
cls = {'aeroplane','bicycle','bird','boat','bottle','bus','car',
  'cat','chair','cow','diningtable','dog','horse','motorbike',
  'person','pottedplant','sheep','sofa','train','tvmonitor'}

w = utils:visualize_detections(im,bboxes,scores,threshold,cls)
