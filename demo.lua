-- This is a demo file for performing detection given some proposals
-- Require the detection package
require 'detection'
-- Require the utils class
local utils = detection.GeneralUtils()

-- Paths
local image_path = './data/demo/test.jpg'
local selective_search_path = './data/demo/proposals.mat'
local param_path = 'data/models/torch_fast_rcnn_models/CaffeNet/FRCNN_CaffeNet.t7'
local model_path = 'models/CaffeNet/FRCNN.lua'

-- Creating the network
network = detection.Net(model_path,param_path)

-- Creating the network wrapper
local network_wrapper = detection.NetworkWrapper(network)
network_wrapper:evaluate()

-- Loading proposals from file
local proposals = matio.load(selective_search_path)['boxes']
proposals = proposals:add(1)

-- Loading the image
local im = image.load(image_path)


-- detect !
local scores, bboxes = network_wrapper:detect(im, proposals)

-- visualization
local threshold = 0.5
-- classes from Pascal used for training the model
local cls = {'aeroplane','bicycle','bird','boat','bottle','bus','car',
  'cat','chair','cow','diningtable','dog','horse','motorbike',
  'person','pottedplant','sheep','sofa','train','tvmonitor'}
utils:visualize_detections(im,bboxes,scores,threshold,cls)
