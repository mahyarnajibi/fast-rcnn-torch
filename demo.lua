-- This is a demo file for performing detection given some proposals
-- Require the detection package
require 'detection'
-- Require the utils class
local utils = detection.GeneralUtils()

-- Paths
local image_path = './data/demo/test.jpg'
local selective_search_path = './data/demo/proposals.mat'
local param_path = config.model_weights
local model_path = config.model_def

-- Creating the network
local model_opt = {}
model_opt.fine_tuning = false
model_opt.test = true
if config.dataset== 'MSCOCO' then
   model_opt.nclass = 80
else
   model_opt.nclass = 20
end

network = detection.Net(model_path,param_path,model_opt)

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
