-- Loading dataset
matio = require 'matio'
require 'image'
require 'cudnn'
require 'inn'
require 'nn'
require 'torch'
require 'detection'

debugger = require 'fb.debugger'
config = dofile 'config.lua'
config = config.parse(arg)
utils = detection.GeneralUtils()

local dataset_name = config.dataset
local image_set = config.test_img_set

local dataset_dir = './data/datasets/'..dataset_name
local ss_dir = './data/datasets/selective_search/'
local ss_file =  ss_dir .. dataset_name .. '_' .. image_set .. '.mat'

-- Loading dataset
local dataset = detection.DataSetPascal({image_set = image_set, datadir = dataset_dir, roidbdir = ss_dir , roidbfile = ss_file})
-- Loading roidb
dataset:loadROIDB()


-- Creating the network and its wrapper
local param_path = 'data/models/torch_fast_rcnn_models/CaffeNet/FRCNN_CaffeNet.t7'
local model_path = 'models/CaffeNet/FRCNN.lua'

local network = detection.Net(model_path,param_path)
network:get_net():cuda()
network:get_net():evaluate()

local image_transformer= detection.ImageTransformer{mean_pix={102.9801,115.9465,122.7717},
                                         raw_scale = 255,
                                         swap = {3,2,1}}

local network_wrapper = detection.NetworkWrapper(network,image_transformer)
network_wrapper:evaluate()


-- Perform the test
network_wrapper:testNetwork(dataset)



-- Returned entity after calling attachProposals()
-- {
--   class : CharTensor - size: 3118
--   overlap : FloatTensor - size: 3118
--   overlap_class : FloatTensor - size: 3118x20
--   label : IntTensor - size: 3118
--   boxes : IntTensor - size: 3118x4
--   size : function: 0x4115b7a8
--   correspondance : LongTensor - size: 3118
--   gt : ByteTensor - size: 3118
-- }
