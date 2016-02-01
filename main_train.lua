-- Require the detection package
require 'detection'

-- Paths
local dataset_name = config.dataset
local image_set = config.train_img_set
local dataset_dir = './data/datasets/'..dataset_name
local ss_dir = './data/datasets/selective_search/'
local ss_file =  ss_dir .. dataset_name .. '_' .. image_set .. '.mat'
local param_path = 'data/models/torch_imagenet_models/CaffeNet.v2.t7'
local model_path = 'models/CaffeNet/FRCNN.lua'


-- Loading the dataset
local dataset = detection.DataSetPascal({image_set = image_set, datadir = dataset_dir, roidbdir = ss_dir , roidbfile = ss_file})

-- Creating the detection network
network = detection.Net(model_path,param_path)

-- Creating the network wrapper
local network_wrapper = detection.NetworkWrapper() -- This adds train and test functionality to the global network

-- Train the network on the dataset
print('Training the network...')
network_wrapper:trainNetwork(dataset)