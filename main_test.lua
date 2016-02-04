-- Require the detection package
require 'detection'
-- Paths
local dataset_name = config.dataset
local image_set = config.test_img_set
local dataset_dir = paths.concat(config.dataset_path,dataset_name)
local ss_dir = './data/datasets/selective_search_data/'
local ss_file =  paths.concat(ss_dir,dataset_name .. '_' .. image_set .. '.mat')
local param_path = config.model_weights
local model_path = config.model_def

-- Loading the dataset
local dataset = detection.DataSetPascal({image_set = image_set, datadir = dataset_dir, roidbdir = ss_dir , roidbfile = ss_file})

-- Creating the detection net
network = detection.Net(model_path,param_path)

-- Creating the wrapper
local network_wrapper = detection.NetworkWrapper()

-- Test the network
print('Testing the network...')
network_wrapper:testNetwork(dataset)