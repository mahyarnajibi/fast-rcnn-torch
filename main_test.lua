-- Require the detection package
require 'detection'

-- Paths
local dataset_name = config.dataset
local image_set = config.test_img_set
local dataset_dir = './data/datasets/'..dataset_name
local ss_dir = './data/datasets/selective_search/'
local ss_file =  ss_dir .. dataset_name .. '_' .. image_set .. '.mat'
local param_path = '/mnt/mag5tb/data/frcnn_torch/data/trained_models/frcnn_alexnet_VOC2007_iter_40000_01.31_14.57.t7'
local model_path = 'models/CaffeNet/FRCNN.lua'

-- Loading the dataset
local dataset = detection.DataSetPascal({image_set = image_set, datadir = dataset_dir, roidbdir = ss_dir , roidbfile = ss_file})

-- Creating the detection net
network = detection.Net(model_path,param_path)

-- Creating the wrapper
local network_wrapper = detection.NetworkWrapper()

-- Test the network
print('Testing the network...')
network_wrapper:testNetwork(dataset)