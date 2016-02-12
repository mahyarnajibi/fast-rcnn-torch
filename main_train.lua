-- Require the detection package
require 'detection'

-- Paths
local dataset_name = config.dataset
local image_set = config.train_img_set
local dataset_dir = paths.concat(config.dataset_path,dataset_name)
local ss_dir = './data/datasets/selective_search_data/'
local ss_file =  paths.concat(ss_dir, dataset_name .. '_' .. image_set .. '.mat')
local param_path = config.pre_trained_file
local model_path = config.model_def


-- Loading the dataset
local dataset = detection.DataSetPascal({image_set = image_set, datadir = dataset_dir, roidbdir = ss_dir , roidbfile = ss_file})
--local dataset = detection.DataSetCoco({image_set = image_set, datadir = dataset_dir})

-- Creating the detection network
model_opt = {}
model_opt.test = false
model_opt.fine_tunning = not config.resume_training

-- model_opt.f
network = detection.Net(model_path,param_path,model_opt)
-- Creating the network wrapper
local network_wrapper = detection.NetworkWrapper() -- This adds train and test functionality to the global network

-- Train the network on the dataset
print('Training the network...')
network_wrapper:trainNetwork(dataset)