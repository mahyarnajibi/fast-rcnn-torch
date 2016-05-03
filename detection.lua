require 'image'
require 'cudnn'
require 'inn'
require 'nn'
require 'torch'
require 'xlua'
matio = require 'matio'
tds = require 'tds'
config = dofile 'config.lua'
config = config.parse(arg)
cutorch.setDevice(config.GPU_ID)

-- Setting the random seed
torch.manualSeed(3)

detection = {}

-- General Utilities
torch.include('detection','utils/GeneralUtils.lua')

-- Detection roidb modules
torch.include('detection','ROI/ROI.lua')
torch.include('detection','ROI/ROIPooling.lua')
-- Detection utilities
torch.include('detection','utils/InputMaker.lua')
torch.include('detection','utils/CaffeLoader.lua')
torch.include('detection','utils/CaffeModelConverter.lua')
torch.include('detection','utils/heap/heap.lua')


-- -- Detection datasets
torch.include('detection','datasets/DataSetDetection.lua')
torch.include('detection','datasets/DataSetPascal.lua')
torch.include('detection','datasets/DataSetCoco.lua')

-- Detection Network training and Testing 
-- torch.include('detection','DB_tester.lua')
torch.include('detection','network/NetworkWrapper.lua')
torch.include('detection','network/Net.lua')
torch.include('detection','train/Batcher.lua')
torch.include('detection','train/ParallelTrainer.lua')
torch.include('detection','train/SequentialTrainer.lua')
torch.include('detection','train/WeightedSmoothL1Criterion.lua')
