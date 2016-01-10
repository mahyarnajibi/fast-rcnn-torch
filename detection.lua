require('nn')
require('inn')
require('cudnn')

detection = {}

-- Detection utilities
torch.include('detection','utils/ImageTransformer.lua')
torch.include('detection','utils/CaffeLoader.lua')
torch.include('detection','utils/CaffeModelConverter.lua')
torch.include('detection','utils/GeneralUtils.lua')


-- -- Detection datasets
torch.include('detection','datasets/DataSetDetection.lua')
torch.include('detection','datasets/DataSetPascal.lua')

-- Detection roidb modules
torch.include('detection','ROI/ROI.lua')
torch.include('detection','ROI/ROIPooling.lua')

-- Detection Network training and Testing
-- torch.include('detection','DB_tester.lua')
torch.include('detection','NetworkWrapper.lua')
torch.include('detection','Net.lua')
