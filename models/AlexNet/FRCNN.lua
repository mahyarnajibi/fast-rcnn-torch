require 'detection'
local utils = detection.GeneralUtils()
-- To define new models your file should:
-- 1) return the model
-- 2) return a local variable named regressor pointing to the weights of the bbox regressor
-- 3) return a local variable named classifier pointing ro weights of the classifier (without SoftMax!)
-- 4) return the name of the model (used for saving models and logs)
  
 local function create_model(opt) 
  local name = 'frcnn_alexnet'
  backend = backend or cudnn

  

  
-- SHARED PART
  local shared   = nn.Sequential()
  local conv1 =backend.SpatialConvolution(3,96,11,11,4,4,5,5,1)
  conv1.name = 'conv1'
  -- Freeze conv1
  conv1.accGradParameters = function() end 
  shared:add(conv1)
  shared:add(backend.ReLU(true))
  shared:add(backend.SpatialMaxPooling(3,3,2,2,1,1):ceil())
  shared:add(inn.SpatialCrossResponseNormalization(5,0.0001,0.75,1))
  

  local conv2 = backend.SpatialConvolution(96,256,5,5,1,1,2,2,2)
  conv2.name = 'conv2'
  shared:add(conv2)
  shared:add(backend.ReLU(true))
  shared:add(backend.SpatialMaxPooling(3,3,2,2,1,1):ceil())
  shared:add(inn.SpatialCrossResponseNormalization(5,0.0001,0.75,1))
  

  local conv3 = backend.SpatialConvolution(256,384,3,3,1,1,1,1,1)
  conv3.name = 'conv3'
  shared:add(conv3)
  shared:add(backend.ReLU(true))

  local conv4 = backend.SpatialConvolution(384,384,3,3,1,1,1,1,2)
  conv4.name = 'conv4'
  shared:add(conv4)
  shared:add(backend.ReLU(true))
  
  local conv5 = backend.SpatialConvolution(384,256,3,3,1,1,1,1,2)
  conv5.name = 'conv5'
  shared:add(conv5)
  shared:add(backend.ReLU(true))


  -- Convolutions and roi info
  local shared_roi_info = nn.ParallelTable()
  shared_roi_info:add(shared)
  shared_roi_info:add(nn.Identity())
  
  -- Linear Part
  local linear = nn.Sequential()
  linear:add(nn.View(-1):setNumInputDims(3))
  local fc6 = nn.Linear(9216,4096)
  fc6.name = 'fc6'
  linear:add(fc6)
  linear:add(backend.ReLU(true))
  linear:add(nn.Dropout(0.5))

  local fc7 = nn.Linear(4096,4096)
  fc7.name = 'fc7'
  linear:add(fc7)
  linear:add(backend.ReLU(true))
  linear:add(nn.Dropout(0.5))
  


  -- classifier
  local classifier = nn.Linear(4096,opt.nclass+1)
  classifier.name = 'classifier'
  -- regressor
  local regressor = nn.Linear(4096,4 * (opt.nclass+1))
  regressor.name = 'regressor'

  local output = nn.ConcatTable()
  output:add(classifier)
  output:add(regressor)
  
  -- ROI pooling
  local ROIPooling = detection.ROIPooling(6,6):setSpatialScale(1/16)

  -- Whole Model
  local model = nn.Sequential()
  model:add(shared_roi_info)
  model:add(ROIPooling)
  model:add(linear)
  model:add(output)

  model:cuda()
  return model,classifier,regressor,name
end

return create_model