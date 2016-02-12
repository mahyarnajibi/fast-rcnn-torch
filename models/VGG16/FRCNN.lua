require 'detection'
local utils = detection.GeneralUtils()
-- To define new models your file should:
-- 1) return the model
-- 2) return a local variable named regressor pointing to the weights of the bbox regressor
-- 3) return a local variable named classifier pointing ro weights of the classifier (without SoftMax!)
-- 4) return the name of the model (used for saving models and logs)
  
 local function create_model(opt) 
  	local name = 'frcnn_vgg16'
  	backend = backend or cudnn

	-- SHARED PART
  	local shared   = nn.Sequential()
	local conv1_1 = cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1, 1)
	-- Freeze conv1_1
	conv1_1.accGradParameters = function() end 
  	shared:add(conv1_1)
  	shared:add(cudnn.ReLU(true))
  	-- Freeze conv1_2
  	local conv1_2 = cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1, 1)
  	conv1_2.accGradParameters = function() end 
	shared:add(conv1_2)
	shared:add(cudnn.ReLU(true))
	shared:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())
	-- Freeze conv2_1
	local conv2_1 = cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1, 1)
	conv2_1.accGradParameters = function() end 
	shared:add(conv2_1)
	shared:add(cudnn.ReLU(true))
	-- Freeze conv2_2
	local conv2_2 = cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1, 1)
	conv2_2.accGradParameters = function() end 
	shared:add(conv2_2)
	shared:add(cudnn.ReLU(true))
	shared:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())
	-- Freeze conv3_1
	local conv3_1 = cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1, 1)
	--conv3_1.accGradParameters = function() end
	shared:add(conv3_1)
	shared:add(cudnn.ReLU(true))

	local conv3_2 = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
	--conv3_2.accGradParameters = function() end
	shared:add(conv3_2)
	shared:add(cudnn.ReLU(true))


	local conv3_3 = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
	--conv3_3.accGradParameters = function() end
	shared:add(conv3_3)
	shared:add(cudnn.ReLU(true))
	shared:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())


	local conv4_1 = cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1, 1)
	--conv4_1.accGradParameters = function() end
	shared:add(conv4_1)
	shared:add(cudnn.ReLU(true))


	local conv4_2 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
	--conv4_2.accGradParameters = function() end
	shared:add(conv4_2)
	shared:add(cudnn.ReLU(true))
	
	local conv4_3 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
	--conv4_3.accGradParameters = function() end
	shared:add(conv4_3)
	shared:add(cudnn.ReLU(true))
	shared:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())


	local conv5_1 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
	--conv5_1.accGradParameters = function() end
	shared:add(conv5_1)
	shared:add(cudnn.ReLU(true))

	local conv5_2 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
	--conv5_2.accGradParameters = function() end
	shared:add(conv5_2)
	shared:add(cudnn.ReLU(true))

	local conv5_3 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
	--conv5_3.accGradParameters = function() end
	shared:add(conv5_3)
	shared:add(cudnn.ReLU(true))


	-- Convolutions and roi info
	local shared_roi_info = nn.ParallelTable()
	shared_roi_info:add(shared)
	shared_roi_info:add(nn.Identity())
	  
	-- Linear Part
	local linear = nn.Sequential()
	linear:add(nn.View(-1):setNumInputDims(3))
	linear:add(nn.Linear(25088,4096))
	linear:add(backend.ReLU(true))
	linear:add(nn.Dropout(0.5))

	linear:add(nn.Linear(4096,4096))
	linear:add(backend.ReLU(true))
	linear:add(nn.Dropout(0.5))
  


	-- classifier
	local classifier = nn.Linear(4096,21)
	-- regressor
	local regressor = nn.Linear(4096,84)

	local output = nn.ConcatTable()
	output:add(classifier)
	output:add(regressor)
  
	-- ROI pooling
	local ROIPooling = detection.ROIPooling(7,7):setSpatialScale(1/16)

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























