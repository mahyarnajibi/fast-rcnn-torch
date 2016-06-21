require 'loadcaffe'
local CaffeLoader = torch.class('detection.CaffeLoader')

function CaffeLoader:__init(torch_model,prototxt,caffemodel_path)
	self.prototxt = prototxt
	self.caffemodel_path = caffemodel_path
	self.torch_model = torch_model
end


function CaffeLoader:load()

	local prototxt = self.prototxt
	local caffemodel_path = self.caffemodel_path
	local caffemodel = loadcaffe.load(prototxt,caffemodel_path,'cudnn')
	local opt = {}
	opt.nclass = 20
	local torch_model = dofile(self.torch_model)(opt)
	-- Creating the sequential model from table of moduels
	-- Copying caffe weight models
	local torch_parameters = torch_model:getParameters()
  	local caffeparameters = caffemodel:getParameters()
    torch_parameters:copy(caffeparameters)
	return torch_model
end
