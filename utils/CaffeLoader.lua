require 'loadcaffe'
local CaffeLoader = torch.class('detection.CaffeLoader')

function CaffeLoader:__init(torch_model,prototxt,caffemodel_path)
	self.prototxt = prototxt
	self.caffemodel_path = caffemodel_path
	self.torch_model = torch_model
end


function CaffeLoader:load()

	prototxt = self.prototxt
	caffemodel_path = self.caffemodel_path
	caffemodel = loadcaffe.load(prototxt,caffemodel_path,'cudnn')
	torch_model = dofile(self.torch_model)()
	-- Creating the sequential model from table of moduels
	-- Copying caffe weight models
	local torch_parameters = torch_model:getParameters()
  	local caffeparameters = caffemodel:getParameters()
    torch_parameters:copy(caffeparameters)
	return torch_model
end
