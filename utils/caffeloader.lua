require 'loadcaffe'
local caffeloader = torch.class('caffeloader')

function caffeloader:__init(torch_model,prototxt,caffemodel_path)
	self.prototxt = prototxt
	self.caffemodel_path = caffemodel_path
	self.torch_model = torch_model
end

function caffeloader:load()

	prototxt = self.prototxt
	caffemodel_path = self.caffemodel_path
	caffemodel = loadcaffe.load(prototxt,caffemodel_path,'cudnn')
	torch_model = self.torch_model
	-- Copying caffe weight models
	local torch_parameters = torch_model:parameters()
  	local caffeparameters = caffemodel:parameters()
  
	for k,v in ipairs(torch_parameters) do
	    local cur_caffe_param = caffeparameters[k]
	    assert(cur_caffe_param:numel() == v:numel(), 'Number of elements in layer # ' .. k .. ' does not match!')
	    v:copy(cur_caffe_param)
	end
	return torch_model
end
