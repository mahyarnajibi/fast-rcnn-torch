Net = torch.class('detection.Net')
utils = detection.GeneralUtils()
function Net:__init(model_path,weight_file_path)
	self.model_path = model_path
	self.weight_file_path = weight_file_path

	if model_path == nil then
		error 'The first argument can not be nil!'
	end
	self.model,self.classifier,self.regressor,self.name = dofile(model_path)
	self.model:cuda()
	self.parameters, self.gradParameters = self.model:getParameters()

	if weight_file_path~=nil then
		self:load_weight()
	end
end

function Net:initialize_for_training()
	-- initialize classifier and regressor with appropriate random numbers 
	self.classifier.weight = torch.randn(self.classifier.weight:size()):cuda()* 0.01
	self.classifier.bias:fill(0)
	self.regressor.weight = torch.randn(self.regressor.weight:size()):cuda() * 0.001
	self.regressor.bias:fill(0)
end

function Net:load_weight(weight_file_path)
	if weight_file_path~=nil then
		self.weight_file_path = weight_file_path
	end
	-- Loading parameters
	params = torch.load(self.weight_file_path):getParameters()
	-- Copying parameters
 	self.parameters[{{1,params:size(1)}}]:copy(params)
 	-- Parallelizing the network
 	if config.nGPU > 1 then
 		self.model = utils:makeWholeNetworkParallel(self.model)
 	end
 end

function Net:getParameters()
	return self.parameters, self.gradParameters
end

function Net:training()
	self.model:training()
end

function Net:evaluate()
	self.model:evaluate()
end

function Net:save(save_path,means,stds)
	-- First sanitize the net
	self:_sanitize()
	local tmp_regressor
	if means ~= nil then
	-- Undo the normalization
		tmp_regressor = self.regressor:clone()
		self.regressor.weight = self.regressor.weight:cmul(stds:expand(self.regressor.weight:size()))
		self.regressor.bias = self.regressor.bias:cmul(stds:view(-1)) + means:view(-1)
	end
	-- Save the snapshot
	torch.save(save_path,self.model)
	if means ~=nil then
		self.regressor.weight = tmp_regressor.weight
		self.regressor.bias = tmp_regressor.bias
	end
end

function Net:load_from_caffe(proto_path,caffemodel_path,save_path,model_name)
	caffeModelLoader = detection.CaffeModelConverter(self.model,proto_path,caffemodel_path,model_name,save_path)
	caffeModelLoader:convert()
end

function Net:get_model()
	return self.model
end


function Net:forward(inputs)
  	self.inputs,inputs = utils:recursiveResizeAsCopyTyped(self.inputs,inputs,'torch.CudaTensor')
  	local out = self.model:forward(self.inputs)
  	return out[1],out[2]
end

-- borrowed from https://github.com/soumith/imagenet-multiGPU.torch/blob/master/train.lua
function Net:_sanitize()
  net = self.model
  local list = net:listModules()
  for _,val in ipairs(list) do
    for name,field in pairs(val) do
      if torch.type(field) == 'cdata' then val[name] = nil end
      if name == 'homeGradBuffers' then val[name] = nil end
      if name == 'input_gpu' then val['input_gpu'] = {} end
      if name == 'gradOutput_gpu' then val['gradOutput_gpu'] = {} end
      if name == 'gradInput_gpu' then val['gradInput_gpu'] = {} end
      if (name == 'output' or name == 'gradInput') then
      	if type(field) ~= 'table' then
        	val[name] = field.new()
        end
      end
    end
  end
end



