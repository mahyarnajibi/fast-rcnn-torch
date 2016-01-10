Net = torch.class('detection.Net')
utils = detection.GeneralUtils()
function Net:__init(model_path,weight_file_path)
	self.model_path = model_path
	self.weight_file_path = weight_file_path
	if model_path == nil then
		error 'The first argument can not be nil!'
	end
	self.model = dofile(model_path)
	if weight_file_path~=nil then
		self:load_weight()
	end
end

function Net:load_weight(weight_file_path)
	if weight_file_path~=nil then
		self.weight_file_path = weight_file_path
	end

	-- Loading parameters
	params = torch.load(self.weight_file_path):getParameters()
	-- Copying parameters
 	self.model:getParameters():copy(params)

 end

function Net:load_from_caffe(proto_path,caffemodel_path,save_path,model_name)
	caffeModelLoader = detection.CaffeModelConverter(self.model,proto_path,caffemodel_path,model_name,save_path)
	caffeModelLoader:convert()
end

function Net:get_net()
	return self.model
end


function Net:forward(inputs)
  	self.inputs,inputs = utils:recursiveResizeAsCopyTyped(self.inputs,inputs,'torch.CudaTensor')
  	-- -- FOR DEBUG
  	-- local temp_inp = matio.load('./demo/net_input.mat')
  	-- self.inputs[1][{{},{1}}] = 1 
  	-- local temp_out = model:forward({temp_inp['im']:cuda(),temp_inp['rois']:cuda()})
  	local out = self.model:forward(self.inputs)
  	return out[1],out[2]
 end