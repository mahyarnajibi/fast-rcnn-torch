require 'lfs'
local CaffeModelConverter = torch.class('detection.CaffeModelConverter')

function CaffeModelConverter:__init(torchmodel,proto_path,caffemodel_path,model_name,save_path)
	self.model_name = model_name or 'caffemodel.t7'
	self.save_path = save_path
	self.proto_path = proto_path
	self.caffemodel_path = caffemodel_path
	self.torchmodel = torchmodel
end 

function CaffeModelConverter:convert()
	
	-- Reading Torch Model
	local torchmodel = self.torchmodel


	-- Copying weights from caffe
	local weightloader = detection.CaffeLoader(torchmodel,self.proto_path,self.caffemodel_path)
	local model = weightloader:load()

	-- Saving torch model
	if self.save_path~=nil then
		lfs.mkdir(self.save_path)
		torch.save(self.save_path..self.model_name, model)
	end
	return model
end

