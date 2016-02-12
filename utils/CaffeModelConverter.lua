local CaffeModelConverter = torch.class('detection.CaffeModelConverter')

function CaffeModelConverter:__init(torch_model_path,proto_path,caffemodel_path,model_name,save_path)
	self.model_name = model_name or 'caffemodel.t7'
	self.save_path = save_path
	self.proto_path = proto_path
	self.caffemodel_path = caffemodel_path
	self.torch_model_path  = torch_model_path
end 

function CaffeModelConverter:convert()
	
	-- Copying weights from caffe
	local weightloader = detection.CaffeLoader(self.torch_model_path,self.proto_path,self.caffemodel_path)
	local model = weightloader:load()

	-- Saving torch model
	if self.save_path~=nil then
		lfs.mkdir(self.save_path)
		torch.save(self.save_path..self.model_name, model)
	end
	return model
end

