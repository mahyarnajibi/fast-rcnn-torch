require('utils.caffeloader')

local caffeModelConverter = torch.class('caffeModelConverter')

function caffeModelConverter:__init(save_path,torchmodel_path,proto_path,caffemodel_path,model_name)
	self.model_name = model_name or 'caffemodel.t7'
	self.save_path = save_path
	self.proto_path = proto_path
	self.caffemodel_path = caffemodel_path
	self.torchmodel_path = torchmodel_path
end 

function caffeModelConverter:convert()
	
	-- Reading Torch Model
	local torchmodel = dofile(self.torchmodel_path)()


	-- Copying weights from caffe
	local weightloader = caffeloader(torchmodel,self.proto_path,self.caffemodel_path)
	local model = weightloader:load()

	-- Saving torch model
	debugger.enter()
	torch.save(self.save_path..self.model_name, model)



end

