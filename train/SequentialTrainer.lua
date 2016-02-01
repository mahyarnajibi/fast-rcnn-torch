require 'optim'
require 'paths'
SequentialTrainer = torch.class('detection.SequentialTrainer')
local utils = detection.GeneralUtils()
local _iter_show = 20
function SequentialTrainer:__init(cls_criterion,reg_criterion,roi)
	-- Initializing...
	self._avg_cls_loss = 0
	self._avg_reg_loss = 0
	network:training()
	network:initialize_for_training()
	self._cls_criterion = cls_criterion
	self._reg_criterion = reg_criterion
	self._timer = torch.Timer()
	self._batcher = detection.Batcher(roi:get_roidb())
	self._roi_means = torch.zeros(4,1):cat(roi.means:view(-1,1),1):cuda()
	self._roi_stds = torch.zeros(4,1):cat(roi.stds:view(-1,1),1):cuda()
	self._db_name = roi.db_name
	self._parameters, self._gradParameters = network:getParameters()
	self._inputs = torch.CudaTensor()
	self._labels = torch.CudaTensor()
	self._loss_weights = torch.CudaTensor()
	local _log_name = network.name .. '_' .. self._db_name .. os.date('_%m.%d_%H.%M')
	self._log_path = paths.concat(config.log_path,_log_name)
	self._logger = optim.Logger(self._log_path)
end


function SequentialTrainer:train()
	local num_regimes = #config.optim_regimes
	local start_iter = 1
	local end_iter = 0
	for r = 1,num_regimes do
		self._optimState = {
	         learningRate = config.optim_regimes[r][2],
	         learningRateDecay = 0.0,
	         momentum = config.optim_momentum,
	         dampening = 0.0,
	         weightDecay = config.optim_regimes[r][3],
	         evalCounter = 0.0
      	}
		end_iter = end_iter + config.optim_regimes[r][1]
		for i=start_iter,end_iter do
			local inputs,labels,loss_weights = self._batcher:getNextBatch()
			self:_trainBatch(inputs,labels,loss_weights,i)
		end
		start_iter = end_iter+1		
	end
end

function SequentialTrainer:_trainBatch(inputs_cpu,labels_cpu,loss_weights_cpu,iter)
	collectgarbage()

	-- transfer the cpu data into the gpu data
	self._inputs,inputs_cpu = utils:recursiveResizeAsCopyTyped(self._inputs,inputs_cpu,'torch.CudaTensor')
	self._labels,labels_cpu = utils:recursiveResizeAsCopyTyped(self._labels,labels_cpu,'torch.CudaTensor')
	self._loss_weights,loss_weights_cpu = utils:recursiveResizeAsCopyTyped(self._loss_weights,loss_weights_cpu,'torch.CudaTensor')

	-- Perform sgd
	local cls_err,reg_err,outputs
	feval = function(x)
		network.model:zeroGradParameters()
		 -- Zero label is the background class!
		local outputs = network.model:forward(self._inputs)
		cls_err = self._cls_criterion:forward(outputs[1],self._labels[1]:view(-1)+1)
		reg_err = self._reg_criterion:forward(outputs[2],{self._labels[2],self._loss_weights})
		local cls_grad_out = self._cls_criterion:backward(outputs[1],self._labels[1]:view(-1)+1) 
		local reg_grad_out = self._reg_criterion:backward(outputs[2],{self._labels[2],self._loss_weights})
		network.model:backward(self._inputs,{cls_grad_out,reg_grad_out})
		return cls_err+reg_err, self._gradParameters
	end

    self._timer:reset()
	optim.sgd(feval, self._parameters, self._optimState)
	self._avg_cls_loss = self._avg_cls_loss + cls_err
	self._avg_reg_loss = self._avg_reg_loss + reg_err

	local training_time = self._timer:time().real
	if iter%_iter_show==0 then
		print(string.format('Iteration = %d, Classification Loss = %2.2f, Regression Loss = %2.2f, Time = %2.2f',iter, self._avg_cls_loss/_iter_show, self._avg_reg_loss/_iter_show,training_time))
		self._avg_cls_loss = 0
		self._avg_reg_loss = 0
	end

	-- Log state
	self._logger:add{['classification loss'] = cls_err,
            ['regression loss'] = reg_err}
	if iter % config.optim_snapshot_iters ==0 then
		-- Saving the network
		print('Saving the network for iter '.. iter)
		local file_name = network.name .. '_' .. self._db_name .. '_iter' .. '_' .. iter .. os.date('_%m.%d_%H.%M')
		local save_path = paths.concat(config.save_path ,file_name)
		local net_path = save_path .. '.t7'
		network:save(net_path,self._roi_means,self._roi_stds)
		-- Saving parameters
		local txt_path = save_path .. '.txt'
		local file = io.open(txt_path,'w')
		file:write('The file trained on ' .. os.date(' %m.%d.%Y %H:%M:%S') .. '\n\n')
		file:write('Log file for this run: '.. self._log_path)
		file:write(utils:table2str(config))
		file:close()
		print('Network saved in: ' .. save_path .. '\n\n')
		self._logger:style{['classification loss'] = '-',   -- define styles for plots
                 ['regression loss'] = '-'}
    	self._logger:plot() 
	end
end