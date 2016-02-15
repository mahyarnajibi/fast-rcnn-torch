require 'optim'
local threads = require 'threads'
ParallelTrainer = torch.class('detection.ParallelTrainer')
local utils = detection.GeneralUtils()
local _avg_cls_loss
local _avg_reg_loss
local _cls_criterion
local _reg_criterion
local _timer
local _batcher
local _roi_means
local _roi_stds
local _db_name
local _parameters
local _gradParameters
local _inputs
local _labels
local _loss_weights
local _log_path
local _logger
local _optim_state
local _iter_show = 20
function ParallelTrainer:__init(cls_criterion,reg_criterion,roi)
	-- Initializing...
	_avg_cls_loss = 0
	_avg_reg_loss = 0
	network:training()
	network:initialize_for_training()
	_cls_criterion = cls_criterion
	_reg_criterion = reg_criterion
	_timer = torch.Timer()
	_batcher = detection.Batcher(roi)
	_roi_means = torch.zeros(4,1):cat(roi.means:view(-1,1),1):cuda()
	_roi_stds = torch.zeros(4,1):cat(roi.stds:view(-1,1),1):cuda()
	_db_name = roi.db_name
	_parameters, _gradParameters = network:getParameters()
	_inputs = torch.CudaTensor()
	_labels = torch.CudaTensor()
	_loss_weights = torch.CudaTensor()
	local _log_name = network.name .. '_' .. _db_name .. os.date('_%m.%d_%H.%M')
	_log_path = paths.concat(config.log_path,_log_name)
	_logger = optim.Logger(_log_path)
	local localConfig = config
	-- Create a pool of threads
	if config.n_threads > 1 then
		self._thread_pool = threads( config.n_threads, 
			function ()
				require 'torch'
				require 'detection'
				require 'nn'
				require 'cudnn'
				require 'image'
				require 'optim'
				config = localConfig -- Setting the global config for threads
			end,
			function(idx)
				batcher = _batcher--upvalue
				tid = idx
				print(string.format('Worker thread with id = %d is created!', tid))
		end)
	end
end


function ParallelTrainer:_computeGamma(lr_range,n_iter)
	return math.exp((1/n_iter) * math.log(lr_range[2]/lr_range[1]))
end

function ParallelTrainer:_getLR(iter,base_lr,gamma)
	return base_lr * gamma ^ (iter)
end


function ParallelTrainer:train()
	local num_regimes = #config.optim_regimes
	local start_iter = 1
	local end_iter = 0

	-- Loading the optim state if we are resume training
	if config.resume_training then
		print('Loading the optmizer state for resuming training...')
		local optim_path = paths.concat(config.save_path,network.name .. '_' ..  _db_name .. '.t7') 
		_optimState = torch.load(optim_path)
	else
		_optimState = {}
	end


	for r = 1,num_regimes do

		-- Set the optim parameters...
		_optimState.learningRateDecay = 0.0
		_optimState.momentum = config.optim_momentum
		_optimState.weightDecay = config.optim_regimes[r][3]
		_optimState.dampening = 0.0
		_optimState.evalCounter = 0.0

		-- Compute the current gamma
		local gamma
		if config.optim_lr_decay_policy == 'exp' then
			gamma = self:_computeGamma(config.optim_regimes[r][2],config.optim_regimes[r][1])
		end

		end_iter = end_iter + config.optim_regimes[r][1]
		for i=start_iter,end_iter do

			local lr
			if config.optim_lr_decay_policy == 'fixed' then
				lr = config.optim_regimes[r][2]
			else
				lr = self:_getLR(i-start_iter,config.optim_regimes[r][2][1],gamma)
			end

			local img_ids = _batcher:getNextIds()
			if config.n_threads > 1 then
				self._thread_pool:addjob(
					function()
						iter = i -- upvlaue
						my_ids = img_ids --upvalue
						cur_lr = lr
						local inputs,labels,loss_weights = batcher:getBatch(my_ids)
						return inputs,labels,loss_weights,iter,cur_lr
					end,
					self._trainBatch
					)

				  if self._thread_pool:haserror() then
         			self._thread_pool:synchronize()
     			  end
			else
				local inputs,labels,loss_weights = _batcher:getBatch(img_ids)
				self._trainBatch(inputs,labels,loss_weights,i,lr)
			end
		end
		start_iter = end_iter + 1
	end
	self._thread_pool:synchronize()
	cutorch.synchronize()
end



function ParallelTrainer._trainBatch(inputs_cpu,labels_cpu,loss_weights_cpu,iter,lr)
	_inputs,inputs_cpu = utils:recursiveResizeAsCopyTyped(_inputs,inputs_cpu,'torch.CudaTensor')
	_labels,labels_cpu = utils:recursiveResizeAsCopyTyped(_labels,labels_cpu,'torch.CudaTensor')
	_loss_weights,loss_weights_cpu = utils:recursiveResizeAsCopyTyped(_loss_weights,loss_weights_cpu,'torch.CudaTensor')

	-- Perform sgd
	local cls_err,reg_err,outputs
	feval = function(x)
		network.model:zeroGradParameters()
		 -- Zero label is the background class!
		local outputs = network.model:forward(_inputs)
		cls_err = _cls_criterion:forward(outputs[1],_labels[1]:view(-1)+1)
		reg_err = _reg_criterion:forward(outputs[2],{_labels[2],_loss_weights})
		local cls_grad_out = _cls_criterion:backward(outputs[1],_labels[1]:view(-1)+1) 
		local reg_grad_out = _reg_criterion:backward(outputs[2],{_labels[2],_loss_weights})
		network.model:backward(_inputs,{cls_grad_out,reg_grad_out})
		return cls_err+reg_err, _gradParameters
	end
	_timer:reset()

	-- Set the learning rate
    _optimState.learningRate = lr
	optim.sgd(feval, _parameters, _optimState)


    local training_time =_timer:time().real
	_avg_cls_loss = _avg_cls_loss + cls_err
	_avg_reg_loss = _avg_reg_loss + reg_err


	if iter%_iter_show==0 then
		print(string.format('Iteration = %d, Classification Loss = %2.2f, Regression Loss = %2.2f, Time = %2.2f',iter, _avg_cls_loss/_iter_show, _avg_reg_loss/_iter_show,training_time))
		_avg_cls_loss = 0
		_avg_reg_loss = 0
	end

	-- Log state
	_logger:add{['classification loss'] = cls_err,
            ['regression loss'] = reg_err}
	if iter % config.optim_snapshot_iters ==0 then
		-- Saving the network
		print('Saving the network for iter '.. iter)
		local file_name = network.name .. '_' .. _db_name .. '_iter' .. '_' .. iter .. os.date('_%m.%d_%H.%M')
		local save_path = paths.concat(config.save_path ,file_name)
		local net_path = save_path .. '.t7'
		network:save(net_path,_roi_means,_roi_stds)

		-- Save the optim state
		local optim_path = paths.concat(config.save_path,network.name .. '_' ..  _db_name .. '.t7') 
		torch.save(optim_path,_optimState)

		-- Saving parameters
		local txt_path = save_path .. '.txt'
		local file = io.open(txt_path,'w')
		file:write('The file trained on ' .. os.date(' %m.%d.%Y %H:%M:%S') .. '\n\n')
		file:write('Log file for this run: '.. _log_path)
		file:write(utils:table2str(config))
		file:close()
		print('Network saved in: ' .. save_path .. '\n\n')
		_logger:style{['classification loss'] = '-',   -- define styles for plots
                 ['regression loss'] = '-'}
    	_logger:plot() 
	end
end