require 'optim'
local threads = require 'threads'
ParallelTrainer = torch.class('detection.ParallelTrainer')
local utils = detection.GeneralUtils()
local _model
local _cls_criterion
local _reg_criterion
local _timer
local _parameters
local _gradParameters
local _inputs
local _labels
function ParallelTrainer:__init(model,cls_criterion,reg_criterion,roidb)
	
	-- Initializing...
	_model = model
	_cls_criterion = cls_criterion
	_reg_criterion = reg_criterion
	_timer = torch.Timer()
	self._batcher = detection.Batcher(roidb)
	_parameters, _gradParameters = _model:getParameters()
	_inputs = torch.CudaTensor()
	_labels = torch.CudaTensor()
	_loss_weights = torch.CudaTensor()
	local localConfig = config
	-- Create a pool of threads
	if config.n_threads > 1 then
		-- Create a global thread pool
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
				batcher = self._batcher--upvalue
				tid = idx
				print(string.format('Worker thread with id = %d is created!', tid))
		end)

	end

	--self._timer = torch.Timer()


end

function ParallelTrainer.model()
	return _model
end

function ParallelTrainer:train()

	local num_regimes = #config.optim_regimes
	for r = 1,num_regimes do
		cutorch.synchronize()
		_model:training()
		for i=1,config.optim_regimes[r][1] do

			local img_ids = self._batcher:getNextIds()
			self._thread_pool:addjob(
				function()
					iter = i -- upvlaue
					my_ids = img_ids
					regime_id = r --upvalue
					cur_ids = img_ids -- upvalue 
					local inputs,labels,loss_weights = batcher:getBatch(img_ids)
					--print(string.format('Training for iteration # %d',iter))
					return inputs,labels,loss_weights,regime_id,iter,my_ids
				end,
				self._trainBatch
				)
		end
		self._thread_pool:synchronize()
		cutorch.synchronize()

		
	end
end



function ParallelTrainer._trainBatch(inputs_cpu,labels_cpu,loss_weights_cpu,regime_id,iter,ids)
	cutorch.synchronize()
	collectgarbage()
	--print 'doing training...'
	-- transfer the cpu data into the gpu data
	_inputs,inputs_cpu = utils:recursiveResizeAsCopyTyped(_inputs,inputs_cpu,'torch.CudaTensor')
	_labels,labels_cpu = utils:recursiveResizeAsCopyTyped(_labels,labels_cpu,'torch.CudaTensor')
	_loss_weights,loss_weights_cpu = utils:recursiveResizeAsCopyTyped(_loss_weights,loss_weights_cpu,'torch.CudaTensor')

	-- Perform sgd

	local cls_err,reg_err,outputs
	--_parameters,_gradParameters = _model:getParameters()
	feval = function(x)
		_model:zeroGradParameters()

		 -- Zero label is the background class!
		local outputs = _model:forward(_inputs)
		cls_err = _cls_criterion:forward(outputs[1],_labels[1]:view(-1)+1)
		reg_err = _reg_criterion:forward(outputs[2],{_labels[2],_loss_weights})
		local cls_grad_out = _cls_criterion:backward(outputs[1],_labels[1]:view(-1)+1)
		local reg_grad_out = _reg_criterion:backward(outputs[2],{_labels[2],_loss_weights})
		_model:backward(_inputs,{cls_grad_out,reg_grad_out})

		return cls_err+reg_err, _gradParameters
	end
	optimState = {
         learningRate = config.optim_regimes[regime_id][2],
         learningRateDecay = 0.0,
         momentum = config.optim_momentum,
         dampening = 0.0,
         weightDecay = config.optim_regimes[regime_id][3],
         nesterov = true
      }
    _timer:reset()
    debugger.enter()
	optim.sgd(feval, _parameters, optimState)

   -- DataParallelTable's syncParameters
  	_model:apply(function(m) if m.syncParameters then m:syncParameters() end end)
	local training_time = _timer:time().real
	cutorch.synchronize()
	if iter%50 ==0 then
		print(string.format('Iteration = %d, Classification Error = %2.2f, Regression Error = %2.2f, Time = %2.2f',iter, cls_err, reg_err,training_time))
	end
end