local WeightedSmoothL1Criterion, parent = torch.class('detection.WeightedSmoothL1Criterion', 'nn.Criterion')

function WeightedSmoothL1Criterion:__init(sizeAverage)
   parent.__init(self)
   -- Local SmoothL1Loss
   self._smoothL1Loss = nn.SmoothL1Criterion():cuda()
   self._smoothL1Loss.sizeAverage = false
end

function WeightedSmoothL1Criterion:updateOutput(input, target)
	-- input contains the output of the network
	-- target has the labels as the first element and loss weights as the second element
   local repeat_size = input:size(2) / target[1]:size(2) 
	local targets = target[1]:repeatTensor(1,repeat_size)
	local weights = target[2]
   local n_row = targets:size(1)
	self.output = self._smoothL1Loss:forward(input[weights],targets[weights])	
   self.output = self.output / n_row
   return self.output
end

function WeightedSmoothL1Criterion:updateGradInput(input, target)
   local repeat_size = input:size(2) / target[1]:size(2) 
   local targets = target[1]:repeatTensor(1,repeat_size)
   local weights = target[2]
   local n_row = targets:size(1)
   local smoothL1grad = self._smoothL1Loss:backward(input[weights],targets[weights])/n_row
   self.gradInput:resizeAs(input):zero()
   self.gradInput[weights] = smoothL1grad
   return self.gradInput
end