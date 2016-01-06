local ROIPooling,parent = torch.class('nnf.ROIPooling','nn.Module')

function ROIPooling:__init(W,H)
  parent.__init(self)
  self.W = W
  self.H = H
  self.pooler = {}--nn.SpatialAdaptiveMaxPooling(W,H)
  self.spatial_scale = 1
  self.gradInput = {torch.Tensor()}
end

function ROIPooling:setSpatialScale(scale)
  self.spatial_scale = scale
  return self
end

function ROIPooling:updateOutput(input)
  local data = input[1]
  local rois = input[2]

  local num_rois = rois:size(1)
  local s = data:size()
  local ss = s:size(1)
  self.output:resize(num_rois,s[ss-2],self.H,self.W)

  rois[{{},{2,5}}]:add(-1):mul(self.spatial_scale):add(1):round()
  rois[{{},2}]:cmin(s[ss])
  rois[{{},3}]:cmin(s[ss-1])
  rois[{{},4}]:cmin(s[ss])
  rois[{{},5}]:cmin(s[ss-1])

  -- element access is faster if not a cuda tensor
  if rois:type() == 'torch.CudaTensor' then
    self._rois = self._rois or torch.FloatTensor()
    self._rois:resize(rois:size()):copy(rois)
    rois = self._rois
  end

  if not self._type then self._type = self.output:type() end

  if #self.pooler < num_rois then
    local diff = num_rois - #self.pooler
    for i=1,diff do
      table.insert(self.pooler,nn.SpatialAdaptiveMaxPooling(self.W,self.H):type(self._type))
    end
  end

  for i=1,num_rois do
    local roi = rois[i]
    local im_idx = roi[1]
    local im = data[{im_idx,{},{roi[3],roi[5]},{roi[2],roi[4]}}]
    self.output[i] = self.pooler[i]:updateOutput(im)
  end
  return self.output
end

function ROIPooling:updateGradInput(input,gradOutput)
  local data = input[1]
  local rois = input[2]
  if rois:type() == 'torch.CudaTensor' then
    rois = self._rois
  end
  local num_rois = rois:size(1)
  local s = data:size()
  local ss = s:size(1)
  self.gradInput[1]:resizeAs(data):zero()

  for i=1,num_rois do
    local roi = rois[i]
    local im_idx = roi[1]
    local r = {im_idx,{},{roi[3],roi[5]},{roi[2],roi[4]}}
    local im = data[r]
    local g  = self.pooler[i]:updateGradInput(im,gradOutput[i])
    self.gradInput[1][r]:add(g)
  end
  return self.gradInput
end

function ROIPooling:type(type)
  parent.type(self,type)
  for i=1,#self.pooler do
    self.pooler[i]:type(type)
  end
  self._type = type
  return self
end
