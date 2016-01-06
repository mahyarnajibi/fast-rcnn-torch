local ImageDetect = torch.class('nnf.ImageDetect')
local recursiveResizeAsCopyTyped = paths.dofile('utils.lua').recursiveResizeAsCopyTyped

function ImageDetect:__init(model, feat_provider)
  self.model = model
  self.feat_provider = feat_provider
  --self.sm = nn.SoftMax():cuda()
end

-- supposes boxes is in [x1,y1,x2,y2] format
function ImageDetect:detect(im,boxes)
  local feat_provider = self.feat_provider

  local inputs = feat_provider:getFeature(im,boxes)

  local output0 = feat_provider:compute(self.model, inputs)
  local output,boxes_p = feat_provider:postProcess(im,boxes,output0)
  --self.sm:forward(output0)

  self.output,output = recursiveResizeAsCopyTyped(self.output,output,'torch.FloatTensor')
  return self.output,boxes_p
end
