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

  local scores,bbox_deltas = feat_provider:compute(self.model, inputs)
  debugger.enter()
  local predicted_boxes = feat_provider:bbox_decode(boxes,bbox_deltas,{im:size()[2],im:size()[3]})
  -- self.sm:forward(output0)

  self.scores,scores = recursiveResizeAsCopyTyped(self.scores,scores,'torch.FloatTensor')
  return self.scores,predicted_boxes
end
