local flipBoundingBoxes = paths.dofile('utils.lua').flipBoundingBoxes
local recursiveResizeAsCopyTyped = paths.dofile('utils.lua').recursiveResizeAsCopyTyped
local FRCNN = torch.class('nnf.FRCNN')
FRCNN._isFeatureProvider = true

local argcheck = require 'argcheck'
local initcheck = argcheck{
  pack=true,
  noordered=true,
  {name="scale",
   type="table",
   default={600},
   help="image scales"},
  {name="max_size",
   type="number",
   default=1000,
   help="maximum dimension of an image"},
  {name="inputArea",
   type="number",
   default=224^2,
   help="input area of the bounding box"},
  {name="image_transformer",
   type="nnf.ImageTransformer",
   default=nnf.ImageTransformer{},
   help="Class to preprocess input images"},
}


function FRCNN:__init(...)

  local opts = initcheck(...)
  for k,v in pairs(opts) do self[k] = v end

  self.train = true
end

function FRCNN:training()
  self.train = true
end

function FRCNN:evaluate()
  self.train = false
end

function FRCNN:processImages(input_imgs,do_flip)
  local output_imgs = self._feat[1]
  local num_images
  local im
  if self.train then
    num_images = #input_imgs
  else
    num_images = #self.scale
    im = self.image_transformer:preprocess(input_imgs[1])
  end

  local imgs = {}
  local im_sizes = {}
  local im_scales = {}

  for i=1,num_images do
    local scale
    if self.train then
      im = input_imgs[i]
      im = self.image_transformer:preprocess(im)
      scale = self.scale[math.random(1,#self.scale)]
    else
      scale = self.scale[i]
    end
    local flip = do_flip and (do_flip[i] == 1) or false
    if flip then
      im = image.hflip(im)
    end
    local im_size = im[1]:size()
    local im_size_min = math.min(im_size[1],im_size[2])
    local im_size_max = math.max(im_size[1],im_size[2])
    local im_scale = scale/im_size_min
    if torch.round(im_scale*im_size_max) > self.max_size then
       im_scale = self.max_size/im_size_max
    end
    local im_s = {torch.round(im_size[1]*im_scale),torch.round(im_size[2]*im_scale)}
    table.insert(imgs,image.scale(im,im_s[2],im_s[1]))
    table.insert(im_sizes,im_s)
    table.insert(im_scales,im_scale)
  end
  -- create single tensor with all images, padding with zero for different sizes
  im_sizes = torch.IntTensor(im_sizes)
  local max_shape = im_sizes:max(1)[1]
  output_imgs:resize(num_images,3,max_shape[1],max_shape[2]):zero()
  for i=1,num_images do
    output_imgs[i][{{},{1,imgs[i]:size(2)},{1,imgs[i]:size(3)}}]:copy(imgs[i])
  end
  return im_scales,im_sizes
end

function FRCNN:projectImageROIs(im_rois,scales,do_flip,imgs_size)
  local rois = self._feat[2]
  -- we consider two cases:
  -- During training, the scales are sampled randomly per image, so
  -- in the same image all the bboxes have the same scale, and we only
  -- need to take into account the different images that are provided.
  -- During testing, we consider that there is only one image at a time,
  -- and the scale for each bbox is the one which makes its area closest
  -- to self.inputArea
  if self.train or #scales == 1 then
    local total_bboxes = 0
    local cumul_bboxes = {0}
    for i=1,#scales do
      total_bboxes = total_bboxes + im_rois[i]:size(1)
      table.insert(cumul_bboxes,total_bboxes)
    end
    rois:resize(total_bboxes,5)
    for i=1,#scales do
      local idx = {cumul_bboxes[i]+1,cumul_bboxes[i+1]}
      rois[{idx,1}]:fill(i)
      rois[{idx,{2,5}}]:copy(im_rois[i]):add(-1):mul(scales[i]):add(1)
      if do_flip and do_flip[i] == 1 then
        flipBoundingBoxes(rois[{idx,{2,5}}],imgs_size[{i,2}])
      end
    end
  else -- not yet tested
    error('Multi-scale testing not yet tested')
    local scales = torch.FloatTensor(scales)
    im_rois = im_rois[1]
    local widths = im_rois[{{},3}] - im_rois[{{},1}] + 1
    local heights = im_rois[{{},4}] - im_rois[{{}, 2}] + 1

    local areas = widths * heights
    local scaled_areas = areas:view(-1,1) * scales:view(1,-1):pow(2)
    local diff_areas = scaled_areas:add(-1,self.inputArea):abs() -- no memory copy
    local levels = select(2, diff_areas:min(2))

    local num_boxes = im_rois:size(1)
    rois:resize(num_boxes,5)
    for i=1,num_boxes do
      local s = levels[i]
      rois[{i,{2,5}}]:copy(im_rois[i]):add(-1):mul(scales[s]):add(1)
      rois[{i,1}] = s
    end
  end
  return rois
end

function FRCNN:getFeature(imgs,bboxes,flip)
  self._feat = self._feat or {torch.FloatTensor(),torch.FloatTensor()}

  -- if it's in test mode, adapt inputs
  if torch.isTensor(imgs) then
    imgs = {imgs}
    if type(bboxes) == 'table' then
      bboxes = torch.FloatTensor(bboxes)
      bboxes = bboxes:dim() == 1 and bboxes:view(1,-1) or bboxes
    end
    bboxes = {bboxes}
    if flip == false then
      flip = {0}
    elseif flip == true then
      flip = {1}
    end
  end

  local im_scales, im_sizes = self:processImages(imgs,flip)
  self:projectImageROIs(bboxes,im_scales,flip,im_sizes)
  
  return self._feat
end

-- do the bbox regression
function FRCNN:postProcess(im,boxes,output)
  -- not implemented yet
  return output,boxes
end

function FRCNN:compute(model, inputs)
  local ttype = model.output:type() -- fix when doing bbox regression
  self.inputs,inputs = recursiveResizeAsCopyTyped(self.inputs,inputs,ttype)
  return model:forward(self.inputs)
end

function FRCNN:__tostring()
  local str = torch.type(self)
  str = str .. '\n  Image scales: [' .. table.concat(self.scale,', ')..']'
  str = str .. '\n  Max image size: ' .. self.max_size
  str = str .. '\n  Input area: ' .. self.inputArea
  return str
end


function FRCNN:bbox_decode(boxes,box_deltas)
  -- Function to decode the output of the network
  local eps = 1e-14
  local pred_boxes = torch.Tensor(box_deltas:size())
  local widths = boxes[{{},{3}}] - boxes[{{},{1}}] + eps
  local heights = boxes[{{},{4}}] - boxes[{{},{2}}] + eps
  local centers_x = boxes[{{},{1}] + 0.5 * widths
  local centers_y = boxes[{{},{3}}] + 0.5 * heights

  local x_inds = torch.range(1,box_deltas:size()[2],4):long()
  local y_inds = torch.range(2,box_deltas:size()[2],4):long()
  local w_inds = torch.range(3,box_deltas:size()[2],4):long()
  local h_inds = torch.range(4,box_deltas:size()[2],4):long()

  local dx = box_deltas:index(2,x_inds)
  local dy = box_deltas:index(2,y_inds)
  local dw = box_deltas:index(2,w_inds)
  local dh = box_deltas:index(2,h_inds)


  local predicted_center_x = dx * widths:expand(dx:size()) + centers_x:expand(dx:size())
  local predicted_center_y = dy * heights:expand(dy:size()) + centers_y:expand(dy:size())
  local predicted_w = torch.exp(dw) * widths:expand(dw:size())
  local predicted_h = torch.exp(dh) * heights:expand(dh:size())

  local predicted_boxes = torch.Tensor(box_deltas:size())
  predicted_boxes:indexCopy(2,x_inds,predicted_center_x)
  predicted_boxes:indexCopy(2,y_inds,predicted_center_y)
  predicted_boxes:indexCopy(2,w_inds,predicted_w)
  predicted_boxes:indexCopy(2,h_inds,predicted_h)

  return predicted_boxes
end