local NetworkWrapper = torch.class('detection.NetworkWrapper')
utils = detection.GeneralUtils()
ROI = detection.ROI()

function NetworkWrapper:__init(net,image_transformer)
  self.train = true
  self.image_transformer = image_transformer
  self.net = net
end

function NetworkWrapper:training()
  self.train = true
end

function NetworkWrapper:evaluate()
  self.train = false
end

function NetworkWrapper:processImages(input_imgs,do_flip)
  local output_imgs = self._feat[1]
  local num_images
  local im
  if self.train then
    num_images = #input_imgs
  else
    num_images = #config.scale
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
      scale = config.scale[math.random(1,#self.scale)]
    else
      scale = config.scale[i]
    end
    local flip = do_flip and (do_flip[i] == 1) or false
    if flip then
      im = image.hflip(im)
    end

    local im_size = im[1]:size()
    local im_size_min = math.min(im_size[1],im_size[2])
    local im_size_max = math.max(im_size[1],im_size[2])
    local im_scale = scale/im_size_min
    if torch.round(im_scale*im_size_max) > config.max_size then
       im_scale = config.max_size/im_size_max
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



function NetworkWrapper:prepare_inputs(imgs,bboxes,flip)
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
  self._feat[2] = ROI:projectImageROIs(bboxes,im_scales,flip,im_sizes,self.train)
  return self._feat
end


function NetworkWrapper:__tostring()
  local str = torch.type(self)
  str = str .. '\n  Image scales: [' .. table.concat(self.scale,', ')..']'
  str = str .. '\n  Max image size: ' .. self.max_size
  str = str .. '\n  Input area: ' .. self.inputArea
  return str
end



function NetworkWrapper:detect(im,boxes)

  local inputs = self:prepare_inputs(im,boxes)

  local scores,bbox_deltas = self.net:forward(inputs)
  local predicted_boxes = ROI:bbox_decode(boxes,bbox_deltas,{im:size()[2],im:size()[3]})

  self.scores,scores = utils:recursiveResizeAsCopyTyped(self.scores,scores,'torch.FloatTensor')
  return self.scores,predicted_boxes
end
