local heap = require 'utils.heap.binary_heap'
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
  self.net:get_net():evaluate()
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


function NetworkWrapper:testNetwork(db) 

  -- preparing the dataset
  local n_image = db:size()
  local n_class = db.num_classes
  db:loadROIDB()

  -- heuristics
  local max_per_set = 40* n_image
  local max_per_image = 100
  -- make torch and network to work in the evaluate mode

  self:evaluate()

  -- Container to save detections per class
  local all_detections = {}
  for i=1,n_class do all_detections[i] = {} end

  -- Adaptive class thresholds
  local thresholds = {}
  for i=1,n_class do thresholds[i] = -math.huge end

  -- Min heaps for maintaining max_per_set constraint
    local heaps={}
    for i=1,n_class do heaps[i] = heap() end

  -- Do the detection and save the results
  local detect_timer = torch.Timer()
  local misc_timer = torch.Timer()
  local avg_det_time = 0
  local avg_misc_time = 0


  for i=1,n_image do
    local im = db:getImage(i)
    -- Get the bounding boxes 
    local bboxes = db:getROIBoxes(i)
    local n_box = bboxes:size(1)


    -- Do the detection
    detect_timer:reset()
    local scores,pred_boxes = self:detect(im,bboxes)
    local det_time = detect_timer:time().real
    avg_det_time = avg_det_time + det_time

    misc_timer:reset()
    
    for j= 1,n_class do
        local class_scores = scores[{{},{j+1}}]
        local sel_inds = torch.range(1,n_box)[class_scores:ge(thresholds[j])]:long()
        if sel_inds:numel() == 0 then
          all_detections[j][i] = torch.FloatTensor()
        else

          local class_boxes = pred_boxes:index(1,sel_inds)[{{},{j*4+1,(j+1)*4}}]
          class_scores = class_scores:index(1,sel_inds)
          -- keep top k scoring boxes
          -- here you can use torch.topk if your torch is up-to-date
          local top_scores , top_inds = torch.sort(class_scores, 1, true)
          top_inds = top_inds[{{1,math.min(max_per_image,top_inds:numel())},{}}]:long()
          top_inds = top_inds:resize(top_inds:numel())

          class_scores = class_scores:index(1,top_inds):resize(top_inds:numel())
          class_boxes = class_boxes:index(1,top_inds)

          
          
          -- Push all values into the corresponding heap
          for k=1,class_scores:numel() do
            heaps[j]:add(class_scores[k])
          end

          -- Pop if you collected more than max_per_set
          if heaps[j]:getSize()> max_per_set then
            while heaps[j]:getSize()>max_per_set do
              heaps[j]:pop()
              thresholds[j] = heaps[j]:top()
            end
          end

          -- Save all detections for now

          all_detections[j][i] = class_boxes:float():cat(class_scores)

        end
    end
    local misc_time = misc_timer:time().real
    avg_misc_time = avg_misc_time +misc_time
    print('Image# = '.. i .. ', detection time = ' .. det_time .. ', misc time = ' .. misc_time)

  end
  avg_misc_time = avg_misc_time / n_image
  avg_det_time = avg_det_time / n_image
  print(n_image .. ' images detected! ' .. ',average detection time = ' .. avg_det_time .. ', average misc time = ' .. avg_misc_time)
  
  debugger.enter()
  local det_save_path = config.cache .. '/' .. db.dataset_name .. '_' .. db.image_set .. '_detections.t7'
  local thresholds_save_path = config.cache .. '/' .. db.dataset_name .. '_' .. db.image_set .. '_thresholds.t7'
  torch.save(det_save_path,all_detections)
  torch.save(thresholds_save_path,thresholds)
  print('Detections saved into '.. det_save_path)

  -- prune the detections and apply nms
  for i=1,n_class do
    for j=1,n_image do
      if all_detections[i][j]:numel()~=0 then 
        local n_box = all_detections[i][j]:size()[1]
        local sel_inds = torch.range(1,n_box)[all_detections[i][j][{{},-1}]:gt(thresholds[i])]:long()
        if sel_inds:numel() == 0 then
          all_detections[i][j] = torch.FloatTensor()
        else
          all_detections[i][j] = all_detections[i][j]:index(1,sel_inds)
          -- apply nms
          local nms_inds = utils:nms(all_detections[i][j],config.nms)
          all_detections[i][j] = all_detections[i][j]:index(1,nms_inds)
        end
      end
    end
  end


  -- Try to evaluate using the official eval functions
  local return_val = db:evaluate(all_detections)
  if return_val ~=0 then
    local test_result = torch.FloatTensor(n_class)
    for i=1,n_class do
      local cur_class = db.classes[i]
      test_result[i] = utils:VOCevaldet(db,all_detections[i],cur_class)
    end

    utils:print_scores(db,test_result)
  end
 
end