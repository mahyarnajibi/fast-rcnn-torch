--------------------------------------------------------------------------------
-- utility functions for the evaluation part
--------------------------------------------------------------------------------

-- can be replaced by the new torch.cat function
local function joinTable(input,dim)
  local size = torch.LongStorage()
  local is_ok = false
  for i=1,#input do
    local currentOutput = input[i]
    if currentOutput:numel() > 0 then
      if not is_ok then
        size:resize(currentOutput:dim()):copy(currentOutput:size())
        is_ok = true
      else
        size[dim] = size[dim] + currentOutput:size(dim)
      end    
    end
  end
  local output = input[1].new():resize(size)
  local offset = 1
  for i=1,#input do
    local currentOutput = input[i]
    if currentOutput:numel() > 0 then
      output:narrow(dim, offset,
                    currentOutput:size(dim)):copy(currentOutput)
      offset = offset + currentOutput:size(dim)
    end
  end
  return output
end

local function recursiveResizeAsCopyTyped(t1,t2,type)
  if torch.type(t2) == 'table' then
    t1 = (torch.type(t1) == 'table') and t1 or {t1}
    for key,_ in pairs(t2) do
      t1[key], t2[key] = recursiveResizeAsCopyTyped(t1[key], t2[key], type)
    end
  elseif torch.isTensor(t2) then
    local type = type or t2:type()
    t1 = torch.isTypeOf(t1,type) and t1 or torch.Tensor():type(type)
    t1:resize(t2:size()):copy(t2)
  else
    error("expecting nested tensors or tables. Got "..
    torch.type(t1).." and "..torch.type(t2).." instead")
  end
  return t1, t2
end

local function concat(t1,t2,dim)
  local out
  assert(t1:type() == t2:type(),'tensors should have the same type')
  if t1:dim() > 0 and t2:dim() > 0 then
    dim = dim or t1:dim()
    out = torch.cat(t1,t2,dim)
  elseif t1:dim() > 0 then
    out = t1:clone()
  else
    out = t2:clone()
  end
  return out
end

-- modify bbox input
local function flipBoundingBoxes(bbox, im_width)
  if bbox:dim() == 1 then 
    local tt = bbox[1]
    bbox[1] = im_width-bbox[3]+1
    bbox[3] = im_width-tt     +1
  else
    local tt = bbox[{{},1}]:clone()
    bbox[{{},1}]:fill(im_width+1):add(-1,bbox[{{},3}])
    bbox[{{},3}]:fill(im_width+1):add(-1,tt)
  end
end

--------------------------------------------------------------------------------

local function keep_top_k(boxes,top_k)
  local X = joinTable(boxes,1)
  if X:numel() == 0 then
    return
  end
  local scores = X[{{},-1}]:sort(1,true)
  local thresh = scores[math.min(scores:numel(),top_k)]
  for i=1,#boxes do
    local bbox = boxes[i]
    if bbox:numel() > 0 then
      local idx = torch.range(1,bbox:size(1)):long()
      local keep = bbox[{{},-1}]:ge(thresh)
      idx = idx[keep]
      if idx:numel() > 0 then
        boxes[i] = bbox:index(1,idx)
      else
        boxes[i]:resize()
      end
    end
  end
  return boxes, thresh
end

--------------------------------------------------------------------------------
-- evaluation
--------------------------------------------------------------------------------

local function VOCap(rec,prec)

  local mrec = rec:totable()
  local mpre = prec:totable()
  table.insert(mrec,1,0); table.insert(mrec,1)
  table.insert(mpre,1,0); table.insert(mpre,0)
  for i=#mpre-1,1,-1 do
      mpre[i]=math.max(mpre[i],mpre[i+1])
  end
  
  local ap = 0
  for i=1,#mpre-1 do
    if mrec[i] ~= mrec[i+1] then
      ap = ap + (mrec[i+1]-mrec[i])*mpre[i+1]
    end
  end
  return ap
end

--------------------------------------------------------------------------------

local function boxoverlap(a,b)
  local b = b.xmin and {b.xmin,b.ymin,b.xmax,b.ymax} or b
    
  local x1 = a:select(2,1):clone()
  x1[x1:lt(b[1])] = b[1] 
  local y1 = a:select(2,2):clone()
  y1[y1:lt(b[2])] = b[2]
  local x2 = a:select(2,3):clone()
  x2[x2:gt(b[3])] = b[3]
  local y2 = a:select(2,4):clone()
  y2[y2:gt(b[4])] = b[4]
  
  local w = x2-x1+1;
  local h = y2-y1+1;
  local inter = torch.cmul(w,h):float()
  local aarea = torch.cmul((a:select(2,3)-a:select(2,1)+1) ,
                           (a:select(2,4)-a:select(2,2)+1)):float()
  local barea = (b[3]-b[1]+1) * (b[4]-b[2]+1);
  
  -- intersection over union overlap
  local o = torch.cdiv(inter , (aarea+barea-inter))
  -- set invalid entries to 0 overlap
  o[w:lt(0)] = 0
  o[h:lt(0)] = 0
  
  return o
end

--------------------------------------------------------------------------------

local function VOCevaldet(dataset,scored_boxes,cls)
  local num_pr = 0
  local energy = {}
  local correct = {}
  
  local count = 0
  
  for i=1,dataset:size() do   
    local ann = dataset:getAnnotation(i)   
    local bbox = {}
    local det = {}
    for idx,obj in ipairs(ann.object) do
      if obj.name == cls and obj.difficult == '0' then
        table.insert(bbox,{obj.bndbox.xmin,obj.bndbox.ymin,
                           obj.bndbox.xmax,obj.bndbox.ymax})
        table.insert(det,0)
        count = count + 1
      end
    end
    
    bbox = torch.Tensor(bbox)
    det = torch.Tensor(det)
    
    local num = scored_boxes[i]:numel()>0 and scored_boxes[i]:size(1) or 0
    for j=1,num do
      local bbox_pred = scored_boxes[i][j]
      num_pr = num_pr + 1
      table.insert(energy,bbox_pred[5])
      
      if bbox:numel()>0 then
        local o = boxoverlap(bbox,bbox_pred[{{1,4}}])
        local maxo,index = o:max(1)
        maxo = maxo[1]
        index = index[1]
        if maxo >=0.5 and det[index] == 0 then
          correct[num_pr] = 1
          det[index] = 1
        else
          correct[num_pr] = 0
        end
      else
          correct[num_pr] = 0        
      end
    end
    
  end
  
  if #energy == 0 then
    return 0,torch.Tensor(),torch.Tensor()
  end
  
  energy = torch.Tensor(energy)
  correct = torch.Tensor(correct)
  
  local threshold,index = energy:sort(true)

  correct = correct:index(1,index)

  local n = threshold:numel()
  
  local recall = torch.zeros(n)
  local precision = torch.zeros(n)

  local num_correct = 0

  for i = 1,n do
      --compute precision
      num_positive = i
      num_correct = num_correct + correct[i]
      if num_positive ~= 0 then
          precision[i] = num_correct / num_positive;
      else
          precision[i] = 0;
      end
      
      --compute recall
      recall[i] = num_correct / count
  end

  ap = VOCap(recall, precision)
  io.write(('AP = %.4f\n'):format(ap));

  return ap, recall, precision
end


--------------------------------------------------------------------------------
-- data preparation
--------------------------------------------------------------------------------

-- Caffe models are in BGR format, and they suppose the images range from 0-255.
-- This function modifies the model read by loadcaffe to use it in torch format
-- location is the postion of the first conv layer in the module. If you have
-- nested models (like sequential inside sequential), location should be a
-- table with as many elements as the depth of the network.
local function convertCaffeModelToTorch(model,location)
  local location = location or {1}
  local m = model
  for i=1,#location do
    m = m:get(location[i])
  end
  local weight = m.weight
  local weight_clone = weight:clone()
  local nchannels = weight:size(2)
  for i=1,nchannels do
    weight:select(2,i):copy(weight_clone:select(2,nchannels+1-i))
  end
  weight:mul(255)
end


--------------------------------------------------------------------------------
-- nn
--------------------------------------------------------------------------------

local function reshapeLastLinearLayer(model,nOutput)
  local layers = model:findModules('nn.Linear')
  local layer = layers[#layers]
  local nInput = layer.weight:size(2)
  layer.gradBias:resize(nOutput):zero()
  layer.gradWeight:resize(nOutput,nInput):zero()
  layer.bias:resize(nOutput)
  layer.weight:resize(nOutput,nInput)
  layer:reset()
end

-- borrowed from https://github.com/soumith/imagenet-multiGPU.torch/blob/master/train.lua
-- clear the intermediate states in the model before saving to disk
-- this saves lots of disk space
local function sanitize(net)
  local list = net:listModules()
  for _,val in ipairs(list) do
    for name,field in pairs(val) do
      if torch.type(field) == 'cdata' then val[name] = nil end
      if name == 'homeGradBuffers' then val[name] = nil end
      if name == 'input_gpu' then val['input_gpu'] = {} end
      if name == 'gradOutput_gpu' then val['gradOutput_gpu'] = {} end
      if name == 'gradInput_gpu' then val['gradInput_gpu'] = {} end
      if (name == 'output' or name == 'gradInput') then
        val[name] = field.new()
      end
    end
  end
end

--------------------------------------------------------------------------------
-- packaging
--------------------------------------------------------------------------------

local utils = {}

utils.keep_top_k = keep_top_k
utils.VOCevaldet = VOCevaldet
utils.VOCap = VOCap
utils.convertCaffeModelToTorch = convertCaffeModelToTorch
utils.reshapeLastLinearLayer = reshapeLastLinearLayer
utils.sanitize = sanitize
utils.recursiveResizeAsCopyTyped = recursiveResizeAsCopyTyped
utils.flipBoundingBoxes = flipBoundingBoxes
utils.concat = concat
utils.boxoverlap = boxoverlap

return utils


