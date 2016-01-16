--------------------------------------------------------------------------------
-- utility functions for the evaluation part
--------------------------------------------------------------------------------

local GeneralUtils = torch.class('detection.GeneralUtils')


function GeneralUtils:__init()
  return self
end
-- can be replaced by the new torch.cat function
function GeneralUtils:joinTable(input,dim)
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

function GeneralUtils:shuffle(dim,tensor)
  local randperm = torch.randperm(tensor:size(dim)):long()
  return tensor:index(dim,randperm)
end

function GeneralUtils:tableDeepCopy(tab)
  if type(tab)=='userdata' and tab.clone ~= nil then return tab:clone() end -- for dealing with tensors
  if type(tab) ~= 'table' then return tab end
  local res = setmetatable({}, getmetatable(tab))
  for k, v in pairs(tab) do res[self:tableDeepCopy(k)] = self:tableDeepCopy(v) end
  return res
end

function GeneralUtils:logical2ind(logical)
  return torch.range(1,logical:numel())[logical:gt(0)]:long()
end


function GeneralUtils:recursiveResizeAsCopyTyped(t1,t2,type)
  if torch.type(t2) == 'table' then
    t1 = (torch.type(t1) == 'table') and t1 or {t1}
    for key,_ in pairs(t2) do
      t1[key], t2[key] = self:recursiveResizeAsCopyTyped(t1[key], t2[key], type)
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

function GeneralUtils:concat(t1,t2,dim)
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
function GeneralUtils:flipBoundingBoxes(bbox, im_width)
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

function GeneralUtils:keep_top_k(boxes,top_k)
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

function GeneralUtils:VOCap(rec,prec)

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

function GeneralUtils:boxoverlap(a,b)
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



function GeneralUtils:VOCevaldet(dataset,scored_boxes,cls)
  local num_pr = 0
  local energy = {}
  local correct = {}
  
  local count = 0
  
  for i=1, dataset:size() do   
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
        local o = self:boxoverlap(bbox,bbox_pred[{{1,4}}])
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

  ap = self:VOCap(recall, precision)
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
function GeneralUtils:convertCaffeModelToTorch(model,location)
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

function GeneralUtils:reshapeLastLinearLayer(model,nOutput)
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
function GeneralUtils:sanitize(net)
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



function GeneralUtils:nms(boxes, overlap)
  
    local pick = torch.LongTensor()

    if boxes:numel() == 0 then
      return pick
    end

    local x1 = boxes[{{},1}]
    local y1 = boxes[{{},2}]
    local x2 = boxes[{{},3}]
    local y2 = boxes[{{},4}]
    local s = boxes[{{},-1}]
    
    local area = boxes.new():resizeAs(s):zero()
    area:map2(x2,x1,function(xx,xx2,xx1) return xx2-xx1+1 end)
    area:map2(y2,y1,function(xx,xx2,xx1) return xx*(xx2-xx1+1) end)

    local vals, I = s:sort(1)

    pick:resize(s:size()):zero()
    local counter = 1
    local xx1 = boxes.new()
    local yy1 = boxes.new()
    local xx2 = boxes.new()
    local yy2 = boxes.new()

    local w = boxes.new()
    local h = boxes.new()

    while I:numel()>0 do 
      local last = I:size(1)
      local i = I[last]
      pick[counter] = i
      counter = counter + 1
      if last == 1 then
        break
      end
      I = I[{{1,last-1}}]
      
      xx1:index(x1,1,I)
      xx1:cmax(x1[i])
      yy1:index(y1,1,I)
      yy1:cmax(y1[i])
      xx2:index(x2,1,I)
      xx2:cmin(x2[i])
      yy2:index(y2,1,I)
      yy2:cmin(y2[i])
      
      w:resizeAs(xx2):zero()
      w:map2(xx2,xx1,function(xx,xxx2,xxx1) return math.max(xxx2-xxx1+1,0) end)
      h:resizeAs(yy2):zero()
      h:map2(yy2,yy1,function(xx,yyy2,yyy1) return math.max(yyy2-yyy1+1,0) end)
      
      local inter = w
      inter:cmul(h)

      local o = h
      xx1:index(area,1,I)
      torch.cdiv(o,inter,xx1+area[i]-inter)
      I = I[o:le(overlap)]
    end

    pick = pick[{{1,counter-1}}]
    return pick
end


function GeneralUtils:visualize_detections(im,boxes,scores,thresh,cl_names)
  local ok = pcall(require,'qt')
  if not ok then
    error('You need to run visualize_detections using qlua')
  end
  require 'qttorch'
  require 'qtwidget'

  -- select best scoring boxes without background
  local max_score,idx = scores[{{},{2,-1}}]:max(2)

  local idx_thresh = max_score:gt(thresh)
  max_score = max_score[idx_thresh]
  idx = idx[idx_thresh]

  local r = torch.range(1,boxes:size(1)):long()
  local rr = r[idx_thresh]
  if rr:numel() == 0 then
    error('No detections with a score greater than the specified threshold')
  end
  local boxes_thresh = boxes:index(1,rr)
  
  local keep = self:nms(torch.cat(boxes_thresh:float(),max_score:float(),2),0.3)
  
  boxes_thresh = boxes_thresh:index(1,keep)
  max_score = max_score:index(1,keep)
  idx = idx:index(1,keep)

  local num_boxes = boxes_thresh:size(1)
  local widths  = boxes_thresh[{{},3}] - boxes_thresh[{{},1}]
  local heights = boxes_thresh[{{},4}] - boxes_thresh[{{},2}]

  local x,y = im:size(3),im:size(2)
  local w = qtwidget.newwindow(x,y,"Detections")
  local qtimg = qt.QImage.fromTensor(im)
  w:image(0,0,x,y,qtimg)
  local fontsize = 15

  for i=1,num_boxes do
    local x,y = boxes_thresh[{i,1}],boxes_thresh[{i,2}]
    local width,height = widths[i], heights[i]
    
    -- add bbox
    w:rectangle(x,y,width,height)
    
    -- add score
    w:moveto(x,y+fontsize)
    w:setcolor("red")
    w:setfont(qt.QFont{serif=true,italic=true,size=fontsize,bold=true})
    if cl_names then
      w:show(string.format('%s: %.2f',cl_names[idx[i]],max_score[i]))
    else
      w:show(string.format('%d: %.2f',idx[i],max_score[i]))
    end
  end
  w:setcolor("red")
  w:setlinewidth(2)
  w:stroke()
  return w
end



function GeneralUtils:print_scores(dataset,res)
  print('Results:')
  -- print class names
  io.write('|')
  for i = 1, dataset.num_classes do
    io.write(('%5s|'):format(dataset.classes[i]))
  end
  io.write('\n|')
  -- print class scores
  for i = 1, dataset.num_classes do
    local l = #dataset.classes[i] < 5 and 5 or #dataset.classes[i]
    local l = res[i] == res[i] and l-5 or l-3
    if l > 0 then
      io.write(('%.3f%'..l..'s|'):format(res[i],' '))
    else
      io.write(('%.3f|'):format(res[i]))
    end
  end
  io.write('\n')
  io.write(('mAP: %.4f\n'):format(res:mean(1)[1]))
end
