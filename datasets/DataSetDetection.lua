local utilities = detection.GeneralUtils()

local DataSetDetection = torch.class('detection.DataSetDetection')
DataSetDetection._isDataSet = true

function DataSetDetection:__init()
  self.classes = nil
  self.num_classes = nil
  self.image_set = nil
  self.dataset_name = nil
end

function DataSetDetection:getImage(i)
end

function DataSetDetection:getImageSize(i)
end

function DataSetDetection:getAnnotation(i)
end

function DataSetDetection:getROIBoxes(i)
end

function DataSetDetection:getGTBoxes(i)
end

function DataSetDetection:size()
  return #self.img_ids
end

function DataSetDetection:nclass()

end

function DataSetDetection:evaluate()
end



function DataSetDetection:__tostring__()
  local str = torch.type(self)
  str = str .. '\n  Dataset Name: ' .. self.dataset_name
  str = str .. '\n  ImageSet: '.. self.image_set
  str = str .. '\n  Number of images: '.. self:size()
  str = str .. '\n  Classes:'
  for k,v in ipairs(self.classes) do
    str = str .. '\n    '..v
  end
  return str
end

function DataSetDetection:bestOverlap(all_boxes, gt_boxes, gt_classes)
  local num_total_boxes = all_boxes:size(1)
  local num_gt_boxes = gt_boxes:dim() > 0 and gt_boxes:size(1) or 0
  local overlap_class = torch.FloatTensor(num_total_boxes,self.num_classes):zero()
  local overlap = torch.FloatTensor(num_total_boxes,num_gt_boxes):zero()
  for idx=1,num_gt_boxes do
    local o = utilities:boxoverlap(all_boxes,gt_boxes[idx])
    local tmp = overlap_class[{{},gt_classes[idx]}] -- pointer copy
    tmp[tmp:lt(o)] = o[tmp:lt(o)]
    overlap[{{},idx}] = o
  end
  -- get max class overlap
  --rec.overlap,rec.label = rec.overlap:max(2)
  --rec.overlap = torch.squeeze(rec.overlap,2)
  --rec.label   = torch.squeeze(rec.label,2)
  --rec.label[rec.overlap:eq(0)] = 0
  local correspondance
  if num_gt_boxes > 0 then
    overlap,correspondance = overlap:max(2)
    overlap = torch.squeeze(overlap,2)
    correspondance   = torch.squeeze(correspondance,2)
    correspondance[overlap:eq(0)] = 0
  else
    overlap = torch.FloatTensor(num_total_boxes):zero()
    correspondance = torch.LongTensor(num_total_boxes):zero()
  end
  return overlap, correspondance, overlap_class
end

function DataSetDetection:attachProposals(i)

  local boxes = self:getROIBoxes(i)
  local gt_boxes,gt_classes,valid_objects,anno = self:getGTBoxes(i)

  local all_boxes = utilities:concat(gt_boxes,boxes,1)

  local num_boxes = boxes:dim() > 0 and boxes:size(1) or 0
  local num_gt_boxes = #gt_classes
  
  local rec = {}
  rec.gt = utilities:concat(torch.ByteTensor(num_gt_boxes):fill(1),
                  torch.ByteTensor(num_boxes):fill(0)    )
  
  rec.overlap, rec.correspondance, rec.overlap_class =
                    self:bestOverlap(all_boxes,gt_boxes,gt_classes)
  rec.label = torch.IntTensor(num_boxes+num_gt_boxes):fill(0)
  for idx=1,(num_boxes+num_gt_boxes) do
    local corr = rec.correspondance[idx]
    if corr > 0 then
      rec.label[idx] = gt_classes[corr]
    end
  end
  
  rec.boxes = all_boxes
  rec.class = utilities:concat(torch.CharTensor(gt_classes),
                     torch.CharTensor(num_boxes):fill(0))



  if self.save_objs then
    rec.objects = {}
    for _,idx in pairs(valid_objects) do
      table.insert(rec.objects,anno.object[idx])
    end
  end
  
  function rec:size()
    return (num_boxes+num_gt_boxes)
  end
  
  rec.image_size = self:getImageSize(i)

  rec.flipped = false
  rec.image_path = string.format(self.imgpath,self.img_ids[i])

  return rec
end

