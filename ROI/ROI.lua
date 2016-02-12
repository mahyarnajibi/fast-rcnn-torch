ROI = torch.class('detection.ROI')
utils = detection.GeneralUtils()
function ROI:__init()
    self._roidb = tds.Hash()
    self.n_class = 0
end


function ROI:bbox_decode(boxes,box_deltas,im_size)
  -- Function to decode the output of the network
  local eps = config.eps
  -- Check to see whether boxes are empty or not 
  if boxes:size()[1] == 0 then
    return torch.Tensor(0,boxes:size()[2]):zero()
  end

  box_deltas = box_deltas:double()
  local widths = boxes[{{},{3}}]:double() - boxes[{{},{1}}]:double() + eps
  local heights = boxes[{{},{4}}]:double() - boxes[{{},{2}}]:double() + eps
  local centers_x = boxes[{{},{1}}]:double() + widths * 0.5
  local centers_y = boxes[{{},{2}}]:double() + heights * 0.5

  local x_inds = torch.range(1,box_deltas:size()[2],4):long()
  local y_inds = torch.range(2,box_deltas:size()[2],4):long()
  local w_inds = torch.range(3,box_deltas:size()[2],4):long()
  local h_inds = torch.range(4,box_deltas:size()[2],4):long()

  local dx = box_deltas:index(2,x_inds)
  local dy = box_deltas:index(2,y_inds)
  local dw = box_deltas:index(2,w_inds)
  local dh = box_deltas:index(2,h_inds)


  local predicted_center_x = dx:cmul(widths:expand(dx:size())) + centers_x:expand(dx:size())
  local predicted_center_y = dy:cmul(heights:expand(dy:size())) + centers_y:expand(dy:size())
  local predicted_w = torch.exp(dw):cmul(widths:expand(dw:size()))
  local predicted_h = torch.exp(dh):cmul(heights:expand(dh:size()))

  local predicted_boxes = torch.Tensor(box_deltas:size()):zero()
  local half_w = predicted_w * 0.5
  local half_h = predicted_h * 0.5
  predicted_boxes:indexCopy(2,x_inds,predicted_center_x - half_w)
  predicted_boxes:indexCopy(2,y_inds,predicted_center_y -  half_h)
  predicted_boxes:indexCopy(2,w_inds,predicted_center_x + half_w)
  predicted_boxes:indexCopy(2,h_inds,predicted_center_y + half_h)
  predicted_boxes = self:_clip(predicted_boxes,im_size)

  return predicted_boxes
end

function ROI:get_roidb()
  return self._roidb
end


function ROI:create_roidb(db)
  -- Function that creates the roidb needed for training
    -- save the name of the database
    self.db_name = db.dataset_name
    self.n_class = db:nclass()
    db:loadROIDB()
    print('Creating the training dataset...')
    for i = 1,db:size() do
      -- Attach proposal information
      xlua.progress(i,db:size())
      self._roidb[i] = db:attachProposals(i)  
    end

    if config.use_flipped then
      self:_add_flipped()
    end

    -- Add targets
    self:_add_targets()
    -- Computing mean and stds for the targets
    local n_class = db:nclass()
    local counts = torch.zeros(n_class,1) + config.eps
    local means = torch.zeros(n_class,4)
    local stds = torch.zeros(n_class,4)

    for i=1,#self._roidb do
      local cur_targets = self._roidb[i].targets[{{},{2,5}}]
       local cur_labels = self._roidb[i].targets[{{},{1}}]
      for c = 1, n_class do
        local c_inds = utils:logical2ind(cur_labels:eq(c))
        if c_inds:numel()>0 then
          counts[c] = counts[c]+ c_inds:numel()
          means[c] = means[c] + cur_targets:index(1,c_inds):sum(1)
          stds[c] = stds[c] + cur_targets:index(1,c_inds):pow(2):sum(1)
        end
      end
    end

    means:cdiv(counts:expand(means:size()))
    stds= (stds:cdiv(counts:expand(stds:size())) - torch.pow(means,2)):sqrt()

    -- Do the normalization
    for i=1,#self._roidb do
      local cur_targets = self._roidb[i].targets[{{},{2,5}}]
      local cur_labels = self._roidb[i].targets[{{},{1}}]
      for c = 1,n_class do
        c_inds = utils:logical2ind(cur_labels:eq(c))
        if c_inds:numel()>0 then
          cur_targets:indexCopy(1,c_inds,cur_targets:index(1,c_inds) - means[c]:resize(1,means[c]:numel()):expand(torch.LongStorage{c_inds:numel(),means:size(2)}))
          cur_targets:indexCopy(1,c_inds,cur_targets:index(1,c_inds):cdiv(stds[c]:resize(1,stds[c]:numel()):expand(torch.LongStorage{c_inds:numel(),means:size(2)})))
        end
      end
    end
    self.means = means
    self.stds = stds
    -- Return the computed mean and std for reverting the normalization in further computations
    return means,stds
end



function ROI:_add_flipped()
  local n_recs = #self._roidb
  print('Adding flipped images to the training dataset...')
  local cur_loc = n_recs + 1
  for i=1,n_recs do
    xlua.progress(i,n_recs)
    local cur_roidb = self._roidb[i]
    local new_rec = utils:tableDeepCopy(cur_roidb)
    -- Adjust the coordinates
    local width = cur_roidb.image_size[1]
    new_rec.boxes[{{},{1}}] = -(cur_roidb.boxes[{{},{3}}]-width) + 1
    new_rec.boxes[{{},{3}}] = -(cur_roidb.boxes[{{},{1}}]-width) + 1
    new_rec.flipped = true
    self._roidb[cur_loc] = new_rec
    cur_loc = cur_loc + 1
  end

end


function ROI:_add_targets()

  for i=1,#self._roidb do
    self._roidb[i].targets = self:_get_targets(self._roidb[i])
  end
end


function ROI:_get_targets(roidb_entry)
    -- This function determines targets for the bounding boxes

    local boxes = roidb_entry.boxes:float()
    local gt_boxes = boxes:index(1,utils:logical2ind(roidb_entry.gt))
    local max_overlaps = roidb_entry.overlap

    local selected_ids = utils:logical2ind(max_overlaps:ge(config.bbox_threshold))
    assert(selected_ids:numel() > 0,'There is no ground truth box in one of the images!')
    local selected_boxes = boxes:index(1,selected_ids)

    -- Determine Targets
    local target_gts = gt_boxes:index(1,roidb_entry.correspondance:index(1,selected_ids):long())

    -- Encode the targets
    local targets = torch.DoubleTensor(boxes:size(1),5):zero() -- Regression label concatenated at the end
    local encoded_selected_targets = self:_bbox_encode(selected_boxes,target_gts)

    -- Concat regression targets with their labels
    local selected_labels = roidb_entry.label:index(1,selected_ids):double()
    targets:indexCopy(1,selected_ids,torch.cat(selected_labels,encoded_selected_targets,2))


    return targets
end

function ROI:_bbox_encode(bboxes,targets)
  -- The function encodes the delta in the bounding box regression
  local bb_widths = bboxes[{{},{3}}] - bboxes[{{},{1}}] + config.eps
  local bb_heights = bboxes[{{},{4}}] - bboxes[{{},{2}}] + config.eps
  local bb_ctr_x = bboxes[{{},{1}}] + bb_widths * 0.5
  local bb_ctr_y = bboxes[{{},{2}}] + bb_heights * 0.5

  local target_widths = targets[{{},{3}}] - targets[{{},{1}}] + config.eps
  local target_heights = targets[{{},{4}}] - targets[{{},{2}}] + config.eps
  local target_ctr_x = targets[{{},{1}}] + target_widths * 0.5
  local target_ctr_y = targets[{{},{2}}] + target_heights * 0.5

  local targets = torch.DoubleTensor(bboxes:size(1),4)
  targets[{{},{1}}] = torch.cdiv((target_ctr_x - bb_ctr_x), bb_widths)
  targets[{{},{2}}] = torch.cdiv((target_ctr_y - bb_ctr_y), bb_heights)
  targets[{{},{3}}] = torch.log(torch.cdiv(target_widths,bb_widths))
  targets[{{},{4}}] = torch.log(torch.cdiv(target_heights,bb_heights))

  return targets
end


function ROI:_clip(boxes,im_size)

    local x1_inds = torch.range(1,boxes:size()[2],4):long()
    local y1_inds = torch.range(2,boxes:size()[2],4):long()
    local x2_inds = torch.range(3,boxes:size()[2],4):long()
    local y2_inds = torch.range(4,boxes:size()[2],4):long()

    local x1 = boxes:index(2,x1_inds)
    local y1 = boxes:index(2,y1_inds)
    local x2 = boxes:index(2,x2_inds)
    local y2 = boxes:index(2,y2_inds)

    x1[x1:lt(1)] = 1
    y1[y1:lt(1)] = 1
    x2[x2:gt(im_size[2])] = im_size[2]
    y2[y2:gt(im_size[1])] = im_size[1]

    boxes:indexCopy(2,x1_inds,x1)
    boxes:indexCopy(2,y1_inds,y1)
    boxes:indexCopy(2,x2_inds,x2)
    boxes:indexCopy(2,y2_inds,y2)

    return boxes
end

-- The following function is borrowed and changed from https://github.com/fmassa/object-detection.torch
function ROI:projectImageROIs(im_rois,scales,train_mode)
  local rois = torch.FloatTensor()
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
  end
  return rois
end