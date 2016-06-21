local Batcher = torch.class('detection.Batcher')
local inputMaker = detection.InputMaker()
local utils = detection.GeneralUtils()
function Batcher:__init(roi)
	self._roidb = roi:get_roidb()
	self._n_images = #self._roidb
	self._cur_image = 1
	self._rand_perm = torch.randperm(self._n_images)
	self.n_class = roi.n_class
end


function Batcher:_restartBatcher()
	self._cur_image = 1
	self._rand_perm = torch.randperm(self._n_images)
end

function Batcher:getNextIds()
	if self._cur_image +  config.img_per_batch - 1 > self._n_images then
		self:_restartBatcher()
	end  
	local img_inds = self._rand_perm[{{self._cur_image,self._cur_image + config.img_per_batch -1}}]
	self._cur_image = self._cur_image + config.img_per_batch
	return img_inds
end

function Batcher:getBatch(img_inds)
	return self:_process_batch(img_inds)
end

function Batcher:getNextBatch()
	local img_inds = self:getNextIds()
	return self:_process_batch(img_inds)
end

function Batcher:_process_batch(ids)

	local debug = false
	local n_images = ids:numel()
	local min_num_rois = math.huge
	-- Allocating memory for all images
	local imgs = {}
	local roi_batch = torch.FloatTensor()
	local labels = torch.IntTensor()
	local bbox_tagets = torch.FloatTensor()
	local loss_weights = torch.ByteTensor()
	-- Process images and create the batch
	for i=1,n_images do
		-- Load Image
		local cur_image = image.load(self._roidb[ids[i]].image_path,3,'float')
		if self._roidb[ids[i]].flipped then
			cur_image = image.hflip(cur_image)
		end
		-- Sample roidb
		local cur_ids, cur_rois, cur_labels, cur_bbox_targets = self:_select_rois(self._roidb[ids[i]])
		-- Compute loss weights
		
		if debug then
			local pos_samples = utils:logical2ind(cur_labels:ge(1))
			self:_visualize_batch(cur_image,cur_rois:index(1,pos_samples))
			local neg_samples = utils:logical2ind(cur_labels:eq(0))
			for i=1,neg_samples:numel() do
				self:_visualize_batch(cur_image,cur_rois[{{neg_samples[i]},{}}])
			end
		end


		local cur_loss_weights, cur_bbox_targets = self:_get_loss_weights(cur_bbox_targets)
		-- Pre-process image and sampled boxes
		local proc_image,cur_rois = inputMaker:process(cur_image,cur_rois)
		-- Add image ids
		if cur_rois:size(1) < min_num_rois then
			min_num_rois = cur_rois:size(1)
		end
		cur_rois[{{},{1}}] = torch.ones(cur_rois:size(1)) * i

		-- Add processed images and rois to the batch
		if i==1 then
			roi_batch = cur_rois
			labels = cur_labels
			bbox_targets = cur_bbox_targets
			loss_weights = cur_loss_weights
		else
			roi_batch = roi_batch:cat(cur_rois,1)
			labels = labels:cat(cur_labels,1)
			bbox_targets = bbox_targets:cat(cur_bbox_targets,1)
			loss_weights = loss_weights:cat(cur_loss_weights,1)
		end
		imgs[i] = proc_image
	end
	-- Make all batches the same size if we are in the multiGPU setting
	if config.nGPU > 1 then
		local keep_inds = torch.LongTensor(min_num_rois*n_images)
		for i =1,n_images do
			local cur_inds = utils:logical2ind(roi_batch[{{},{1}}]:eq(i))
			keep_inds[{{(i-1)*min_num_rois+1,i*min_num_rois}}] = cur_inds[{{1,min_num_rois}}]
		end
		roi_batch = roi_batch:index(1,keep_inds)
		labels = labels:index(1,keep_inds)
		bbox_targets = bbox_targets:index(1,keep_inds)
		loss_weights = loss_weights:index(1,keep_inds)
	end
	-- Batch images
	local max1 = -math.huge
	local max2 = -math.huge
	for i=1,n_images do
		local cur_size = {imgs[i]:size()[2],imgs[i]:size()[3]}
		if cur_size[1] > max1 then
			max1 = cur_size[1]
		end
		if cur_size[2] >max2 then
			max2 = cur_size[2]
		end
	end 
	local imgs_batch = torch.FloatTensor(n_images,3,max1,max2):zero()
	for i=1,n_images do
		imgs_batch[{{i},{},{1,imgs[i]:size(2)},{1,imgs[i]:size(3)}}] = imgs[i]
	end

	return {imgs_batch,roi_batch}, {labels,bbox_targets}, loss_weights

end

function Batcher:_get_loss_weights(bbox_targets)

	local loss_weights = torch.ByteTensor(bbox_targets:size(1),4*self.n_class+4):zero()
	local labels = bbox_targets[{{},{1}}]
	local bbox_targets = bbox_targets[{{},{2,5}}]
	for i=1,bbox_targets:size(1) do
		local cur_label = labels[i][1]
		if cur_label > 0 then
			loss_weights[{{i},{(cur_label)*4+1, (cur_label+1)*4}}] =1
		end
	end
	return loss_weights,bbox_targets
end

function Batcher:_select_rois(roidb_entry)

	local overlaps = roidb_entry.overlap
	local fg_inds = utils:logical2ind(overlaps:ge(config.fg_threshold))
	local bg_inds = utils:logical2ind(overlaps:ge(config.bg_threshold_lo):cmul(overlaps:lt(config.bg_threshold_hi)))
	local fg_per_img = torch.round(config.fg_fraction * config.roi_per_img)
	local cur_num_fg = math.min(fg_per_img, fg_inds:numel())
	local cur_num_bg = math.min(config.roi_per_img - cur_num_fg, bg_inds:numel())




	-- Sampling fgs without replacement
	local selected_fg_inds = torch.LongTensor()
	if cur_num_fg > 0 then
		fg_inds = utils:shuffle(1,fg_inds)
		selected_fg_inds = fg_inds[{{1,cur_num_fg}}]
	end

	-- Sampling bgs without replacement
	local selected_bg_inds = torch.LongTensor()
	if cur_num_bg > 0 then
		bg_inds = utils:shuffle(1,bg_inds)
		selected_bg_inds = bg_inds[{{1,cur_num_bg}}]
	end

	-- Create the sampled batch
	local batch_ids
	if selected_fg_inds:numel()>0 and selected_bg_inds:numel() > 0 then
		batch_ids = selected_fg_inds:cat(selected_bg_inds)
	elseif selected_bg_inds:numel() > 0 then
		batch_ids = selected_bg_inds
	elseif selected_fg_inds:numel() > 0 then
		batch_ids = selected_fg_inds
	else
		print('There is a sample with no positive and negative bounding boxes!')
	end
	local batch_rois = roidb_entry.boxes:index(1,batch_ids)
	local batch_labels = torch.IntTensor(batch_ids:numel(),1):zero()
	batch_labels[{{1,selected_fg_inds:numel()}}]= roidb_entry.label:index(1,selected_fg_inds)
	local batch_bbox_targets = roidb_entry.targets:index(1,batch_ids)

	return batch_ids,batch_rois,batch_labels,batch_bbox_targets
end


function Batcher:_visualize_batch(im,boxes)
  local ok = pcall(require,'qt')
  if not ok then
    error('You need to run visualize_detections using qlua')
  end
  require 'qttorch'
  require 'qtwidget'

  local num_boxes = boxes:size(1)
  local widths  = boxes[{{},3}] - boxes[{{},1}]
  local heights = boxes[{{},4}] - boxes[{{},2}]

  local x,y = im:size(3),im:size(2)
  local w = qtwidget.newwindow(x,y,"Detections")
  local qtimg = qt.QImage.fromTensor(im)
  w:image(0,0,x,y,qtimg)
  local fontsize = 15

  for i=1,num_boxes do
    local x,y = boxes[{i,1}],boxes[{i,2}]
    local width,height = widths[i], heights[i]
    
    -- add bbox
    w:setcolor("red")
    w:rectangle(x,y,width,height)
    w:moveto(x,y+fontsize)
    w:setfont(qt.QFont{serif=true,italic=true,size=fontsize,bold=true})

  end

  w:setcolor("red")
  w:setlinewidth(2)
  w:stroke()
  return w
end