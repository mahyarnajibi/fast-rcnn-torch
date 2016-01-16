Batcher = torch.class('detection.Batcher')
local inputMaker = detection.InputMaker()
local utils = detection.GeneralUtils()
function Batcher:__init(roidb)
	self._roidb = roidb
	self._n_images = #roidb
	self._cur_image = 1
	self._rand_perm = torch.randperm(self._n_images)

end

function Batcher:_restartBatcher()
	self._cur_image = 1
	self._rand_perm = torch.randperm(self._n_images)
end

function Batcher:getNextBatch()
	if self._cur_image +  config.img_per_batch - 1 > self._n_images then
		self._restartBatcher()
	end  
	local img_inds = self._rand_perm[{{self._cur_image,self._cur_image + config.img_per_batch -1}}]
	self._cur_image = self._cur_image + config.img_per_batch
	local inputs,outputs = self:_process_batch(img_inds)
	return inputs,outputs
	
end

function Batcher:_process_batch(ids)
	local n_images = ids:numel()
	-- Allocating memory for all images
	local imgs = {}
	local roi_batch = torch.FloatTensor()
	local labels = torch.IntTensor()
	local bbox_tagets = torch.FloatTensor()

	-- Process images and create the batch
	for i=1,n_images do

		-- Load Image
		local cur_image = image.load(self._roidb[ids[i]].image_path,3,'float')
		-- Sample roidb
		local cur_ids, cur_rois, cur_labels, cur_bbox_targets = self:_select_rois(self._roidb[ids[i]])
		-- Pre-process image and sampled boxes
		local proc_image,cur_rois = inputMaker:process(cur_image,cur_rois)
		-- Add image ids
		cur_rois[{{},{1}}] = torch.ones(cur_rois:size(1)) * i

		-- Add processed images and rois to the batch
		if i==1 then
			roi_batch = cur_rois
			labels = cur_labels
			bbox_targets = cur_bbox_targets
		else
			roi_batch = roi_batch:cat(cur_rois,1)
			labels = labels:cat(cur_labels,1)
			bbox_targets = bbox_targets:cat(cur_bbox_targets,1)
		end
		imgs[i] = proc_image
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
	debugger.enter()
	local imgs_batch = torch.FloatTensor(n_images,3,max1,max2)
	for i=1,n_images do
		imgs_batch[{{i},{},{1,imgs[i]:size(2)},{1,imgs[i]:size(3)}}] = imgs[i]
	end

	return {imgs_batch,roi_batch}, {labels,bbox_targets}

end

function Batcher:_select_rois(roidb_entry)
	local overlaps = roidb_entry.overlap
	local fg_inds = utils:logical2ind(overlaps:ge(config.fg_threshold))
	local bg_inds = utils:logical2ind(overlaps:ge(config.bg_threshold_lo):cmul(overlaps:lt(config.bg_threshold_hi)))
	local fg_per_img = torch.round(config.fg_fraction * config.roi_per_img)
	local cur_num_fg = math.min(fg_per_img, fg_inds:numel())
	local cur_num_bg = math.min(config.roi_per_img - cur_num_fg, bg_inds:numel())


	-- Sampling fgs without replacement
	local selected_fg_inds = torch.ByteTensor()
	if cur_num_fg > 0 then
		fg_inds = utils:shuffle(1,fg_inds)
		selected_fg_inds = fg_inds[{{1,cur_num_fg}}]
	end


	-- Sampling bgs without replacement
	local selected_bg_inds = torch.ByteTensor()
	if cur_num_bg > 0 then
		bg_inds = utils:shuffle(1,bg_inds)
		selected_bg_inds = bg_inds[{{1,cur_num_bg}}]
	end

	-- Create the sampled batch
	local batch_ids = selected_fg_inds:cat(selected_bg_inds)
	local batch_rois = roidb_entry.boxes:index(1,batch_ids)
	local batch_labels = torch.IntTensor(batch_ids:numel(),1):zero()
	batch_labels[{{1,selected_fg_inds:numel()}}]= roidb_entry.label:index(1,selected_fg_inds)
	local batch_bbox_targets = roidb_entry.targets:index(1,batch_ids)

	return batch_ids,batch_rois,batch_labels,batch_bbox_targets
end