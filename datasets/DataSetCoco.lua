local DataSetCoco,parent = torch.class('detection.DataSetCoco', 'detection.DataSetDetection')
local env = require 'argcheck.env'
local argcheck = dofile'datasets/argcheck.lua'
local json = require 'dkjson'
matio.use_lua_strings = true

local env = require 'argcheck.env' -- retrieve argcheck environement
-- this is the default type function
-- which can be overrided by the user
function env.istype(obj, typename)
  local t = torch.type(obj)
  if t:find('torch.*Tensor') then
    return 'torch.Tensor' == typename
  end
  return torch.type(obj) == typename
end

local initcheck = argcheck{
  pack=true,
  noordered=true,
  help=[[
    Coco dataset for the detection package.
]],
  {name="image_set",
   type="string",
   help="ImageSet name"},
  {name="datadir",
   type="string",
   help="Path to dataset",
   check=function(datadir)
           return paths.dirp(datadir)
         end},
  {name="with_cloud",
   type="boolean",
   help="Whether to load proposals with cloud annotation",
   opt = true},
   {name="year",
   type="string",
   help="MSCOCO year",
   opt = true},
  {name="proposal_root_path",
   type="string",
   help="This class is written to work with the format provided by Hosang et. al.",
   opt = true},
  {name="annotaion_root_path",
   type="string",
   help="Path to the annotations",
   opt = true},
  {name="dataset_name",
   type="string",
   help="Name of the dataset",
   opt = true},
   {name = "proposal_method",
   type = "string",
   help = "Name of the annotation method",
	opt = true},
	{name = "img_path",
   type = "string",
   help = "image path",
	opt = true
	},
	{name = "crowd_threshold",
   type = "number",
   help = "threshold used for prunning proposals overlapping the crowd groun truth boxes",
	opt = true
	},
	{name = "top_k",
   type = "number",
   help = "Number of proposals to be used",
	opt = true
	},
	{ name = "res_save_path",
	type = "string",
	opt = true
	},
	{name = "eval_res_save_path",
	typr = "string",
	opt = true
	},
	{name = "load_from_cache",
	type = "boolean",
	opt = true
	}
}

function DataSetCoco:__init(...)
	 parent.__init(self)

	 local args = initcheck(...)
	 for k,v in pairs(args) do self[k] = v end
	 if not self.image_set then
	 	error 'The image-set should be specified for MSCOCO dataset'
	 end
	 if not self.year then
	 	self.year = 2014
	 end
	 if not self.load_from_cache then
	 	self.load_from_cache = true
	 end
	 if not self.top_k then
	 	self.top_k = 2000 
	 end
	 if not self.crowd_threshold then
	 	self.crowd_threshold = 0.7
	 end
	 if not self.datadir then
	 	self.datadir = config.dataset_path
	 end
	 if not self.res_save_path then
	 	local res_path = paths.concat(self.datadir,'results')
	 	if not paths.dirp(res_path) then
	 		paths.mkdir(res_path)
	 	end
	 	self.res_save_path = paths.concat(res_path,'detections_'..self.image_set..self.year..'.json')
	 end
	 if not self.eval_res_save_path then
	 	local res_path = paths.concat(self.datadir,'results')
	 	if not paths.dirp(res_path) then
	 		paths.mkdir(res_path)
	 	end
	 	self.eval_res_save_path = paths.concat(res_path,'evaluation_results_'..self.image_set..self.year..'.json')
	 end
	 if not self.img_path then
	 	self.img_path = paths.concat(self.datadir,self.image_set..self.year)
	 end
	 if not self.dataset_name then
	 	self.dataset_name = config.dataset
	 end
	 if not self.proposal_method then
	 	self.proposal_method = 'selective_search'
	 end
	 if not self.annotation_root_path then
	 	self.annotation_root_path = paths.concat(self.datadir,'annotations')
	 end

	 if not self.proposal_root_path then
	 	local file_name = 'COCO_'..self.image_set..self.year
	 	if self.image_set == 'test' then
	 		file_name = file_name .. '_'
	 	end
	 	if self.image_set == 'val' then
	 		file_name = file_name .. '_0' 
	 	end
	 	self.proposal_root_path = paths.concat(self.datadir, 'precomputed-coco', self.proposal_method,'mat','COCO_'..self.image_set..self.year)
	 end
	 -- Loading annotation file
	 local annotations_cache_path = paths.concat(config.cache,'coco_'.. self.image_set ..self.year ..'_annotations_cached.t7')
	 if paths.filep(annotations_cache_path) and self.load_from_cache then
	 	print('Reading annotations from cached file...')
	 	local loaded_annotations = torch.load(annotations_cache_path)
	 	for name,value in pairs(loaded_annotations) do
	 		self[name] = value
	 	end
	 	self.num_classes = #self.classes
	 else
	 	-- Load and save only the needed data
	 	print('Loading COCO annotations...')
	 	 self:_loadAnnotations()

	 	 -- Mapping between our class ids and coco class ids
	 	 local n_class = #self.annotations['categories']
		 self.class_mapping_from_coco = tds.Hash()
		 self.class_mapping_to_coco = tds.Hash()
		 self.classes = tds.Hash()
		 for c = 1, n_class do
		 	self.class_mapping_from_coco[self.annotations['categories'][c].id] = c
		 	self.class_mapping_to_coco[c] = self.annotations['categories'][c].id
		 	self.classes[c] = self.annotations['categories'][c].name
		 end
		 self.num_classes = n_class
		 -- Map image ids
	 	self:_mapImageIDs()

	 	-- Loading GTs
	 	self:_prepareGTs()

		 -- Save the computations
		 print('Caching the dataset...')
	 	 local saved_annotations = tds.Hash()
	 	 saved_annotations.class_mapping_from_coco = self.class_mapping_from_coco
	 	 saved_annotations.class_mapping_to_coco = self.class_mapping_to_coco
	 	 saved_annotations.classes = self.classes
	 	 saved_annotations.image_paths = self.image_paths
	 	 saved_annotations.image_ids = self.image_ids
	 	 saved_annotations.image_sizes = self.image_sizes
	 	 saved_annotations.inverted_image_ids = self.inverted_image_ids
	 	 saved_annotations.gts = self.gts
	 	 torch.save(annotations_cache_path,saved_annotations)
	 end

	 -- Loading and pruning the proposals
	 self:loadROIDB()
	 self:_filterCrowd()

end


function DataSetCoco:getImage(i)
  return image.load(self.image_paths[i],3,'float')
end

function DataSetCoco:_mapImageIDs()
	local n_images = #self.annotations['images']
	self.image_ids = torch.IntTensor(n_images)
	self.inverted_image_ids = tds.Hash()
	self.image_paths =  tds.Hash()
	self.image_sizes = torch.IntTensor(n_images,2)
	for i=1,n_images do
		local cur_id = self.annotations['images'][i].id
		self.image_ids[i] = cur_id
		self.inverted_image_ids[cur_id] = i
		self.image_paths[i] = paths.concat(self.img_path,self.annotations['images'][i].file_name)
		self.image_sizes[{{i},{}}] = torch.IntTensor({self.annotations['images'][i].width,self.annotations['images'][i].height})
	end
end

function DataSetCoco:_createSampleSet(nSample_per_class,min_objs_per_img,new_image_set_name)

	local n_images = self:size()
	local n_class = self:nclass()
	local class_counts = torch.IntTensor(n_images,n_class):zero()
	for i=1,n_images do
		local cur_classes = self.gts[i].classes
		for c=1,cur_classes:numel() do
			class_counts[i][cur_classes[c]] = class_counts[i][cur_classes[c]] + 1
		end
	end
	local already_selected = torch.ByteTensor(n_images):zero()
	local new_roidb = tds.Hash()
	local new_gts = tds.Hash()
	local new_paths = tds.Hash()
	local new_image_ids = torch.IntTensor()
	local new_image_sizes = torch.IntTensor()
	local new_inverted_image_ids = tds.Hash()
	local objs_per_img = class_counts:sum(2)
	local img_count = 1
	debugger.enter()
	for c=1,n_class do
		local valid_samples = class_counts[{{},{c}}]:ge(1):cmul(objs_per_img:ge(min_objs_per_img)):cmul(already_selected:eq(0)):eq(1)
		valid_samples = utils:logical2ind(valid_samples)
		-- select a random subset
		valid_samples = utils:shuffle(1,valid_samples)
		local n_samples = math.min(valid_samples:numel(),nSample_per_class)
		for i =1, n_samples do
			already_selected[valid_samples[i]] = 1
			new_gts[img_count] = self.gts[valid_samples[i]]
			new_paths[img_count] = self.image_paths[valid_samples[i]]
			new_inverted_image_ids[self.image_ids[valid_samples[i]]] = img_count
			new_roidb[img_count] = self.roidb[valid_samples[i]]
			if img_count == 1 then
				new_image_sizes = self.image_sizes[valid_samples[i]]:view(1,2)
				new_image_ids = torch.IntTensor({self.image_ids[valid_samples[i]]})
			else
				new_image_ids = new_image_ids:cat(torch.IntTensor({self.image_ids[valid_samples[i]]}))
				new_image_sizes = new_image_sizes:cat(self.image_sizes[valid_samples[i]]:view(1,2),1)
			end

			img_count = img_count + 1
		end
	end

	-- Save the new imageset annotations
	local annotations_cache_path = paths.concat(config.cache,'coco_'.. new_image_set_name ..self.year ..'_annotations_cached.t7')
	local saved_annotations = tds.Hash()
	saved_annotations.class_mapping_from_coco = self.class_mapping_from_coco
	saved_annotations.class_mapping_to_coco = self.class_mapping_to_coco
	saved_annotations.classes = self.classes
	saved_annotations.image_paths = new_paths
	saved_annotations.image_ids = new_image_ids
	saved_annotations.image_sizes = new_image_sizes
	saved_annotations.inverted_image_ids = new_inverted_image_ids
	saved_annotations.gts = new_gts
	torch.save(annotations_cache_path,saved_annotations)

	-- Save the new proposals file
	local cache_name = 'proposals_' .. self.proposal_method ..'_coco' .. self.year .. '_' .. new_image_set_name .. '_top' .. self.top_k ..'.t7'
	local cache_path = paths.concat(config.cache,cache_name)
	torch.save(cache_path,new_roidb)


end

function DataSetCoco:_prepareGTs()
	-- Index annotations by image number
	if self.gts then
		return
	end
	self.gts = tds.Hash()
	for i=1,#self.annotations['images'] do
		self.gts[i] = tds.Hash()
		self.gts[i].bboxes = torch.FloatTensor()
		self.gts[i].areas = torch.FloatTensor()
		self.gts[i].iscrowd = torch.ByteTensor()
		-- Uncomment the following line if you are interested in the segmetation data
		--self.gts[i].segmentation = tds.Hash()
		self.gts[i].classes = torch.ByteTensor()
	end
	-- Look for annotations for each image
	for i=1,#self.annotations['annotations'] do

		local cur_id = self.inverted_image_ids[self.annotations['annotations'][i].image_id]
		if cur_id==56 then
			debugger.enter()
		end
		local cur_box = self.annotations['annotations'][i].bbox
		cur_box = torch.FloatTensor({{cur_box[1],cur_box[2],cur_box[3],cur_box[4]}})
		cur_box = self:_convert_to_x1y1x2y2(cur_box) + 1
		local cur_class = self.class_mapping_from_coco[self.annotations['annotations'][i].category_id]
		if self.gts[cur_id].bboxes:numel() ==0 then
			self.gts[cur_id].bboxes = cur_box
			self.gts[cur_id].areas = torch.FloatTensor({self.annotations['annotations'][i].area})
			self.gts[cur_id].iscrowd = torch.ByteTensor({self.annotations['annotations'][i].iscrowd})
			-- Uncomment the following line if you are interested in the segmentation data
			--self.gts[cur_id].segmentation[1] = self.annotations['annotations'][i].segmentation
			self.gts[cur_id].classes = torch.ByteTensor({cur_class})
		else
			self.gts[cur_id].bboxes = self.gts[cur_id].bboxes:cat(torch.FloatTensor(cur_box),1)
			self.gts[cur_id].areas = self.gts[cur_id].areas:cat(torch.FloatTensor({self.annotations['annotations'][i].area}))
			self.gts[cur_id].iscrowd = self.gts[cur_id].iscrowd:cat(torch.ByteTensor({self.annotations['annotations'][i].iscrowd}))

			-- Uncomment the following line if you are interested in the segmentation data
			--self.gts[cur_id].segmentation[#self.gts[cur_id].segmentation+1] = self.annotations['annotations'][i].segmentation
			self.gts[cur_id].classes = self.gts[cur_id].classes:cat(torch.ByteTensor({cur_class}))
		end
	end

end


function DataSetCoco:getROIBoxes(i)
  if not self.roidb then
    self:loadROIDB()
  end
  return self.roidb[i]--self.roidb[self.img2roidb[self.img_ids[i] ] ]
end

function DataSetCoco:getGTBoxes(i)
	local good_boxes = utils:logical2ind(self.gts[i].iscrowd:eq(0))
	if good_boxes:numel()==0 then
		return torch.FloatTensor(),{}
	end
	local cur_gts = self.gts[i].bboxes:index(1,good_boxes)
	local cur_labels = self.gts[i].classes:index(1,good_boxes)
	-- return a table consisting the labels
	local tabLabels = {}
	for i=1,cur_labels:numel() do
		tabLabels[i] = cur_labels[i]
	end
	return cur_gts,tabLabels
end

function DataSetCoco:size()
  return #self.image_paths
end

function DataSetCoco:_convert_to_x1y1x2y2(boxes)
	return boxes[{{},{1,2}}]:cat(boxes[{{},{1,2}}]+boxes[{{},{3,4}}] - 1) 
end

function DataSetCoco:_convert_to_xywh(boxes)
	return boxes[{{},{1,2}}]:cat(boxes[{{},{3,4}}]-boxes[{{},{1,2}}] + 1)
end


function DataSetCoco:loadROIDB()
	if self.roidb then
		return
	end

	local cache_name = 'proposals_' .. self.proposal_method ..'_coco' .. self.year .. '_' .. self.image_set .. '_top' .. self.top_k ..'.t7'
	local cache_path = paths.concat(config.cache,cache_name)
	if paths.filep(cache_path) then
		print('Loading proposals from a cached file...')
		self.roidb = torch.load(cache_path)
		return
	end

	self.roidb = tds.Hash()

	for i=1,#self:size() do
		if i%1000 ==0 then
			print(string.format('Loaded proposals for %d images!',i))
		end
		-- determine the file name
		local folder_name = tostring(self.image_paths[i]):sub(1,22)
		local file_path = paths.concat(self.proposal_root_path,folder_name,paths.basename(self.image_paths[i],'jpg')..'.mat')
		local proposals = matio.load(file_path)['boxes']
		local n_box = math.min(self.top_k,proposals:size(1))
		self.roidb[i] = proposals[{{1,n_box},{}}]
	end

	-- Cache the loaded proposals
	self:_filterCrowed()
	torch.save(cache_path,self.roidb)
end

function DataSetCoco:_loadAnnotations()
	local annotation_file_path = paths.concat(self.annotation_root_path,'instances_'..self.image_set .. self.year .. '.tds.t7')
	self.annotations = torch.load(annotation_file_path)
end

function DataSetCoco:getImageSize(i)
	return self.image_sizes[i]
end

function DataSetCoco:nclass()
  return #self.classes
end


function DataSetCoco:evaluate(all_detections)
	self:_write_detections(all_detections)
	-- Doing the evaluation using Python COCO API
	local coco_wrapper_path = paths.concat('utils','COCO-python-wrapper','evaluate_coco.py')
	local coco_anno_path = paths.concat(self.annotation_root_path,'instances_'..self.image_set..self.year..'.json')
	local args = string.format(' -i \'%s\' -o \'%s\' -a \'%s\'', self.res_save_path, self.eval_res_save_path,coco_anno_path)
	local cmd = 'python ' .. coco_wrapper_path .. args
	return os.execute(cmd)
end

function DataSetCoco:_test_evaluation()
	local all_detections = tds.Hash()
	for i=1,#self.classes do all_detections[i] = tds.Hash() end
	for i=1, self:size() do
		local gts = self.gts[i].bboxes
		local classes = self.gts[i].classes
		if classes == nil then
			debugger.enter()
		end
		for j=1,#self.classes do
			local cur_inds = utils:logical2ind(classes:eq(j))
			if cur_inds:numel() > 0 then
				all_detections[j][i] = gts:index(1,cur_inds):cat(torch.FloatTensor(cur_inds:numel()):fill(1))
			else
				all_detections[j][i] = torch.FloatTensor()
			end
		end
	end
	self:evaluate(all_detections)
end

function DataSetCoco:_write_detections(all_detections)
	-- Writing detections in the coco format
	print(string.format('Writing detections to: %s \n',self.res_save_path))
	local file = io.open(self.res_save_path,'w')
	file:write('[')
	local n_class = self:nclass()
	local n_images = self:size()
	local first_write = true
	for c = 1,n_class do
		for i=1,n_images do
			local cur_detections = all_detections[c][i]
			if cur_detections:numel() > 0 then
				cur_detections[{{},{1,4}}] = self:_convert_to_xywh(cur_detections[{{},{1,4}}] -1 )
				-- Convert to json and write to the file
				local n_detections = cur_detections:size(1)
				for d=1,n_detections do
					if first_write then
						first_write = false
					else
						file:write(',')
					end

					local cur_det = cur_detections[{{d},{1,4}}][1]
					local cur_score = cur_detections[d][-1]
					local json_entry = {image_id = self.image_ids[i], category_id = self.class_mapping_to_coco[c], bbox = {cur_det[1],cur_det[2],cur_det[3],cur_det[4]}, score = cur_score}
					json_entry = json.encode(json_entry)
					file:write(json_entry)
				end
			end
		end
	end
	file:write(']')
	file:close()
end

function DataSetCoco:getImagePath(i)
	return self.image_paths[i]
end

function DataSetCoco:_filterCrowd()
	for i=1,self:size() do
	  local boxes = self:getROIBoxes(i)
	  -- Filter the bboxes
	  local cur_gts = self.gts[i].bboxes
	  if cur_gts:numel() > 0 then
		  local cur_iscrowd = self.gts[i].iscrowd
		  local bad_gt_ids = utils:logical2ind(cur_iscrowd:eq(1))
		  if bad_gt_ids:numel() == cur_gts:size(1) then
		  	print('There is an image with all crowd GTs!')
		  end
		  if bad_gt_ids:numel()>0 then
		  	local bad_gts = cur_gts:index(1,bad_gt_ids)
		  	local good_bbox_ids = torch.LongTensor()
		  	for j=1,bad_gt_ids:numel() do
		  		local cur_overlaps = utils:boxoverlap(boxes,bad_gts[j])
		  		if good_bbox_ids:numel()==0 then
		  			good_bbox_ids = utils:logical2ind(cur_overlaps:lt(self.crowd_threshold))
		  		else
		  			good_bbox_ids = good_bbox_ids:cat(utils:logical2ind(cur_overlaps:lt(self.crowd_threshold)))
		  		end 
		  	end
		  	if good_bbox_ids:numel()==0 then
		  		print('There is an image with no good bounding boxes!')
		  	else
		  		self.roidb[i] = boxes:index(1,good_bbox_ids)
		  	end
		  end
		end 
	end
end

