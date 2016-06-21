local InputMaker = torch.class('detection.InputMaker')
local roi = detection.ROI()
function InputMaker:__init()
end
function InputMaker:process(orig_im,bboxes)
	-- This function performs the needed processing on input images
	if type(bboxes) == 'table' then
      bboxes = torch.FloatTensor(bboxes)
      bboxes = bboxes:dim() == 1 and bboxes:view(1,-1) or bboxes
    end

	-- Process the image
	local im = self:_process_image(orig_im)
    local im_size = {im:size(2), im:size(3)}
	-- Scale the im and bboxes
    local im_size_min = math.min(im_size[1],im_size[2])
    local im_size_max = math.max(im_size[1],im_size[2])
    local im_scale = config.scale/im_size_min
    if torch.round(im_scale*im_size_max) > config.max_size then
       im_scale = config.max_size/im_size_max
    end
    local new_size = {torch.round(im_size[1]*im_scale),torch.round(im_size[2]*im_scale)}
    local out_im = image.scale(im,new_size[2],new_size[1],'bicubic')

    -- Project ROIs
    local out_bboxes = roi:projectImageROIs({bboxes},{im_scale})

  	return out_im,out_bboxes,im_scale
end

function InputMaker:_process_image(im)
	local im = im:clone():float()
	-- Correcting dimension
  	if im:dim() == 2 then
	  im = im:view(1,im:size(1) , im:size(2))
	end
	if im:size(1) == 1 then
	  im = im:expand(3,im:size(2),im:size(3))
	end
	-- Scale to 255
	im:mul(255.0)
	-- Swap channels
	local out_im = torch.FloatTensor(im:size())
	local swap_order = {3,2,1}

	for i=1,im:size(1) do
	   out_im[i] = im[swap_order[i]]
	end
	-- Subtracting mean from pixels
	for i=1,3 do
	   out_im[i]:add(-config.pixel_means[i])
	end
	return out_im
end
