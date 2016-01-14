InputMaker = torch.class('detection.InputMaker')

function InputMaker:__init()
end
function InputMaker:process(im,bboxes)
	-- This function performs the needed processing on input images
	if type(bboxes) == 'table' then
      bboxes = torch.FloatTensor(bboxes)
      bboxes = bboxes:dim() == 1 and bboxes:view(1,-1) or bboxes
    end
    -- Get image size


	-- Process the image
	im = self:_process_image(im)
    local im_size = {im:size(2), im:size(3)}
	-- Scale the im and bboxes
    local im_size_min = math.min(im_size[1],im_size[2])
    local im_size_max = math.max(im_size[1],im_size[2])
    local im_scale = config.scale/im_size_min
    if torch.round(im_scale*im_size_max) > config.max_size then
       im_scale = config.max_size/im_size_max
    end
    local new_size = {torch.round(im_size[1]*im_scale),torch.round(im_size[2]*im_scale)}
    im = image.scale(im,new_size[2],new_size[1])

    -- Project ROIs
    bboxes = ROI:projectImageROIs({bboxes},{im_scale})

  	return im,bboxes,im_scale
end

function InputMaker:_process_image(im)

	local out_im = im:float()
	-- Correcting dimension
  	if out_im:dim() == 2 then
	  out_im = out_im:view(1,out_im:size(1),out_im:size(2))
	end
	if out_im:size(1) == 1 then
	  out_im = out_im:expand(3,out_im:size(2),out_im:size(3))
	end


	-- Swap channels
	swap_order = {3,1,2}

	for i=1,im:size(1) do
	   out_im[i] = out_im[swap_order[i]]
	end

	-- Scale to 255
	out_im:mul(255)


	-- Subtracting mean from pixels
	for i=1,3 do
	   out_im[i]:add(-config.pixel_means[i])
	end

	return out_im
end
