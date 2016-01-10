-- All parameters goes here
config = config or {}

function config.parse(arg)
	local cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Fast R-CNN for Torch')
	cmd:text()
	cmd:text('')
	-- Parameters
	cmd:option('-scale', {600}, 'Scales used for training and testing, currently single scale!')
	cmd:option('-max_size', 1000, 'Max pixel size of the longest side of a scaled input image')
	cmd:option('-ims_per_batch', 2, 'Images to use per minibatch')
	cmd:option('-batch_size', 128, 'Minibatch size')
	cmd:option('-fg_fraction', 0.25, 'Fraction of minibatch that is labeled foreground (i.e. class > 0)')
	cmd:option('-fg_threshold', 0.5, 'Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)')
	cmd:option('-bg_threshold_hi', 0.5, 'High overlap threshold for a ROI to be considered background')
	cmd:option('-bg_threshold_lo',0.1, 'Low overlap threshold for a ROI to be considered background')
	cmd:option('-use_flipped', true, 'Use horizontally-flipped images during training?')
	cmd:option('-bbox_threshold', 0.5, 'Overlap required between a ROI and ground-truth box in order for that ROI to be used as a bounding-box regression training example')
	cmd:option('-snapshot_iters', 10000, 'Iterations between snapshots')
	cmd:option('-nms', 0.3, 'Overlap threshold used for non-maximum suppression (suppress boxes with IoU >= this val)')
	cmd:option('-pixel_means', {102.9801,115.9465,122.7717}, 'Pixel mean values (BGR order)')
	cmd:option('-eps', 1e-14, 'Epsilon')
	cmd:option('-log_path','./cache/','Path used for saving log data')

	-- Parsing the command line 
	config = cmd:parse(arg or {})

	return config
end

return config
