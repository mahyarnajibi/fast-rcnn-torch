local BatchProvider, parent = torch.class('nnf.BatchProviderIC','nnf.BatchProviderBase')

local argcheck = require 'argcheck'

local env = require 'argcheck.env' -- retrieve argcheck environement
-- this is the default type function
-- which can be overrided by the user
function env.istype(obj, typename)
  if typename == 'DataSet' then
    return obj and obj._isDataSet
  end
  if typename == 'FeatureProvider' then
    return obj and obj._isFeatureProvider
  end
  return torch.type(obj) == typename
end


local initcheck = argcheck{
  pack=true,
  noordered=true,
  {name="dataset",
   type="DataSet",
   help="A dataset class" 
  },
  {name="feat_provider",
   type="nnf.FRCNN",
   help="A feat provider class" 
  },
  {name="batch_size",
   type="number",
   opt=true,
   help="batch size"},
  {name="imgs_per_batch",
   type="number",
   default=2,
   help="number of images to sample in a batch"},
  {name="fg_fraction",
   type="number",
   default=0.25,
   help="foreground fraction in batch" 
  },
  {name="fg_threshold",
   type="number",
   default=0.5,
   help="foreground threshold" 
  },
  {name="bg_threshold",
   type="table",
   default={0.1,0.5},
   help="background threshold, in the form {LO,HI}" 
  },
  {name="do_flip",
   type="boolean",
   default=true,
   help="sample batches with random flips" 
  },
}

function BatchProvider:__init(...)
  parent.__init(self)

  local opts = initcheck(...)
  for k,v in pairs(opts) do self[k] = v end
end

-- setup is the same

function BatchProvider:permuteIdx()
  local total_img    = self.dataset:size()
  local imgs_per_batch = self.imgs_per_batch

  self._cur = self._cur or math.huge

  if self._cur + imgs_per_batch > total_img  then
    self._perm = torch.randperm(total_img)
    self._cur = 1
  end

  local img_idx = self._perm[{{self._cur,self._cur + self.imgs_per_batch - 1}}]
  self._cur     = self._cur + self.imgs_per_batch

  local img_idx_end  = imgs_per_batch

  local fg_windows = {}
  local bg_windows = {}
  for i=1,img_idx_end do
    local curr_idx = img_idx[i]
    bg_windows[i] = {}
    if self.bboxes[curr_idx][0] then
      for j=1,self.bboxes[curr_idx][0]:size(1) do
        table.insert(bg_windows[i],{curr_idx,j})
      end
    end
    fg_windows[i] = {}
    if self.bboxes[curr_idx][1] then
      for j=1,self.bboxes[curr_idx][1]:size(1) do
        table.insert(fg_windows[i],{curr_idx,j})
      end
    end
  end
  local do_flip = torch.FloatTensor(imgs_per_batch):random(0,1)
  local opts = {img_idx=img_idx,img_idx_end=img_idx_end,do_flip=do_flip}
  return fg_windows,bg_windows,opts

end

function BatchProvider:selectBBoxes(fg_windows,bg_windows)
  local fg_num_each  = torch.round(self.fg_num_each/self.imgs_per_batch)
  local bg_num_each  = torch.round(self.bg_num_each/self.imgs_per_batch)

  local bboxes = {}
  local labels = {}
  for im=1,self.imgs_per_batch do
    local window_idx = torch.randperm(#bg_windows[im])
    local end_idx = math.min(bg_num_each,#bg_windows[im])
    local bbox = {}
    for i=1,end_idx do
      local curr_idx = bg_windows[im][window_idx[i] ][1]
      local position = bg_windows[im][window_idx[i] ][2]
      local dd = self.bboxes[curr_idx][0][position][{{2,6}}]
      table.insert(bbox,{dd[1],dd[2],dd[3],dd[4]})
      table.insert(labels,dd[5])
    end

    window_idx = torch.randperm(#fg_windows[im])
    local end_idx = math.min(fg_num_each,#fg_windows[im])
    for i=1,end_idx do
      local curr_idx = fg_windows[im][window_idx[i] ][1]
      local position = fg_windows[im][window_idx[i] ][2]
      local dd = self.bboxes[curr_idx][1][position][{{2,6}}]
      table.insert(bbox,{dd[1],dd[2],dd[3],dd[4]})
      table.insert(labels,dd[5])
    end
    table.insert(bboxes,torch.FloatTensor(bbox))
  end
  labels = torch.IntTensor(labels)
  return bboxes, labels
end

function BatchProvider:getBatch()
  local dataset = self.dataset
  
  self.fg_num_each = self.fg_fraction * self.batch_size
  self.bg_num_each = self.batch_size - self.fg_num_each
  
  local fg_windows,bg_windows,opts = self:permuteIdx()
  
  self.targets = self.targets or torch.FloatTensor()
  
  local batches = self.batches
  local targets = self.targets
  
  local imgs = {}
  for i=1,opts.img_idx:size(1) do
    table.insert(imgs,dataset:getImage(opts.img_idx[i]))
  end
  local boxes,labels = self:selectBBoxes(fg_windows,bg_windows)
  self.batches = self.feat_provider:getFeature(imgs,boxes,opts.do_flip)

  targets:resize(labels:size()):copy(labels)
  
  return self.batches, self.targets
end
