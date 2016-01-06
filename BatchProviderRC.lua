local BatchProvider,parent = 
                    torch.class('nnf.BatchProviderRC','nnf.BatchProviderBase')


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
   type="FeatureProvider",
   help="A feat provider class"
  },
  {name="batch_size",
   type="number",
   default=128,
   help="batch size"},
  {name="iter_per_batch",
   type="number",
   default=10,
   help=""},
  {name="nTimesMoreData",
   type="number",
   default=10,
   help=""},
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
  {name="target_dim",
   type="number",
   default=1,
   help=""},
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

  self.batch_dim = self.feat_provider.output_size
  
end

function BatchProvider:permuteIdx()
  local fg_num_each  = self.fg_num_each
  local bg_num_each  = self.bg_num_each
  local fg_num_total = self.fg_num_total
  local bg_num_total = self.bg_num_total
  local total_img    = self.dataset:size()
    
  local img_idx      = torch.randperm(total_img)
  local pos_count    = 0
  local neg_count    = 0
  local img_idx_end  = 0
  
  local toadd
  local curr_idx
  while (pos_count <= fg_num_total*self.nTimesMoreData  or
         neg_count <= bg_num_total*self.nTimesMoreData) and 
         img_idx_end < total_img do
    
    img_idx_end = img_idx_end + 1
    curr_idx = img_idx[img_idx_end]

    toadd = self.bboxes[curr_idx][1] and self.bboxes[curr_idx][1]:size(1) or 0
    pos_count = pos_count + toadd
    
    toadd = self.bboxes[curr_idx][0] and self.bboxes[curr_idx][0]:size(1) or 0
    neg_count = neg_count + toadd
    
  end
  
  local fg_windows = {}
  local bg_windows = {}
  for i=1,img_idx_end do
    local curr_idx = img_idx[i]
    if self.bboxes[curr_idx][0] then
      for j=1,self.bboxes[curr_idx][0]:size(1) do
        table.insert(bg_windows,{curr_idx,j})
      end
    end
    if self.bboxes[curr_idx][1] then
      for j=1,self.bboxes[curr_idx][1]:size(1) do
        table.insert(fg_windows,{curr_idx,j})
      end
    end
  end
  
  local opts = {img_idx=img_idx,img_idx_end=img_idx_end}
  return fg_windows,bg_windows,opts
end


function BatchProvider:selectBBoxes(fg_windows,bg_windows)
  local fg_w = {}
  local bg_w = {}

  local window_idx = #bg_windows>0 and torch.randperm(#bg_windows) or torch.Tensor()
  for i=1,self.bg_num_total do
    local curr_idx = bg_windows[window_idx[i] ][1]
    local position = bg_windows[window_idx[i] ][2]
    if not bg_w[curr_idx] then
      bg_w[curr_idx] = {}
    end
    local dd = self.bboxes[curr_idx][0][position]
    table.insert(bg_w[curr_idx],dd)
  end
  
  window_idx = #fg_windows>0 and torch.randperm(#fg_windows) or torch.Tensor()
  for i=1,self.fg_num_total do
    local curr_idx = fg_windows[window_idx[i] ][1]
    local position = fg_windows[window_idx[i] ][2]
    if not fg_w[curr_idx] then
      fg_w[curr_idx] = {}
    end
    local dd = self.bboxes[curr_idx][1][position]
    table.insert(fg_w[curr_idx],dd)
  end
  
  return fg_w,bg_w
end

-- depends on the model
function BatchProvider:prepareFeatures(im_idx,bboxes,fg_label,bg_label)

  local num_pos = bboxes[1] and #bboxes[1] or 0
  local num_neg = bboxes[0] and #bboxes[0] or 0

  fg_label:resize(num_pos,self.target_dim)
  bg_label:resize(num_neg,self.target_dim)
  
  local flip = false
  if self.do_flip then
    flip = torch.random(0,1) == 0
  end

  local s_boxes = {}
  for i=1,num_pos do
    local bbox = {bboxes[1][i][2],bboxes[1][i][3],bboxes[1][i][4],bboxes[1][i][5]}
    table.insert(s_boxes,bbox)
    fg_label[i][1] = bboxes[1][i][6]
  end
  
  for i=1,num_neg do
    local bbox = {bboxes[0][i][2],bboxes[0][i][3],bboxes[0][i][4],bboxes[0][i][5]}
    table.insert(s_boxes,bbox)
    bg_label[i][1] = bboxes[0][i][6]
  end

  -- compute the features
  local feats = self.feat_provider:getFeature(im_idx,s_boxes,flip)
  local fg_data = num_pos > 0 and feats:narrow(1,1,num_pos) or nil
  local bg_data = num_neg > 0 and feats:narrow(1,num_pos+1,num_neg) or nil

  return fg_data, bg_data
end

function BatchProvider:prepareBatch(batches,targets)
  local dataset = self.dataset
  
  self.fg_num_each = self.fg_fraction * self.batch_size
  self.bg_num_each = self.batch_size - self.fg_num_each
  self.fg_num_total = self.fg_num_each * self.iter_per_batch
  self.bg_num_total = self.bg_num_each * self.iter_per_batch
  
  local fg_windows,bg_windows,opts = self:permuteIdx()
  local fg_w,bg_w = self:selectBBoxes(fg_windows,bg_windows)
  
  local batches = batches or torch.FloatTensor()
  local targets = targets or torch.IntTensor()
  
  batches:resize(self.iter_per_batch,self.batch_size,unpack(self.batch_dim))
  targets:resize(self.iter_per_batch,self.batch_size,self.target_dim)
  
  local fg_rnd_idx = self.fg_num_total>0 and torch.randperm(self.fg_num_total) or torch.Tensor()
  local bg_rnd_idx = self.bg_num_total>0 and torch.randperm(self.bg_num_total) or torch.Tensor()
  local fg_counter = 0
  local bg_counter = 0
  
  local fg_data,bg_data,fg_label,bg_label
  fg_label = torch.IntTensor()
  bg_label = torch.IntTensor()

  local pass_index = torch.type(self.feat_provider) == 'nnf.SPP' and true or false

  print('==> Preparing Batch Data')
  for i=1,opts.img_idx_end do
    xlua.progress(i,opts.img_idx_end)

    local curr_idx = opts.img_idx[i]
    
    local nfg = fg_w[curr_idx] and #fg_w[curr_idx] or 0
    local nbg = bg_w[curr_idx] and #bg_w[curr_idx] or 0
    
    nfg = type(nfg)=='number' and nfg or nfg[1]
    nbg = type(nbg)=='number' and nbg or nbg[1]
    
    local bboxes = {}
    bboxes[0] = bg_w[curr_idx]
    bboxes[1] = fg_w[curr_idx]
  
    local data
    if pass_index then
      data = curr_idx
    else
      data = dataset:getImage(curr_idx)
    end
    fg_data,bg_data = self:prepareFeatures(data,bboxes,fg_label,bg_label)
    
    for j=1,nbg do
      bg_counter = bg_counter + 1
      local idx = bg_rnd_idx[bg_counter]
      local b = math.ceil(idx/self.bg_num_each)
      local s = (idx-1)%self.bg_num_each + 1
      batches[b][s]:copy(bg_data[j])
      targets[b][s]:copy(bg_label[j])
    end

    for j=1,nfg do
      fg_counter = fg_counter + 1
      local idx = fg_rnd_idx[fg_counter]
      local b = math.ceil(idx/self.fg_num_each)
      local s = (idx-1)%self.fg_num_each + 1 + self.bg_num_each 
      batches[b][s]:copy(fg_data[j])
      targets[b][s]:copy(fg_label[j])
    end
    collectgarbage()
  end
  collectgarbage()
  return batches,targets
end

function BatchProvider:getBatch()
  self._cur = self._cur or math.huge
  -- we have reached the end of our batch pool, need to recompute
  if self._cur > self.iter_per_batch then
    self._batches,self._targets = self:prepareBatch(self._batches,self._targets)
    self._cur = 1
  end

  self.batches = self._batches[self._cur]
  self.targets = self._targets[self._cur]
  self._cur = self._cur + 1

  return self.batches, self.targets

end
