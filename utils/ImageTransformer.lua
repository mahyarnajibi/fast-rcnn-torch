local ImageTransformer = torch.class('detection.ImageTransformer')

function ImageTransformer:__init(...)
  self.mean_pix = {128/255,128/255,128/255}
  self.raw_scale = 1
  self.swap = nil 
  local args = ...
  for k,v in pairs(args) do self[k] = v end
end

local function channel_swap(I,swap)
  if not swap then
    return I:clone()
  end
  local out = I.new():resizeAs(I)
  for i=1,I:size(1) do
    out[i] = I[swap[i] ]
  end
  return out
end

function ImageTransformer:preprocess(I)
  local I = I
  if I:dim() == 2 then
    I = I:view(1,I:size(1),I:size(2))
  end
  if I:size(1) == 1 then
    I = I:expand(3,I:size(2),I:size(3))
  end
  I = channel_swap(I,self.swap)
  if self.raw_scale ~= 1 then
    I:mul(self.raw_scale)
  end
  for i=1,3 do
    I[i]:add(-self.mean_pix[i])
  end
  return I
end

function ImageTransformer:__tostring()
  local str = torch.type(self)
  if self.swap then
    str = str .. '\n  Channel swap: [' .. table.concat(self.swap,', ') .. ']'
  end
  str = str .. '\n  Raw scale: '.. self.raw_scale
  str = str .. '\n  Mean pixel: [' .. table.concat(self.mean_pix,', ') .. ']'
  return str
end
