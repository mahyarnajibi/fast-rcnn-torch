local function loadModel(params,backend)

  backend = backend or cudnn

  local features   = nn.Sequential()
  local classifier = nn.Sequential()
  
  features:add(backend.SpatialConvolution(3,96,11,11,4,4,5,5,1))
  features:add(backend.ReLU(true))
  features:add(backend.SpatialMaxPooling(3,3,2,2,1,1))
  --features:add(backend.SpatialCrossMapLRN(5,0.0001,0.75,1))
  features:add(inn.SpatialCrossResponseNormalization(5,0.0001,0.75,1))
  
  features:add(backend.SpatialConvolution(96,256,5,5,1,1,1,1,2))
  features:add(backend.ReLU(true))
  features:add(backend.SpatialMaxPooling(3,3,2,2,1,1))
  --features:add(backend.SpatialCrossMapLRN(5,0.0001,0.75,1))
  features:add(inn.SpatialCrossResponseNormalization(5,0.0001,0.75,1))
  
  features:add(backend.SpatialConvolution(256,384,3,3,1,1,1,1,1))
  features:add(backend.ReLU(true))

  features:add(backend.SpatialConvolution(384,384,3,3,1,1,1,1,2))
  features:add(backend.ReLU(true))
  
  features:add(backend.SpatialConvolution(384,256,3,3,1,1,1,1,2))
  features:add(backend.ReLU(true))
  --features:add(backend.SpatialMaxPooling(3,3,2,2,1,1))
  
  classifier:add(nn.Linear(9216,4096))
  classifier:add(backend.ReLU(true))
  classifier:add(nn.Dropout(0.5))

  classifier:add(nn.Linear(4096,4096))
  classifier:add(backend.ReLU(true))
  classifier:add(nn.Dropout(0.5))
  
  classifier:add(nn.Linear(4096,21))

  local prl = nn.ParallelTable()
  prl:add(features)
  prl:add(nn.Identity())
  
  local ROIPooling = nnf.ROIPooling(6,6):setSpatialScale(1/16)

  local model = nn.Sequential()
  model:add(prl)
  model:add(ROIPooling)
  model:add(nn.View(-1):setNumInputDims(3))
  model:add(classifier)
  
    
  local lparams = model:parameters()
  
  assert(#lparams == #params, 'provided parameters does not match')
  
  for k,v in ipairs(lparams) do
    local p = params[k]
    assert(p:numel() == v:numel(), 'wrong number of parameter elements !')
    v:copy(p)
  end
  
  return model
end

return loadModel
