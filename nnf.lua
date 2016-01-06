require 'nn'
require 'image'
--require 'inn'
require 'xlua'

nnf = {}

torch.include('nnf','ImageTransformer.lua')


torch.include('nnf','BatchProviderBase.lua')
torch.include('nnf','BatchProviderIC.lua')
torch.include('nnf','BatchProviderRC.lua')

torch.include('nnf','FRCNN.lua')
torch.include('nnf','ROIPooling.lua')


torch.include('nnf','ImageDetect.lua')
--return nnf
