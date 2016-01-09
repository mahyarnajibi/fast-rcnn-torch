require 'nnf'
require 'image'
require 'cudnn'
require 'inn'
require 'nn'

-- load pre-trained Fast-RCNN model
debugger = require('fb.debugger')

params = torch.load('data/models/fast_rcnn_alexnet.t7')
loadModel = dofile 'models/frcnn_alexnet.lua'
model = loadModel(params)



model:evaluate()
model:cuda()

-- prepare detector
image_transformer= nnf.ImageTransformer{mean_pix={102.9801,115.9465,122.7717},
                                        raw_scale = 255,
                                        swap = {3,2,1}}
feat_provider = nnf.FRCNN{image_transformer=image_transformer}
feat_provider:evaluate() -- testing mode
debugger.enter()
detector = nnf.ImageDetect(model, feat_provider)

-- Load an image
I = image.lena()
-- generate some random bounding boxes
torch.manualSeed(500) -- fix seed for reproducibility
bboxes = torch.Tensor(100,4)
bboxes:select(2,1):random(1,I:size(3)/2)
bboxes:select(2,2):random(1,I:size(2)/2)
bboxes:select(2,3):random(I:size(3)/2+1,I:size(3))
bboxes:select(2,4):random(I:size(2)/2+1,I:size(2))

-- detect !
scores, bboxes = detector:detect(I, bboxes)

-- visualization
dofile 'visualize_detections.lua'
threshold = 0.5
-- classes from Pascal used for training the model
cls = {'aeroplane','bicycle','bird','boat','bottle','bus','car',
  'cat','chair','cow','diningtable','dog','horse','motorbike',
  'person','pottedplant','sheep','sofa','train','tvmonitor'}

w = visualize_detections(I,bboxes,scores,threshold,cls)