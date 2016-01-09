-- This is a demo file for performing detection give some proposals

matio = require 'matio'
-- Loading selective search
require 'nnf'
require 'image'
require 'cudnn'
require 'inn'
require 'nn'

debugger =require 'fb.debugger'


image_path = './demo/test.jpg'
selective_search_path = './demo/proposals.mat'

model = torch.load('data/models/fast_rcnn_alexnet.t7')
loadModel = dofile 'models/frcnn_alexnet.lua'
model = loadModel(model:getParameters())

-- Loading proposals

---NOTE that the code assumes that the minimum bounding box position is at 1
-- Also the order is like as the original selective search output 
-- In other words the code assumes that it is reading the original selective search mat file!

proposals = matio.load(selective_search_path)['boxes']
proposals = proposals:add(1)
-- Changing the orders to [x1,y1,x2,y2]
-- proposals = proposals:index(2,torch.LongTensor{2,1,4,3})


-- Loading processed image for debug!
--im = matio.load('./demo/im.mat')['im']

-- Loading image

im = image.load(image_path)

model:evaluate()
model:cuda()

-- prepare detector
image_transformer= nnf.ImageTransformer{mean_pix={102.9801,115.9465,122.7717},
                                        raw_scale = 255,
                                        swap = {3,2,1}}


feat_provider = nnf.FRCNN{image_transformer=image_transformer}
feat_provider:evaluate() -- testing mode

detector = nnf.ImageDetect(model, feat_provider)


-- detecting

scores, bboxes = detector:detect(im, proposals)
debugger.enter()