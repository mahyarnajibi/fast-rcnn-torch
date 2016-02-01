# Torch implementation of the Fast R-CNN
This is a torch implementation of the Fast R-CNN proposed by Girshick et .al. [1].
## Requirements
* You need to install torch.
* The current implementation uses these torch packages: ```nn, inn, cudnn, image, matio, optim, paths ```
* You need a machine with CUDA GPU.
*  You need to download the required weights and proposal files as discussed below.

## Running the demo
For running the demo you only need to download the weights of the Fast R-CNN network. Please run ``` ./scripts/get_frcnn_models.sh``` to get the trained Fast R-CNN models. After downloading the weights file you can run the demo in the terminal as follows:
```lua
qlua demo.lua
```
After running this file you should see the following detections:

![alt text](data/demo/demo_detections.png "Detections with AlexNet")
## Training the Fast-RCNN network

