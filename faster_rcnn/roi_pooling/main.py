import torch
import torch.nn as nn
from torch.autograd import Variable
from modules.roi_pool import RoIPool

spatial_scale = 0.0625
group_size = 7
output_dim = 392


indata = torch.randn(1,3,224,224)
indata = indata.cuda()
indata = Variable(indata)


rois = torch.IntTensor([[1,120,100,220,200],[1,230,240,430,440]])
rois = rois.cuda()
inrois = Variable(rois)

psroi_pooling = RoIPool(7,7,1/16)
print psroi_pooling
raw_input()
psroi_pooling.forward(indata,inrois)
