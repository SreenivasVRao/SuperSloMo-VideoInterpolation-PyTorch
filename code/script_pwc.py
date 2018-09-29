import sys
import cv2
import torch
import numpy as np
from math import ceil
from torch.autograd import Variable
from scipy.ndimage import imread
import SuperSloMo.models as models


"""
Contact: Deqing Sun (deqings@nvidia.com); Zhile Ren (jrenzhile@gmail.com)
"""


def writeFlowFile(filename, uv):
    """
    According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein
    Contact: dqsun@cs.brown.edu
    Contact: schar@middlebury.edu
    """
    TAG_STRING = np.array(202021.25, dtype=np.float32)
    if uv.shape[2] != 2:
        sys.exit("writeFlowFile: flow must have two bands!");
    H = np.array(uv.shape[0], dtype=np.int32)
    W = np.array(uv.shape[1], dtype=np.int32)
    with open(filename, 'wb') as f:
        f.write(TAG_STRING.tobytes())
        f.write(W.tobytes())
        f.write(H.tobytes())
        f.write(uv.tobytes())


# im1_fn = 'SuperSloMo/data/frame_0010.png';
# im2_fn = 'SuperSloMo/data/frame_0011.png';
# flow_fn = 'SuperSloMo/tmp/test_ref_01.flo';

im1_fn = "Adobe_image0.png"
im2_fn = "Adobe_image1.png"
flow_fn = 'SuperSloMo/tmp/adobe_test.flo'

if len(sys.argv) > 1:
    im1_fn = sys.argv[1]
if len(sys.argv) > 2:
    im2_fn = sys.argv[2]
if len(sys.argv) > 3:
    flow_fn = sys.argv[3]

pwc_model_fn = '../weights/pwc_net.pth.tar';

im_all = [imread(img) for img in [im1_fn, im2_fn]]
im_all = [im[:, :, :3] for im in im_all]

# rescale the image size to be multiples of 64
divisor = 64.
H = im_all[0].shape[0]
W = im_all[0].shape[1]
print(im_all[0].shape)

H_ = int(ceil(H / divisor) * divisor)
W_ = int(ceil(W / divisor) * divisor)
for i in range(len(im_all)):
    im_all[i] = cv2.resize(im_all[i], (W_, H_))

for _i, _inputs in enumerate(im_all):
    im_all[_i] = im_all[_i][:, :, ::-1]
    im_all[_i] = 1.0 * im_all[_i] / 255.0

    im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
    im_all[_i] = torch.from_numpy(im_all[_i])
    im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])
    im_all[_i] = im_all[_i].float()

im_all = torch.autograd.Variable(torch.cat(im_all, 1).cuda(), volatile=True)

net = models.pwc_dc_net(pwc_model_fn)
net = net.cuda()
net.eval()

flo = net(im_all)

# flo = flo[0] * 20.0
# flo = flo.cpu().data.numpy()
#
# # scale the flow back to the input size
# flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2)  #
#
# u_ = cv2.resize(flo[:, :, 0], (W, H))
# v_ = cv2.resize(flo[:, :, 1], (W, H))
# u_ *= W / float(W_)
# v_ *= H / float(H_)
# flo = np.dstack((u_, v_))
#
# print(flo.shape)
# writeFlowFile(flow_fn, flo)

flo = flo*20.0
print(type(flo))
upsampled_flo = torch.nn.functional.upsample(flo, size=(H, W), mode='bilinear')
upsampled_flo[:, 0, ...] = upsampled_flo[:, 0, ...] * W / float(W_)
upsampled_flo[:, 1, ...] = upsampled_flo[:, 1, ...] * H / float(H_)
upsampled_flo = upsampled_flo.permute(0, 2, 3, 1)

flo = upsampled_flo.cpu().data.numpy()
flo = flo[0, ... ]
print(flo.shape)
writeFlowFile(flow_fn, flo)


