"""
Code borrowed from
https://github.com/Johswald/flow-code-python
"""
import os
import sys
import numpy as np

TAG_FLOAT = 202021.25


def read(file):
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file, 'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    # if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    data = np.fromfile(f, np.float32, count=2 * w * h)
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow


def write(filename, uv):
    """
    Borrowed from PWC Net repo
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


if __name__ == '__main__':
    ref_flo_01 = read("/media/sreenivas/Data/UMASS/Thesis/code/SuperSloMo/tmp/pwc_ref_01.flo")
    ref_flo_10 = read("/media/sreenivas/Data/UMASS/Thesis/code/SuperSloMo/tmp/pwc_ref_10.flo")

    custom_flo_01 = read("/media/sreenivas/Data/UMASS/Thesis/code/SuperSloMo/tmp/cv_ref_01.flo")
    custom_flo_10 = read("/media/sreenivas/Data/UMASS/Thesis/code/SuperSloMo/tmp/cv_ref_10.flo")

    print(np.linalg.norm(custom_flo_01 - ref_flo_01))
    print(np.linalg.norm(custom_flo_10 - ref_flo_10))

    custom_flo_01 = read("/media/sreenivas/Data/UMASS/Thesis/code/SuperSloMo/tmp/test_ref_01.flo")
    custom_flo_10 = read("/media/sreenivas/Data/UMASS/Thesis/code/SuperSloMo/tmp/test_ref_10.flo")

    print(np.linalg.norm(custom_flo_01 - ref_flo_01))
    print(np.linalg.norm(custom_flo_10 - ref_flo_10))

