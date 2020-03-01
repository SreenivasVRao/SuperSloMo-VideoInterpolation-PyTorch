import torch
import torch.nn
import numpy as np
import cv2
import numbers

cv2.setNumThreads(0)


class Binarize(object):
    """
    Flips the images horizontally 50% of the time.
    Performs a random rotation of the data.
    """
    def __call__(self, buffers):
        """
        :param frames: N H W C array.
        :return: same array after augmentation.
        """
        img_buffer, gt_buffer = buffers

        N, H, W, C = gt_buffer.shape

        new_buffer = np.zeros([N, H, W, 1])

        # rotate around random center.

        for idx in range(N):
            gt = gt_buffer[idx, ...]
            gt = gt.astype(np.uint8)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
            ret, gt = cv2.threshold(gt, 1, 255, cv2.THRESH_BINARY)
            new_buffer[idx, ..., 0] = gt
        new_buffer = new_buffer / 255.0

        return [img_buffer, new_buffer]


class RandomMirrorRotate(object):
    """
    Flips the images horizontally 50% of the time.
    Performs a random rotation of the data.
    """
    def __call__(self, frames):
        """
        :param frames: N H W C array.
        :return: same array after augmentation.
        """
        if np.random.randint(0, 2) == 1:
            frames = frames[:, :, ::-1, :]  # horizontal flip 50% of the time

        N, H, W, C = frames.shape

        cx = np.random.randint(0, W)
        cy = np.random.randint(0, H)
        theta = np.random.uniform(-5, 5)
        M = cv2.getRotationMatrix2D((cx, cy), theta, 1)
        # rotate around random center.

        for idx in range(N):
            img = frames[idx, ...]
            frames[idx, ...] = cv2.warpAffine(img, M, (W, H))

        return frames


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs):
        h, w, _ = inputs[0].shape
        th, tw = self.size
        if w == tw and h == th:
            return inputs

        x1 = np.random.randint(0, w - tw)
        y1 = np.random.randint(0, h - th)

        new_frames = np.zeros((inputs.shape[0], th, tw, 3))

        for i in range(inputs.shape[0]):
            new_frames[i, ...] = inputs[i, ...][y1:y1 + th, x1:x1 + tw]

        return new_frames


class ResizeCrop(object):
    """
    Convert 720 x 1280 frames to 352 x 352 -> Resize + Random Cropping
    """
    def __init__(self, crop_imh, crop_imw, resize_ratio=0.5):
        self.crop_imh = crop_imh
        self.crop_imw = crop_imw
        self.resize_ratio = resize_ratio

    def __call__(self, sample_frames):
        _, h, w, c = sample_frames.shape
        # assert h==720 and w==1280, "invalid dimensions"

        new_imh = int(h * self.resize_ratio)
        new_imw = int(w * self.resize_ratio)

        # deal with too small images
        if new_imh < self.crop_imh or new_imw < self.crop_imw:
            rh = self.crop_imh / new_imh
            rw = self.crop_imw / new_imw
            if rh > rw:
                new_imw = int(rh * new_imw)
                new_imh = self.crop_imh
            else:
                new_imh = int(rw * new_imh)
                new_imw = self.crop_imw
        new_frames = np.zeros((sample_frames.shape[0], new_imh, new_imw, 3))

        if new_imh < self.crop_imh or new_imw < self.crop_imw:
            print('orig_imh: {}, orig_imw: {}'.format(h, w))
            print('new_imh: {}, new_imw: {}'.format(new_imh, new_imw))
            print('self.crop_imh: {}, self.crop_imw: {}'.format(
                self.crop_imh, self.crop_imw))
            raise RuntimeError('Input images are too small.')

        for idx in range(sample_frames.shape[0]):
            new_frames[idx, ...] = cv2.resize(sample_frames[idx, ...],
                                              (new_imw, new_imh))
        h_start = np.random.randint(0, new_imh - self.crop_imh + 1)
        w_start = np.random.randint(0, new_imw - self.crop_imw + 1)
        new_frames = new_frames[:, h_start:h_start + self.crop_imh,
                                w_start:w_start + self.crop_imw, ...]

        return new_frames


class EvalPad(object):
    """
    Zero padding for evaluation alone. 720 x 1280 -> 736x1280
    """
    def __init__(self, padding, target_dims=None):
        self.pad = padding
        self.target_dims = target_dims

    def get_padding(self, h_in, w_in):
        """
        SlowFlow dataset has different image sizes. I'm making all of them equal to 1024x1280.
        """

        h_out, w_out = self.target_dims

        h_pad = h_out - h_in
        w_pad = w_out - w_in

        top = h_pad // 2
        bottom = h_pad - top

        left = w_pad // 2
        right = w_pad - left

        return torch.nn.ZeroPad2d([left, right, top, bottom])

    def __call__(self, sample_tensor):
        _, c, h, w = sample_tensor.shape

        # assert h == 720 and w == 1280, "invalid dimensions"

        if self.target_dims is not None:
            new_pad = self.get_padding(h, w)
            sample_tensor = new_pad(sample_tensor)
        else:
            sample_tensor = self.pad(sample_tensor)

        return sample_tensor


class Normalize(object):
    def __init__(self, pix_mean, pix_std, divisor=255.0):
        self.pix_mean = pix_mean
        self.pix_std = pix_std
        self.divisor = divisor

    def __call__(self, sample_tensor):
        sample_tensor = (sample_tensor / self.divisor -
                         self.pix_mean) / self.pix_std
        return sample_tensor


class ToTensor(object):
    """
    Converts np 0-255 uint8 to 0-1 tensor
    """
    def __call__(self, sample):
        sample = torch.from_numpy(sample.copy())
        sample = sample.permute(0, 3, 1, 2)  # n_frames, H W C -> n_frames, C, H, W
        return sample


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value
