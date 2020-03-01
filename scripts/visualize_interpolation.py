import configparser
import glob
import logging
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import superslomo_r
from utils import flo_utils

log = logging.getLogger(__name__)


def getargs():
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, default="config.ini", help="Path to config.ini file."
    )
    parser.add_argument("--expt", required=True, help="Experiment Name.")
    parser.add_argument("--log", required=True, help="Path to logfile.")
    parser.add_argument("--input_dir", required=True, help="Directory with input images.")
    parser.add_argument("--img_type", required=True, help="Image type")
    parser.add_argument("--is_fps_240", action="store_true", help="Is input footage 240 fps?")
    parser.add_argument(
        "--upsample_rate",
        type=int,
        required=True,
        help="Integer upsampling rate. For 30FPS -> 240FP, use 8. For 1080FPS, use 36.",
    )
    parser.add_argument(
        "--show_intermediate_outputs",
        action="store_true",
        help="Save occlusion maps, optical flow maps etc.?",
    )
    parser.add_argument("--output_dir", required=True, help="Directory to output.")
    args = parser.parse_args()
    if (args.upsample_rate) <= 1:
        raise Exception("Upsampling rate has to be greater than 1")
    return args


class Interpolator:
    def __init__(self, config, args):
        self.cfg = config
        self.model = superslomo_r.FullModel(self.cfg).cuda().eval()
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        self.img_type = args.img_type
        self.is_fps_240 = args.is_fps_240
        self.upsample_rate = args.upsample_rate
        self.expt = args.expt
        self.show_intermediate_outputs = args.show_intermediate_outputs
        self.setup_directories()
        self.n_frames = self.cfg.getint("TRAIN", "N_FRAMES")

    def load_batch(self, sample):
        """
        Loads a sample batch of B T C H W float RGB tensor. range 0 - 255. B=1
        """
        frame_buffer = []
        for img_path in sample:
            frame_buffer.append(cv2.imread(img_path)[:, :, ::-1])
            # uses RGB format.

        frame_buffer = np.array(frame_buffer)[None, ...]  # 1 T H W C
        frame_buffer = torch.from_numpy(frame_buffer).float().cuda()
        frame_buffer = frame_buffer.permute(0, 1, 4, 2, 3)
        # B T H W C -> B T C H W tensor.

        _, _, _, H, W = frame_buffer.shape
        padding = [0, 0, 0, 0]  # l, r, top, bottom
        if H % 32 != 0:
            h_pad = 32 - (H % 32)
            padding[2] = h_pad // 2
            padding[3] = h_pad - padding[2]

        if W % 32 != 0:
            w_pad = 32 - (W % 32)
            padding[0] = w_pad // 2
            padding[1] = w_pad - padding[0]

        frame_buffer = F.pad(frame_buffer, padding, mode="constant", value=0)
        return frame_buffer

    def setup_directories(self):
        self.img_dir = os.path.join(self.output_dir, self.expt, "images")
        self.est_flow_01_dir = os.path.join(self.output_dir, self.expt, "estimated_flow_01")
        self.est_flow_10_dir = os.path.join(self.output_dir, self.expt, "estimated_flow_10")
        self.refined_flow_t0_dir = os.path.join(self.output_dir, self.expt, "refined_flow_t0")
        self.refined_flow_t1_dir = os.path.join(self.output_dir, self.expt, "refined_flow_t1")
        self.visibility_dir = os.path.join(self.output_dir, self.expt, "visibility_map")
        os.makedirs(self.img_dir, exist_ok=True)
        if self.show_intermediate_outputs:
            os.makedirs(self.est_flow_01_dir, exist_ok=True)
            os.makedirs(self.est_flow_10_dir, exist_ok=True)
            os.makedirs(self.refined_flow_t0_dir, exist_ok=True)
            os.makedirs(self.refined_flow_t1_dir, exist_ok=True)
            os.makedirs(self.visibility_dir, exist_ok=True)

    def interpolate_frames(self):
        log.info(
            "Looking for %s images in %s. 240 FPS: %s"
            % (self.img_type, self.input_dir, self.is_fps_240)
        )
        images_list = glob.glob(os.path.join(self.input_dir, "*." + self.img_type.lower()))
        images_list.sort()

        count = 0
        img_0, img_1 = None, None
        for sample in self.sliding_window(images_list):
            image_tensor = self.load_batch(sample)  # B T C H W
            t1 = image_tensor.shape[1] // 2 - 1
            t2 = image_tensor.shape[1] // 2

            img_0 = image_tensor[:, t1, ...]
            # corresponds to interpolation at the center.
            img_1 = image_tensor[:, t2, ...]
            # B C H W shape.

            self.save_img_from_tensor(
                img_0,
                count,
                self.img_dir,
                prefix="img",
                flo_img=False,
                # write_text=True,
                # text="Original",
            )

            count += 1

            image_tensor = self.normalize_tensor(image_tensor)
            flowC_01, flowC_10 = None, None
            for idx in range(1, self.upsample_rate):
                t_interp = [float(idx) / self.upsample_rate] * (self.n_frames - 1)
                t_interp = torch.Tensor(t_interp).float().cuda()
                t_interp = t_interp.view(1, self.n_frames - 1, 1, 1, 1)

                model_outputs = self.model(image_tensor, t_interp=t_interp, inference_mode=True)
                est_img_t = model_outputs[0]
                (
                    flowC_01,
                    flowC_10,
                    est_flow_t1,
                    est_flow_t0,
                    refined_flow_t1,
                    refined_flow_t0,
                    v_0t,
                ) = model_outputs[1]

                if self.show_intermediate_outputs:
                    self.save_img_from_tensor(
                        v_0t * 255.0, count, self.visibility_dir, prefix="visibility", flo_img=False
                    )

                    # self.save_img_from_tensor(
                    #     est_flow_t1, count, self.est_flow_dir, prefix="flow_t1", flo_img=True
                    # )
                    # self.save_img_from_tensor(
                    #     est_flow_t0, count, self.est_flow_dir, prefix="flow_t0", flo_img=True
                    # )

                    self.save_img_from_tensor(
                        refined_flow_t1,
                        count,
                        self.refined_flow_t1_dir,
                        prefix="flow_t1",
                        flo_img=True,
                    )

                    self.save_img_from_tensor(
                        refined_flow_t0,
                        count,
                        self.refined_flow_t0_dir,
                        prefix="flow_t0",
                        flo_img=True,
                    )

                est_img_t = self.denormalize_tensor(est_img_t[:, None, ...])
                # B 1 C H W. maintain some backward compatibility.
                est_img_t = est_img_t[:, 0, ...]  # B C H W

                log.info("Interpolated frame: %s" % (count))
                self.save_img_from_tensor(
                    est_img_t,
                    count,
                    self.img_dir,
                    prefix="img",
                    flo_img=False,
                    # write_text=True,
                    # text="Interpolated",
                )

                count += 1

            # flow_01, flow_10
            # log.info(flowC_01.shape)
            if self.show_intermediate_outputs:
                self.save_img_from_tensor(
                    flowC_01, count, self.est_flow_01_dir, prefix="Flow_01", flo_img=True
                )

                self.save_img_from_tensor(
                    flowC_10, count, self.est_flow_10_dir, prefix="Flow_10", flo_img=True
                )

        self.save_img_from_tensor(
            img_1,
            count,
            self.img_dir,
            prefix="img",
            flo_img=False,
            # write_text=True,
            # text="Original",
        )
        count += 1

    def save_img_from_tensor(
        self, tensor_img, img_id, out_dir, prefix="", flo_img=False, write_text=False, text=None
    ):

        tensor_img = tensor_img[0, ...]
        img = tensor_img.permute(1, 2, 0).cpu().data.numpy()

        img_name = os.path.join(out_dir, prefix + "_" + str(img_id).zfill(5) + ".png")
        if flo_img:
            img = flo_utils.flow_to_image(img)
        else:
            img = img[..., ::-1]  # RGB  -> BGR
            img = img.astype(np.uint8)
            if write_text:
                img = cv2.UMat(img)
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (1200, 50)
                fontScale = 1.5
                fontColor = (255, 255, 255)
                lineType = 2

                fontColor = (0, 255, 0)
                cv2.putText(
                    img,
                    text + " " + str(img_id).zfill(5),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType,
                )

        cv2.imwrite(img_name, img)

    def normalize_tensor(self, input_tensor):
        pix_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, -1, 1, 1).cuda()  # B T C H W
        pix_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, -1, 1, 1).cuda()
        input_tensor = (input_tensor / 255.0 - pix_mean) / pix_std

        return input_tensor

    def denormalize_tensor(self, output_tensor):
        pix_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, -1, 1, 1).cuda()  # B T C H W
        pix_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, -1, 1, 1).cuda()
        output_tensor = ((output_tensor * pix_std) + pix_mean) * 255.0
        return output_tensor

    def sliding_window(self, img_paths):
        if self.is_fps_240:
            img_paths = img_paths[::8]

        interp_inputs = list(range(len(img_paths)))
        interp_pairs = list(zip(interp_inputs[:-1], interp_inputs[1:]))

        for interp_start, interp_end in interp_pairs:
            left_start = interp_start - ((self.n_frames - 1) // 2)
            right_end = interp_end + ((self.n_frames - 1) // 2)
            input_locations = list(range(left_start, right_end + 1))
            for idx in range(len(input_locations)):
                if input_locations[idx] < 0:
                    input_locations[idx] = 0
                elif input_locations[idx] >= len(img_paths):
                    input_locations[idx] = len(img_paths) - 1  # final index.
            log.info(input_locations)
            sample = [img_paths[i] for i in input_locations]
            yield sample


if __name__ == "__main__":

    args = getargs()

    config = configparser.RawConfigParser()
    logging.basicConfig(filename=args.log, level=logging.INFO)
    config.read(args.config)

    superslomo = Interpolator(config, args)
    superslomo.interpolate_frames()
    log.info("Interpolation complete.")
