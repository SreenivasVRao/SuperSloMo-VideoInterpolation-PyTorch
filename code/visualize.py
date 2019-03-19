import configparser, cv2, os, glob, logging
from argparse import ArgumentParser
import torch, torch.nn as nn, torch.nn.functional as F
from SuperSloMo.models import SSMR
import numpy as np

log = logging.getLogger(__name__)


def getargs():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True,
                    default="config.ini",
                    help="Path to config.ini file.")
    parser.add_argument("--expt", required=True,
                    help="Experiment Name.")
    parser.add_argument("--log", required=True, help="Path to logfile.")
    parser.add_argument("-d", "--directory", required=True, help = "Directory with input images.")
    parser.add_argument("--fps", required=True, help="Required output fps. Either 60 or 240.")
    args = parser.parse_args()
    return args

class Interpolator:
    
    def __init__(self, config, expt):
        self.cfg = config
        self.model = SSMR.full_model(self.cfg).cuda().eval()
        log_dir = os.path.join(config.get("PROJECT", "DIR"), "logs")
        img_dir = os.path.join(log_dir, expt, "images")
        os.makedirs(img_dir)
        self.img_dir = img_dir
        self.n_frames = self.cfg.getint("TRAIN", "N_FRAMES")

    def load_batch(self, sample):
        """
        Loads a sample batch of B T C H W float RGB tensor. range 0 - 255. B=1
        """
        frame_buffer = []
        for img_path  in sample:
            frame_buffer.append(cv2.imread(img_path)[:,:, ::-1]) # uses RGB format.

        frame_buffer = np.array(frame_buffer)[None, ...] # 1 T H W C
        frame_buffer = torch.from_numpy(frame_buffer).float().cuda()
        frame_buffer = frame_buffer.permute(0, 1, 4, 2, 3) # B T H W C -> B T C H W tensor.

        frame_buffer = F.pad(frame_buffer, [0, 0, 4, 4], mode='constant', value=0)

        return frame_buffer

    def interpolate_frames(self, input_directory, fps240=True, img_type='png'):
        log.info("Looking for %s images in %s. 240 FPS: %s" %(img_type, input_directory, fps240))
        images_list = glob.glob(os.path.join(input_directory, "*."+img_type.lower()))
        images_list.sort()
        
        count = 0
        
        for sample in self.sliding_window(images_list, fps240):
            image_tensor = self.load_batch(sample) # B T C H W
            t1 = image_tensor.shape[1]//2 - 1
            t2 = image_tensor.shape[1]//2
        
            img_0 = image_tensor[0, t1, ...] # corresponds to interpolation at the center.
            img_1 = image_tensor[0, t2, ...] # C H W shape.

            img_0_np = img_0.permute(1, 2, 0).cpu().data.numpy()[..., ::-1] # H W C, 0-255 B G R
            cv2.imwrite(self.img_dir+"/img_"+str(count).zfill(5)+".png", img_0_np.astype(np.uint8))
            count += 1

            image_tensor = self.normalize_tensor(image_tensor)

            for idx in range(1, 8):
                t_interp = [float(idx)/8]*(self.n_frames-1)
                t_interp = torch.Tensor(t_interp).float().cuda()
                t_interp = t_interp.view(1, self.n_frames - 1, 1, 1, 1)

                est_img_t = self.model(image_tensor, dataset_info=None, t_interp=t_interp, compute_loss=False)
                # B C H W tensor.

                est_img_t = self.denormalize_tensor(est_img_t[:, None, ...]) # B T C H W. maintain some backward compatibility.
                est_img_t = est_img_t[0, 0, ...] # C H W
                est_img_t = est_img_t.permute(1, 2, 0).cpu().data.numpy()[..., ::-1] # H W C, 0-255 BGR

                log.info("Interpolated frame: %s"%(count))
                cv2.imwrite(self.img_dir+"/img_"+str(count).zfill(5)+".png", est_img_t.astype(np.uint8))
                count += 1
                
        img_1_np = img_1.permute(1, 2, 0).cpu().data.numpy()[..., ::-1] # H W C, 0 - 255 B G R
        cv2.imwrite(self.img_dir+"/img_"+str(count).zfill(5)+".png", img_1_np.astype(np.uint8))
        count += 1
    
    def normalize_tensor(self, input_tensor):
        pix_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, -1, 1, 1).cuda() # B T C H W
        pix_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, -1, 1, 1).cuda()

        input_tensor = (input_tensor/255.0 - pix_mean)/pix_std

        return input_tensor
        

    def denormalize_tensor(self, output_tensor):
        pix_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, -1, 1, 1).cuda() # B T C H W
        pix_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, -1, 1, 1).cuda()
        output_tensor = ((output_tensor * pix_std) + pix_mean)* 255.0
        return output_tensor
    

    def sliding_window(self, img_paths, fps240=True):
        if fps240:
            # do the sliding window business.
            img_paths = img_paths[::8]
            
        interp_inputs = list(range(len(img_paths)))
        interp_pairs = list(zip(interp_inputs[:-1], interp_inputs[1:]))
        log.info("%s windows." %len(interp_pairs))
        for interp_start, interp_end in interp_pairs:
            left_start = interp_start - ((self.n_frames - 1)//2)
            right_end = interp_end + ((self.n_frames - 1)//2)
            input_locations = list(range(left_start, right_end+1))
            for idx in range(len(input_locations)):
                if input_locations[idx]<0:
                    input_locations[idx]= 0
                elif input_locations[idx]>=len(img_paths):
                    input_locations[idx] = len(img_paths)-1 # final index.
            log.info(input_locations)
            sample = [img_paths[i] for i in input_locations]
            yield sample

if __name__ == "__main__":

     args = getargs()

     config = configparser.RawConfigParser()
     logging.basicConfig(filename=args.log, level=logging.INFO)
     config.read(args.config)

     superslomo = Interpolator(config, args.expt)
     superslomo.interpolate_frames(input_directory=args.directory, fps240=False, img_type='jpg')
     log.info("Interpolation complete.")
