import configparser, cv2, os, glob, logging
from argparse import ArgumentParser
import torch, torch.nn as nn, torch.nn.functional as F
from SuperSloMo.models import SSMR
from SuperSloMo.utils import flo_utils
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
    parser.add_argument("-d", "--directory", required=True,
                        help = "Directory with input images.")
    # parser.add_argument("--fps", required=True, help="Required output fps. Either 60 or 240.")
    args = parser.parse_args()
    return args


class Interpolator:
    
    def __init__(self, config, expt):
        self.cfg = config
        self.model = SSMR.full_model(self.cfg).cuda().eval()
        log_dir = os.path.join(config.get("PROJECT", "DIR"), "logs")
        self.root_dir = '/mnt/nfs/scratch1/sreenivasv/TDAVI_Visualizations'
        self.expt = expt
        self.setup_directories()
        self.n_frames = self.cfg.getint("TRAIN", "N_FRAMES")

    def load_batch(self, sample):
        """
        Loads a sample batch of B T C H W float RGB tensor. range 0 - 255. B=1
        """
        frame_buffer = []
        for img_path  in sample:
            frame_buffer.append(cv2.imread(img_path)[:,:, ::-1])
            # uses RGB format. 

        frame_buffer = np.array(frame_buffer)[None, ...] # 1 T H W C
        frame_buffer = torch.from_numpy(frame_buffer).float().cuda()
        frame_buffer = frame_buffer.permute(0, 1, 4, 2, 3)
        # B T H W C -> B T C H W tensor.

        frame_buffer = F.pad(frame_buffer, [0, 0, 4, 4], mode='constant',
                             value=0)
        return frame_buffer

    def setup_directories(self):
        self.img_dir = os.path.join(self.root_dir, self.expt, "images")

        self.est_flow_dir = os.path.join(self.root_dir, self.expt,
                                         'estimated_flow')
        self.refined_flow_dir = os.path.join(self.root_dir, self.expt,
                                             'refined_flow')
        self.visibility_dir = os.path.join(self.root_dir, self.expt,
                                           'visibility_map')
        os.makedirs(self.img_dir)
        os.makedirs(self.est_flow_dir)
        os.makedirs(self.refined_flow_dir)
        os.makedirs(self.visibility_dir)


    def interpolate_frames(self, input_directory, fps240=True, img_type='png'):
        log.info("Looking for %s images in %s. 240 FPS: %s" %(img_type,
                                                              input_directory,
                                                              fps240))
        images_list = glob.glob(os.path.join(input_directory,
                                             "*."+img_type.lower()))
        images_list.sort()

        count = 0

        for sample in self.sliding_window(images_list, fps240):
            image_tensor = self.load_batch(sample) # B T C H W
            t1 = image_tensor.shape[1]//2 - 1
            t2 = image_tensor.shape[1]//2
    
            img_0 = image_tensor[:, t1, ...]
            # corresponds to interpolation at the center.
            img_1 = image_tensor[:, t2, ...]
            # B C H W shape.

            self.save_img_from_tensor(img_0, count, self.img_dir, prefix='img',
                                      flo_img=False, write_text=True,
                                      text='Original')

            count += 1

            image_tensor = self.normalize_tensor(image_tensor)

            for idx in range(1, 8):
                t_interp = [float(idx)/8]*(self.n_frames-1)
                t_interp = torch.Tensor(t_interp).float().cuda()
                t_interp = t_interp.view(1, self.n_frames - 1, 1, 1, 1)

                model_outputs = self.model(image_tensor, dataset_info=None,
                                           t_interp=t_interp,
                                           compute_loss=False)
                est_img_t, flowC_01, flowC_10, est_flow_t1, est_flow_t0, refined_flow_t1, refined_flow_t0, v_0t = model_outputs
                
                self.save_img_from_tensor(v_0t*255.0, count, self.visibility_dir,
                                          prefix='visibility', flo_img=False)

                self.save_img_from_tensor(est_flow_t1, count, self.est_flow_dir,
                                          prefix='flow_t1', flo_img=True)

                self.save_img_from_tensor(est_flow_t0, count, self.est_flow_dir,
                                          prefix='flow_t0', flo_img=True)

                self.save_img_from_tensor(refined_flow_t1, count,
                                          self.refined_flow_dir,
                                          prefix='flow_t1', flo_img=True)

                self.save_img_from_tensor(refined_flow_t0, count,
                                          self.refined_flow_dir,
                                          prefix='flow_t0', flo_img=True)
                
                est_img_t = self.denormalize_tensor(est_img_t[:, None, ...])
                # B 1 C H W. maintain some backward compatibility.
                est_img_t = est_img_t[:, 0, ...] # B C H W

                log.info("Interpolated frame: %s"%(count))
                self.save_img_from_tensor(est_img_t, count, self.img_dir,
                                          prefix='img', flo_img=False,
                                          write_text=True,
                                          text='Interpolated')

                count += 1

            # flow_01, flow_10
            # log.info(flowC_01.shape)
            self.save_img_from_tensor(flowC_01, count, self.est_flow_dir,
                                      prefix='Flow_01', flo_img=True)

            self.save_img_from_tensor(flowC_10, count, self.est_flow_dir,
                                      prefix='Flow_10', flo_img=True)


        self.save_img_from_tensor(img_1, count, self.img_dir, prefix='img',
                                  flo_img=False,
                                  write_text=True, text='Original')
        count += 1

    def save_img_from_tensor(self, tensor_img, img_id, out_dir, prefix='',
                             flo_img=False, write_text=False, text=None):
        
        tensor_img = tensor_img[0, ...]
        img = tensor_img.permute(1, 2, 0).cpu().data.numpy()

        img_name = os.path.join(out_dir,
                                prefix+'_'+str(img_id).zfill(5)+'.png')
        if flo_img:
            img = flo_utils.computeImg(img)
        else:
            img = img[..., ::-1] # RGB  -> BGR
            img = img.astype(np.uint8)
            
            if write_text:
                font                   = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (1200, 50)
                fontScale              = 1.5
                fontColor              = (255,255,255)
                lineType               = 2

                fontColor              = (0,255,0)
                cv2.putText(img, text+' '+str(img_id).zfill(5),
                            bottomLeftCornerOfText, font, fontScale, fontColor,
                            lineType)

        cv2.imwrite(img_name, img)

    def normalize_tensor(self, input_tensor):
        pix_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, -1, 1, 1).\
            cuda() # B T C H W
        pix_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, -1, 1, 1).\
            cuda()
        input_tensor = (input_tensor/255.0 - pix_mean)/pix_std

        return input_tensor
    

    def denormalize_tensor(self, output_tensor):
        pix_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, -1, 1, 1).\
            cuda() # B T C H W
        pix_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, -1, 1, 1).\
            cuda()
        output_tensor = ((output_tensor * pix_std) + pix_mean)* 255.0
        return output_tensor
    

    def sliding_window(self, img_paths, fps240=True):
        if fps240:
            # do the sliding window business.
            img_paths = img_paths[::8]
            
        interp_inputs = list(range(len(img_paths)))
        interp_pairs = list(zip(interp_inputs[:-1], interp_inputs[1:]))

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
     superslomo.interpolate_frames(input_directory=args.directory, fps240=False,
                                   img_type='jpg')
     log.info("Interpolation complete.")
