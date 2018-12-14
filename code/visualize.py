import ConfigParser, cv2, os, glob, logging
from argparse import ArgumentParser
import torch
from SuperSloMo.models import SSM

log = logging.getLogger(__name__)


def getargs():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True,
                    default="config.ini",
                    help="Path to config.ini file.")
    parser.add_argument("--expt", required=True,
                    help="Experiment Name.")
    parser.add_argument("--log", required=True, help="Path to logfile.")
    parser.add_argument("-i", "--images", required=True, help = "Directory with input images.")
    args = parser.parse_args()
    return args


def get_image(path, flipFlag):
    img = cv2.imread(path)
    img = img/255.0
    if flipFlag:
        img = img.swapaxes(0, 1)
    img = torch.from_numpy(img)
    img = img.cuda().float()
    img = img[None, ...]
    img = img.permute(0, 3, 1, 2) # bhwc => bchw
    pad = torch.nn.ZeroPad2d([0,0, 8, 8])
    img = pad(img)
    return img


args = getargs()

config = ConfigParser.RawConfigParser()
logging.basicConfig(filename=args.log, level=logging.INFO)
config.read(args.config)

log_dir = os.path.join(config.get("PROJECT", "DIR"), "logs")
img_dir = os.path.join(log_dir, args.expt, "images")
os.makedirs(img_dir)

superslomo = SSM.full_model(config).cuda().eval()
# get the superslomo network.

fpath = args.images
            
images_list = glob.glob(os.path.join(fpath, "*.png"))
images_list.sort()

log.info("Input video clip has %s frames"%(len(images_list)/8))

img_0 = cv2.imread(images_list[0])
h, w, c = img_0.shape
vFlag = h > w  # horizontal video => h<=w vertical video => w< h
info = (736, 1280), (1.0, 1.0)

log.info("Interpolation beginning. Original length: %s" % len(images_list))
count = 0
start_idx = 0
window = 25
overlap = 17
end_idx = start_idx + window
iteration = 0

while end_idx <= len(images_list):
    iteration +=1

    current_images = images_list[start_idx:end_idx] #[I_0 - I_3]
    current_images = current_images[0::8] #[I_0, I_1, I_2, I_3]

    image_tensor = [get_image(impath, vFlag) for impath in current_images]
    image_tensor = torch.stack(image_tensor, dim=1)

    img_0 = image_tensor[:, 1, ...] #[I0, I1, I2, I3]
    img_1 = image_tensor[:, 2, ...] #[I0, I1, I2, I3]

    img_0_np = img_0 * 255.0
    img_0_np = img_0_np.permute(0,2, 3, 1)[0, ...]
    img_0_np = img_0_np.cpu().data.numpy()
    cv2.imwrite(img_dir+"/img_"+str(count).zfill(5)+".png", img_0_np)
    count += 1

    for idx in range(1, 8):
        t_interp = float(idx)/8
        interpolation_result = superslomo(image_tensor, info, t_interp, split="VAL", iteration=iteration,
                                          compute_loss=False)
        est_image_t = interpolation_result * 255.0
        est_image_t = est_image_t.permute(0,2, 3, 1)[0, ...]
        est_image_t = est_image_t.cpu().data.numpy()
        cv2.imwrite(img_dir+"/img_"+str(count).zfill(5)+".png", est_image_t)
        count += 1

    start_idx = start_idx + window - overlap
    end_idx = start_idx + window

img_1_np = img_1 * 255.0
img_1_np = img_1_np.permute(0,2, 3, 1)[0, ...]
img_1_np = img_1_np.cpu().data.numpy()
cv2.imwrite(img_dir+"/img_"+str(count).zfill(5)+".png", img_1_np)
count += 1

log.info("Interpolation complete.")
