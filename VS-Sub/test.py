from dataset import *
from einops import rearrange
import argparse
from torchvision.transforms import ToTensor
from model_patch_shuffler import SubCostNet
import time
import cv2
from scipy.ndimage import zoom

# Settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes", type=int, default=9, help="angular resolution")
    parser.add_argument('--fn', type=int, default=16, help="feature nymber in feature extraction process")
    parser.add_argument('--interval', type=float, default=0.25, help="disparity label interval")
    parser.add_argument('--disp_max', type=int, default=4.0, help="number of residual layer in recon block")
    parser.add_argument('--disp_min', type=int, default=-4.0, help="number of residual layer in recon block")
    parser.add_argument('--model_name', type=str, default='FastSubCost1')
    parser.add_argument('--testset_dir', type=str, default='')
    parser.add_argument('--crop', type=bool, default=True)
    parser.add_argument('--patchsize', type=int, default=128)
    parser.add_argument('--minibatch_test', type=int, default=2)
    parser.add_argument('--model_path', type=str, default='./log/SubpixelCost.pth.tar')
    parser.add_argument('--save_path', type=str, default='./Results/')

    return parser.parse_args()


'''
Note: 1) We crop LFs into overlapping patches to save the CUDA memory during inference. 
      2) Since we have not optimize our cropping scheme, when cropping is performed, 
         the inference time will be longer than the one reported in our paper.
'''


def test(cfg):
    net = SubCostNet(cfg)
    net.to(cfg.device)
    model = torch.load(cfg.model_path, map_location='cuda:0')
    net.load_state_dict(model['state_dict'])

    scene_list = os.listdir(cfg.testset_dir)

    angRes = cfg.angRes
    mse_iter_test = []
    bad_iter_test = []
    bad_003_test = []
    bad_001_test = []


    for scenes in scene_list:
        dispGT = np.float32(
            read_pfm(cfg.testset_dir + scenes + '/gt_disp_lowres.pfm'))
        print('Working on scene: ' + scenes + '...')
        temp = imageio.imread(cfg.testset_dir + scenes + '/input_Cam000.png')
        temp = cv2.resize(temp, (512, 512), interpolation=cv2.INTER_LINEAR)
        lf = np.zeros(shape=(9, 9, temp.shape[0], temp.shape[1], 3), dtype=int)
        for i in range(81):
            temp = imageio.imread(cfg.testset_dir + scenes + '/input_Cam0%.2d.png' % i)
            temp = cv2.resize(temp, (512, 512), interpolation=cv2.INTER_LINEAR)
            lf[i // 9, i - 9 * (i // 9), :, :, :] = temp
        lf_gray = np.mean((1 / 255) * lf.astype('float32'), axis=-1, keepdims=False)
        angBegin = (9 - angRes) // 2
        lf_angCrop = lf_gray[angBegin:  angBegin + angRes, angBegin: angBegin + angRes, :, :]

        if cfg.crop == False:
            data = rearrange(lf_angCrop, 'u v h w -> (u h) (v w)')
            data = ToTensor()(data.copy())
            data = data.unsqueeze(0)
            with torch.no_grad():
                disp = net(data.to(cfg.device))
            disp = np.float32(disp[0, 0, :, :].data.cpu())

        else:
            patchsize = cfg.patchsize
            stride = patchsize // 2
            data = torch.from_numpy(lf_angCrop)
            sub_lfs = LFdivide(data.unsqueeze(2), patchsize, stride)
            n1, n2, u, v, c, h, w = sub_lfs.shape
            sub_lfs = rearrange(sub_lfs, 'n1 n2 u v c h w -> (n1 n2) u v c h w')
            mini_batch = cfg.minibatch_test
            num_inference = (n1 * n2) // mini_batch

            out_disp = []
            time_start = time.time()
            for idx_inference in range(num_inference):
                with torch.no_grad():
                    current_lfs = sub_lfs[idx_inference * mini_batch: (idx_inference + 1) * mini_batch, :, :, :, :, :]
                    input_data = rearrange(current_lfs, 'b u v c h w -> b c (u h) (v w)')
                    temp, prob, uncer= net(input_data.to(cfg.device))
                    out_disp.append(temp)

            if (n1 * n2) % mini_batch:
                with torch.no_grad():
                    current_lfs = sub_lfs[(idx_inference + 1) * mini_batch:, :, :, :, :, :]
                    input_data = rearrange(current_lfs, 'b u v c h w -> b c (u h) (v w)')
                    temp, _, _ = net(input_data.to(cfg.device))
                    out_disp.append(temp)
            out_disps = torch.cat(out_disp, dim=0)
            out_disps = rearrange(out_disps, '(n1 n2) c h w -> n1 n2 c h w', n1=n1, n2=n2)
            disp = LFintegrate(out_disps, patchsize, patchsize // 2)
            disp = disp[0: data.shape[2], 0: data.shape[3]]
            disp = np.float32(disp.data.cpu())
            print('Finished! \n')
            write_pfm(disp, cfg.save_path + '%s.pfm' % (scenes))
            mse100 = np.mean((disp - dispGT) ** 2) * 100
            mse_iter_test.append(mse100)
            txtfile = open('MSE100.txt', 'a')
            txtfile.write('mse_{}={:3f}\t'.format(scenes, mse100))
            txtfile.close()
            bad_iter = cal_bad_pixel(disp, dispGT, 0.07)
            bad_iter_test.append(bad_iter)
            txtfile = open('BAD007.txt', 'a')
            txtfile.write('bad_{}={:3f}\t'.format(scenes, bad_iter))
            bad_003 = cal_bad_pixel(disp, dispGT, 0.03)
            bad_003_test.append(bad_003)
            txtfile = open('BAD003.txt', 'a')
            txtfile.write('bad_{}={:3f}\t'.format(scenes, bad_003))
            bad_001 = cal_bad_pixel(disp, dispGT, 0.01)
            bad_001_test.append(bad_001)
            txtfile = open( 'BAD001.txt', 'a')
            txtfile.write('bad_{}={:3f}\t'.format(scenes, bad_001))
            disp = (disp - disp.min()) / (disp.max() - disp.min()) * 255
            cv2.imwrite('v_{}.png'.format(scenes), disp)
    mse_epoch = float(np.array(mse_iter_test).mean())
    bad_epoch = float(np.array(bad_iter_test).mean())
    bad_003_epoch = float(np.array(bad_003_test).mean())
    bad_001_epoch = float(np.array(bad_001_test).mean())
    txtfile = open('MSE100.txt', 'a')
    txtfile.write('mse_average={:3f}\t'.format(mse_epoch))
    txtfile.write('\n')
    txtfile = open('BAD007.txt', 'a')
    txtfile.write('bad_average={:3f}\t'.format(bad_epoch))
    txtfile.write('\n')
    txtfile = open('BAD003.txt', 'a')
    txtfile.write('bad_average={:3f}\t'.format(bad_003_epoch))
    txtfile.write('\n')
    txtfile = open('BAD001.txt', 'a')
    txtfile.write('bad_average={:3f}\t'.format(bad_001_epoch))
    txtfile.write('\n')
    txtfile.close()
    

    return

def cal_bad_pixel(img1, img2, thre):
    diff = np.abs(img1 - img2)
    bp = (diff >= thre)
    bad_pixel_ratio = 100 * np.average(bp)
    return bad_pixel_ratio


if __name__ == '__main__':
    cfg = parse_args()
    test(cfg)
