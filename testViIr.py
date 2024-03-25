import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import argparse, time

import options.options as option
from utils import util
from solvers import create_solver
from data import create_dataloader
from data import create_dataset
from PIL import Image


def main():

    parser = argparse.ArgumentParser(description='Test Super Resolution Models')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    #    opt = option.parse(parser.parse_args().opt)
    opt = option.parse('options/test/test_ATFuse.json')
    opt = option.dict_to_nonedict(opt)

    # initial configure
    scale = opt['scale']
    degrad = opt['degradation']
    network_opt = opt['networks']
    model_name = network_opt['which_model'].upper()
    if opt['self_ensemble']: model_name += 'plus'

    # create test dataloader
    bm_names = []
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        test_loaders.append(test_loader)
        print(f'===> Test Dataset: [{test_set.name()}]   Number of images: [{len(test_set)}]')
        bm_names.append(test_set.name())

    # create solver (and load model)
    solver = create_solver(opt)  ### load train and test model
    # Test phase
    print('===> Start Test')
    print("==================================================")
    print(f"Method: {model_name} || Scale: {scale} || Degradation: {degrad}")

    for bm, test_loader in zip(bm_names, test_loaders):
        print(f"Test set : [{bm}]")

        Fu_list = []
        # KV_list = []
        path_list = []

        total_psnr = []
        total_ssim = []
        total_time = []
        Img_cr=[]
        Img_cb=[]
        need_HR = False

        for iter, batch in enumerate(test_loader):
            solver.feed_data(batch)

            # calculate forward time
            t0 = time.time()
            solver.test()
            t1 = time.time()
            total_time.append((t1 - t0))

            visuals = solver.get_current_visual(need_HR=need_HR)
            Fu_list.append(visuals['Fu'])
            # Img_cr.append(visuals['fused_img_cr'])
            # Img_cb.append(visuals['fused_img_cb'])
            # KV =  solver.get_K_V()
            # KV_list.append(KV)
            # calculate PSNR/SSIM metrics on Python
            if need_HR:
                cc, ssim = util.calc_metrics(visuals['Fu'], visuals['HR'], crop_border=scale)
                total_psnr.append(cc)
                total_ssim.append(ssim)
                path_list.append(os.path.basename(batch['HR_path'][0]).replace('HR', model_name))
                print(f"[{iter + 1}/{len(test_loader)}] {os.path.basename(batch['vi_path'][0])} "
                      f"|| CC/SSIM: {cc:.4f}/{ssim:.4f} || Timer: {t1 - t0:.4f} sec .")
            else:
                path_list.append(os.path.basename(batch['vi_path'][0]))
                print(f"[{iter + 1}/{len(test_loader)}] {os.path.basename(batch['vi_path'][0])} "
                      f"|| Timer: {t1 - t0:.4f} sec.")

        if need_HR:
            print(f"---- Average PSNR(dB) /SSIM /Speed(s) for [{bm}] ----")
            print(f"CC: {sum(total_psnr) / len(total_psnr):.4f}      PSNR: {sum(total_ssim) / len(total_ssim):.4f}"
                  f"      Speed: {sum(total_time) / len(total_time):.4f}.")
        else:
            print(f"---- Average Speed(s) for [{bm}] is {sum(total_time) / len(total_time)} sec ----")

        # save SR results for further evaluation on MATLAB
        if need_HR:
            save_img_path = os.path.join('./results/Fu/' + degrad, model_name, bm, "x%d" % scale)
        else:
            save_img_path = os.path.join('./results/Fu/' + bm, model_name, "x%d/" % scale)
        # save_kv_path = os.path.join('./results/KV/' + bm, model_name, "x%d" % scale)
        print(f"===> Saving Fu images of [{bm}]... Save Path: [{save_img_path}]\n")

        # if not os.path.exists(save_kv_path): os.makedirs(save_kv_path)
        if not os.path.exists(save_img_path): os.makedirs(save_img_path)
        index = 1
        for img, name in zip(Fu_list, path_list):
            # img = img.numpy().transpose(1, 2, 0)
            # cv2.imwrite(save_img_path + name, img)
            for i in range(img.shape[0]):
                img = img[i, :, :].numpy()
                # min_v = np.min(img_v)
                # max_v = np.max(img_v)
                # img_v = (img_v - min_v) / (max_v - min_v)

                # img = (img + 1) * 127.5
                # img_v = img_v * 255

                im = Image.fromarray(img)
                im = im.convert("L")
                im.save(save_img_path + name)

                # im.save(save_img_path + str(index)+'.png')
                # index += 1

        # for img, name,kv in zip(Fu_list, path_list,KV_list):
        #     img = img.numpy().transpose(1, 2, 0)
        #     cv2.imwrite(save_img_path+name, img)
        #     solver.save_K_V(kv,save_kv_path,index)
        #     index += 1
        #

    print("==================================================")
    print("===> Finished !")


if __name__ == '__main__':
    main()
