from __future__ import print_function
import os, time, math, torch, numpy as np
import torch.multiprocessing as mp
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call

# --------------------------------- helpers ----------------------------------

def lcm(a, b):
    """最小公倍數 (for print/display freq)"""
    return abs(a * b) // math.gcd(a, b) if a and b else 0

# ---------------------------- pix2pixHD modules -----------------------------
from options.train_options import TrainOptions
from data.data_loader   import CreateDataLoader
from models.models      import create_model
import util.util as util
from util.visualizer    import Visualizer


# =============================================================================
#  main(): 執行完整訓練流程 — 保證只被主行程呼叫，不被 worker 重新跑。
# =============================================================================

def main():
    # 1 ─ 解析參數 & 處理 resume ------------------------------------------------
    opt = TrainOptions().parse()

    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    if opt.continue_train and os.path.isfile(iter_path):
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        print(f"Resuming from epoch {start_epoch} at iteration {epoch_iter}")
    else:
        start_epoch, epoch_iter = 1, 0

    opt.print_freq = lcm(opt.print_freq, opt.batchSize)
    if opt.debug:
        opt.display_freq = opt.print_freq = 1
        opt.niter, opt.niter_decay, opt.max_dataset_size = 1, 0, 10

    # 2 ─ 建 DataLoader --------------------------------------------------------
    data_loader = CreateDataLoader(opt)
    dataset     = data_loader.load_data()
    dataset_size = len(data_loader)
    print(f"# training images = {dataset_size}")

    # 3 ─ 建 Model、混合精度 & DataParallel ------------------------------------
    model = create_model(opt)
    if opt.fp16:
        from apex import amp
        model, [optim_G, optim_D] = amp.initialize(
            model, [model.optimizer_G, model.optimizer_D], opt_level='O1')
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    else:
        optim_G, optim_D = model.module.optimizer_G, model.module.optimizer_D

    visualizer  = Visualizer(opt)
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    display_delta = total_steps % opt.display_freq
    print_delta   = total_steps % opt.print_freq
    save_delta    = total_steps % opt.save_latest_freq

    torch.backends.cudnn.benchmark = True  # fixed 256×256 → 最快 kernel

    # 準備儲存 loss 曲線
    gen_loss_history = []
    disc_loss_history = []

    # 4 ─ 進入 epoch 迴圈 ------------------------------------------------------
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size

        # 累計本 epoch loss
        epoch_gen_loss = 0.0
        epoch_disc_loss = 0.0
        iter_count = 0

        for i, data in enumerate(dataset, start=epoch_iter):
            if total_steps % opt.print_freq == print_delta:
                iter_start_time = time.time()

            total_steps += opt.batchSize
            epoch_iter  += opt.batchSize

            # forward / backward ------------------------------------------------
            losses, generated = model(
                Variable(data['label']), Variable(data['inst']),
                Variable(data['image']), Variable(data['feat']),
                infer=(total_steps % opt.display_freq == display_delta))

            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model.module.loss_names, losses))
            loss_D = 0.5 * (loss_dict['D_fake'] + loss_dict['D_real'])
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)

            # 累計 loss 到 epoch
            epoch_gen_loss += loss_G.item() if isinstance(loss_G, torch.Tensor) else loss_G
            epoch_disc_loss += loss_D.item() if isinstance(loss_D, torch.Tensor) else loss_D
            iter_count += 1

            # optimize ---------------------------------------------------------
            optim_G.zero_grad()
            if opt.fp16:
                from apex import amp
                with amp.scale_loss(loss_G, optim_G) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_G.backward()
            optim_G.step()

            optim_D.zero_grad()
            if opt.fp16:
                with amp.scale_loss(loss_D, optim_D) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_D.backward()
            optim_D.step()

            # console log ------------------------------------------------------
            if total_steps % opt.print_freq == print_delta:
                errs = {k: (v.item() if not isinstance(v, int) else v) for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.print_freq
                visualizer.print_current_errors(epoch, epoch_iter, errs, t)
                visualizer.plot_current_errors(errs, total_steps)

            # HTML display ------------------------------------------------------
            if (total_steps % opt.display_freq == display_delta) and (not opt.no_html):
                vis = OrderedDict([
                    ('input_label',      util.tensor2label(data['label'][0], opt.label_nc)),
                    ('synthesized_image', util.tensor2im(generated.data[0])),
                    ('real_image',       util.tensor2im(data['image'][0]))])
                visualizer.display_current_results(vis, epoch, total_steps)

            # save checkpoint ---------------------------------------------------
            if total_steps % opt.save_latest_freq == save_delta:
                print(f"Saving latest model (epoch {epoch}, total_steps {total_steps})")
                model.module.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter >= dataset_size:
                break  # 一整個 epoch 已跑完

        # Epoch 完成，計算平均並儲存
        avg_G = epoch_gen_loss / iter_count
        avg_D = epoch_disc_loss / iter_count
        gen_loss_history.append(avg_G)
        disc_loss_history.append(avg_D)

        print(f"End of epoch {epoch} / {opt.niter + opt.niter_decay}\tTime: {time.time() - epoch_start_time:.1f} sec"
              f"\tAvg G: {avg_G:.3f}\tAvg D: {avg_D:.3f}")

        if epoch % opt.save_epoch_freq == 0:
            print(f"Saving model at end of epoch {epoch}")
            model.module.save('latest'); model.module.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            model.module.update_fixed_params()
        if epoch > opt.niter:
            model.module.update_learning_rate()

    # ===== 完成所有 epoch 之後，畫出 loss 曲線 =====
    import matplotlib.pyplot as plt

    epochs = list(range(start_epoch, opt.niter + opt.niter_decay + 1))
    plt.figure(figsize=(8,5))
    plt.plot(epochs, gen_loss_history, label='Generator')
    plt.plot(epochs, disc_loss_history, label='Discriminator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GauGAN Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.join(opt.checkpoints_dir, opt.name), exist_ok=True)
    plt.savefig(os.path.join(opt.checkpoints_dir, opt.name, 'loss_curve.png'))
    plt.show()


# =============================================================================
#  Windows 入口點：一定要在 __main__ 裡設 spawn，再呼叫 main()
# =============================================================================
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
