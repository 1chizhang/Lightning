import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim

from compressai.losses import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer

from lightning.pytorch import LightningModule
from lit_config import Config
from torch_ema import ExponentialMovingAverage
from typing import Any, Dict
from pytorch_msssim import ms_ssim
import copy
from timm.utils import unwrap_model

class CompressorLit(LightningModule):
    """Lightning-based compression model
    """

    def __init__(self, net, net_lr, lmbda, beta=None, gradient_clip_norm=1.0, comment=None,compile=False,ema=False):
        super().__init__()
        self.net = net
        self.compiled = compile
        if self.compiled:
            self.net = torch.compile(self.net)
        self.net_lr = net_lr
        self.gradient_clip_norm = gradient_clip_norm
        self.comment = comment
        self.lmbda = lmbda

        self.criterion = RateDistortionLoss(lmbda=lmbda, beta=beta)
        self.automatic_optimization = False

        self.ema = ema
        if self.ema:
            self.ema_net = ExponentialMovingAverage(self.net.parameters(), decay=0.999)
        self.save_configuration()

    def save_configuration(self):
        self.hparams["a-Model"] = dict()
        self.hparams["b-Trainer"] = dict()
        self.hparams["c-Data"] = dict()

        self.hparams["a-Model"]["a-comment"] = self.comment
        for k, v in dict(vars(Config.Model)).items():
            if not k.startswith("_"):
                self.hparams["a-Model"][k] = v
        for k, v in dict(vars(Config.Trainer)).items():
            if not k.startswith("_"):
                self.hparams["b-Trainer"][k] = v
        for k, v in dict(vars(Config.Data)).items():
            if not k.startswith("_"):
                self.hparams["c-Data"][k] = v

        self.save_hyperparameters(ignore=['net', 'net_lr', 'lmbda', 'beta', 'gradient_clip_norm', 'comment'])

    def on_train_start(self):
        with open(f'{self.trainer.log_dir}/model.txt', mode='w') as f:
            f.write(str(self.net))
        device = self.device
        if self.ema:
            self.ema_net.to(device)

    def configure_optimizers(self):
        conf = {
            "net": {"type": "Adam", "lr": self.net_lr},
        }

        optimizers = net_aux_optimizer(self.net, conf)
        
        print("Estimated stepping batches: ", self.trainer.estimated_stepping_batches)
        warmup_step_size = int(0.01 * self.trainer.estimated_stepping_batches)
        scheduler_step_size = int(0.9 * self.trainer.estimated_stepping_batches)
        print("Scheduler warmup step size: ", warmup_step_size)
        print("Scheduler step size: ", scheduler_step_size)

        #use SequentialLR scheduler consisting of two LR schedulers
        #define the first LR scheduler that the first warmup_step_size steps, the learning rate will be linearly increased from 0 to self.net_lr
        #define the second LR scheduler that the learning rate will be decreased by a factor of 0.1 after scheduler_step_size steps
        scheduler1 =   optim.lr_scheduler.LinearLR(optimizers["net"], start_factor=0.01, end_factor=1.0,total_iters  = warmup_step_size)
        scheduler2 = optim.lr_scheduler.StepLR(optimizers["net"], step_size=scheduler_step_size, gamma=0.1)

        scheduler_list = [scheduler1, scheduler2]

        lr_scheduler = optim.lr_scheduler.SequentialLR(
            optimizers["net"],
            schedulers=scheduler_list,
            milestones = [warmup_step_size],
            

        )


        # lr_scheduler = optim.lr_scheduler.StepLR(optimizers["net"], step_size=scheduler_step_size, gamma=0.1)

        

        return ({"optimizer": optimizers["net"], "lr_scheduler": lr_scheduler})

    def training_step(self, batch, batch_idx):
        if batch_idx%1800==0:
            self.print("Updating quantiles")
            self.net.entropy_bottleneck._update_quantiles()


        net_opt= self.optimizers()
        lr_scheduler = self.lr_schedulers()

        net_opt.zero_grad()

        out_net = self.net(batch)

        out_criterion = self.criterion(out_net, batch)
        self.manual_backward(out_criterion["loss"])
        if self.gradient_clip_norm > 0.0:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.gradient_clip_norm)
        net_opt.step()
        if self.ema:
            self.ema_net.update(self.net.parameters())

        lr_scheduler.step()

        log_info = {
            "train/loss": out_criterion["loss"].item(),
            "train/mse": out_criterion["mse_loss"].item() * 255 ** 2 / 3,
            "train/bpp": out_criterion["bpp_loss"].item(),
            "train/psnr": -10 * math.log10(out_criterion["mse_loss"].item())
        }
        self.log_dict(log_info, sync_dist=True if len(Config.Trainer.devices) > 1 else False)
    def compress_decompress(self,net, x):
        model_t= copy.deepcopy(net)
        model_t.eval()
        model_t.update(force=True)
        out_enc = model_t.compress(x)
        out_dec = model_t.decompress(out_enc['strings'], out_enc['shape'])
        num_pixels = x.size(0) * x.size(2) * x.size(3)
        #calculate the mse
        MSE = torch.nn.functional.mse_loss(out_dec['x_hat'], x)
        bpp_ = 0
        for s in out_enc["strings"]:
            for j in s:
                if isinstance(j, list):
                    for i in j:
                        if isinstance(i, list):
                            for k in i:
                                bpp_ += len(k)
                        else:
                            bpp_ += len(i)
                else:
                    bpp_ += len(j)
        bpp_ *= 8.0 / num_pixels
        return {"mse_loss": MSE, "bpp_loss": bpp_}

    def validation_step(self, batch, batch_idx):
        # out_net = self.net(batch)
        # out_criterion = self.criterion(out_net, batch)
        out_criterion = self.compress_decompress(self.net,batch)
        loss = self.lmbda *255**2 *out_criterion["mse_loss"] + out_criterion["bpp_loss"]

        # EMA
        if self.ema:
            self.ema_net.store(self.net.parameters())
            self.ema_net.copy_to(self.net.parameters())
            out_criterion_ema = self.compress_decompress(self.net,batch)
            loss_ema = self.lmbda *255**2 *out_criterion_ema["mse_loss"] + out_criterion_ema["bpp_loss"]
            self.ema_net.restore(self.net.parameters())

        if self.ema:
            log_info = {
                "valid/loss": loss,
                "valid/mse": out_criterion["mse_loss"] * 255 ** 2 / 3,
                "valid/bpp": out_criterion["bpp_loss"],
                "valid/psnr": -10 * math.log10(out_criterion["mse_loss"]),
                "valid/loss_ema": loss_ema,
                "valid/mse_ema": out_criterion_ema["mse_loss"] * 255 ** 2 / 3,
                "valid/bpp_ema": out_criterion_ema["bpp_loss"],
                "valid/psnr_ema": -10 * math.log10(out_criterion_ema["mse_loss"])
            }
        else:
            log_info = {
                "valid/loss": loss,
                "valid/mse": out_criterion["mse_loss"] * 255 ** 2 / 3,
                "valid/bpp": out_criterion["bpp_loss"],
                "valid/psnr": -10 * math.log10(out_criterion["mse_loss"])
            }
        self.log_dict(log_info, sync_dist=True if len(Config.Trainer.devices) > 1 else False)

    def on_save_checkpoint(self, checkpoint):
        updated_state_dict = unwrap_model(self.net).state_dict()
        checkpoint['state_dict'] = updated_state_dict
        if self.ema:
            ema_state_dict = self.ema_net.state_dict()
            checkpoint['ema_state_dict'] = ema_state_dict

    def on_load_checkpoint(self, checkpoint):
        updated_state_dict = OrderedDict()
        for key in checkpoint['state_dict'].keys():
            if not self.compiled:
                updated_state_dict[f'net.{key}'] = checkpoint['state_dict'][key]  # add "net." for lightning compatibility
            else:
                updated_state_dict[f'net._orig_mod.{key}'] = checkpoint['state_dict'][key]
        checkpoint['state_dict'] = updated_state_dict
        if self.ema:
            self.ema_net.load_state_dict(checkpoint['ema_state_dict'])
