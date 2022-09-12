import torch
from utils.ema import EMAModel
from ddpm.components.gaussian_diffusion import (
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    LossType,
    get_named_beta_schedule,
)
from ddpm.components.resample import (
    create_named_schedule_sampler,
    LossAwareSampler,
)
from ddpm.components.unet import UNetModel


class DDPM(torch.nn.Module):
    """Improved DDPM Model"""

    def __init__(
        self,
        unet_args: dict,
        diffusion_steps: int = 1000,
        beta_schedule: str = "cosine",
        model_mean_type: str = "epsilon",
        model_var_type: str = "learned_range",
        loss_type: str = "rescaled_mse",
        schedule_sampler: str = "uniform",
        lr: float = 0.0001,
        weight_decay: float = 0.0,
        ema_start: int = 5000,
        ema_update: int = 100,
        ema_decay: float = 0.995,
        sample_every: int = 10000,
        num_sample_imgs: int = 9,
    ):

        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.beta_schedule = beta_schedule
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.schedule_sampler = schedule_sampler
        self.lr = lr
        self.weight_decay = weight_decay
        self.ema_start = ema_start
        self.ema_update = ema_update
        self.ema_decay = ema_decay
        self.sample_every = sample_every
        self.num_sample_imgs = num_sample_imgs

        self.net = UNetModel(**unet_args)
        self.diffusion = GaussianDiffusion(
            betas=get_named_beta_schedule(beta_schedule, diffusion_steps),
            model_mean_type=ModelMeanType[model_mean_type.upper()],
            model_var_type=ModelVarType[model_var_type.upper()],
            loss_type=LossType[loss_type.upper()],
        )
        self.schedule_sampler = create_named_schedule_sampler(
            schedule_sampler, self.diffusion
        )

        self.ema_model = EMAModel(model=self.net, decay=self.ema_decay)

        self.num_parameters = sum(p.numel() for p in self.net.parameters())
        self.global_step = 0

    def step_ema(self):
        if self.global_step <= self.ema_start:
            self.ema_model.set(self.net)
        else:
            self.ema_model.update(self.net)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def training_step(self, img):
        self.global_step += 1
        t, weights = self.schedule_sampler.sample(img.shape[0], img.device)
        losses = self.diffusion.training_losses(self.net, img, t)

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

        loss = (losses["loss"] * weights).mean()
        return loss
