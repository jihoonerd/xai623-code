from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from ddpm.ddpm_module import DDPM
import logging
import yaml
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import math

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Load config
with open("configs/ddpm.yaml", "r") as f:
    config = yaml.safe_load(f)

# Tensorboard for logging & Visualization
writer = SummaryWriter()

device = torch.device(
    f"cuda:{config['gpu_id']}" if torch.cuda.is_available() else "cpu"
)

# Load MNIST dataset from torchvision
train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)

train_loader = DataLoader(
    dataset=train_dataset, batch_size=config["batch_size"], shuffle=True
)

log.info("Training dataset size: {}".format(len(train_dataset)))

# Load model
ddpm_model = DDPM(**config["ddpm"]).to(device)
optimizer = ddpm_model.configure_optimizers()
log.info("Number of parameters: {}".format(ddpm_model.num_parameters))

pbar = tqdm(range(config["training_epochs"]))
for epoch in pbar:
    mb_pbar = tqdm(train_loader, leave=False)
    for batch_idx, (img, label) in enumerate(mb_pbar):
        img = img.to(device)
        img = img * 2 - 1  # Move image to [-1, 1]

        optimizer.zero_grad()
        loss = ddpm_model.training_step(img)
        pbar.set_description(f"Epoch: {epoch} | Loss: {loss:.5f}")
        loss.backward()
        optimizer.step()
        ddpm_model.step_ema()

    if epoch % ddpm_model.sample_every == 0:
        pbar.set_description(f"Epoch: {epoch} | [Visualizing]")
        res = ddpm_model.net.image_size
        ema_model = ddpm_model.ema_model.model.eval()
        sampled_img = ddpm_model.diffusion.p_sample_loop(
            model=ddpm_model.ema_model.model,
            shape=(ddpm_model.num_sample_imgs, 1, res, res),
        )
        sampled_img = (sampled_img + 1) * 0.5  # Unnormalize
        grid_img = make_grid(
            sampled_img, nrow=int(math.sqrt(ddpm_model.num_sample_imgs))
        )
        writer.add_image(f"x_0", grid_img, global_step=ddpm_model.global_step)
