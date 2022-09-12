from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging


logging.basicConfig(level = logging.INFO)
log = logging.getLogger(__name__)

# Load MNIST dataset from torchvision
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

log.info("Training dataset size: {}".format(len(train_dataset)))
log.info("Test dataset size: {}".format(len(test_dataset)))


