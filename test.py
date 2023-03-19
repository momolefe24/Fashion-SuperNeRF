from config import *
from Dataset.dataset import ImageDataset
from torch.utils.data import DataLoader
from ESRGAN.model import Generator, Discriminator


def load_without_penalty_checkpoint():
    print("=>Loading checkpoint")
    checkpoint_file = os.path.join(paths_[1], checkpoint_facts[experiment_type]['best_checkpoint_gen'])
    model = Generator().to(device)
    state_dict = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(state_dict)
    return model

# eric input = interpolate (128, 96) * 8 = (1024, 768)
dataset = ImageDataset()
# dataset.__getitem__(0)
loader = DataLoader(dataset, training_facts['batch_size'], shuffle=True, pin_memory=True)

gen = Generator()
disc = Discriminator()
low_res = 32
x = torch.randn((5, 3, low_res, low_res))
gen_out = gen(x)
disc_out = disc(gen_out)

print(gen_out.shape)
print(disc_out.shape)

eric_lr = torch.randn((1,3,100, 100))
eric_out = gen.forward(eric_lr, out_shape=(1024, 768))
print(disc_out.shape)