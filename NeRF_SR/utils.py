import torch


def save_checkpoint(state,filename='my_checkpoint.pth.tar'):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(model,optimizer,checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])