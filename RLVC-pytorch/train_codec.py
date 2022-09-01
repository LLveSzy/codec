import torch
import random
from tqdm import tqdm
from utils.util import *
from models import CodecNet
from utils.visualize import Visual
from torch.utils.data import DataLoader
from dataset.vimeo_dataset import VimeoDataset, VimeoGroupDataset


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    gpu_id = 3
    batch_size = 8
    lr = 1e-5
    epochs = 20
    device = torch.device('cuda:{}'.format(gpu_id))
    v = Visual('exp')
    dataset = VimeoGroupDataset('/data/szy/datasets/vimeo_septuplet/sequences/')
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    n_train = len(dataset)//20
    sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:n_train])
    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)

    global_step = 1
    codec_pretrain = f'./checkpoints/codec_union.pth'
    codec_net = CodecNet('./parameters.cfg', device, 'ALL')
    codec_net = get_model(codec_net, device, codec_pretrain)
    try:
        for epoch in range(1, epochs):
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', ncols=112) as pbar:
                for frames in train_dataloader:
                    loss = 0
                    frames = frames.to(device)
                    last = frames[:, 0, ...]
                    h1_state = h2_state = h3_state = h4_state = None
                    for i in range(1, frames.shape[1]):
                        cur = last = frames[:, i, ...]
                        codec_net(last, cur)
                        codec_net.count_loss()
                        codec_net.optimizer_step()

                    # v.draw_pics(global_step, [frames[:, -2, ...], residual, re_res, retrieval_frames])
                    # v.draw_pics(global_step, [frames[:, -2, ...], current])
                    pbar.set_postfix(**{'loss (batch)': format(codec_net.loss.item(), '.3f')})
                    pbar.update(frames.shape[0])
                    global_step += 1

        save_checkpoint(codec_net, 'codec_union')
    except KeyboardInterrupt:
        save_checkpoint(codec_net, 'codec_union')