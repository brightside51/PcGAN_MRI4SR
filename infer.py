from networks import Generator, ProjectionDiscriminator
import torch
import dill
import argparse
import os
from tqdm import tqdm
import numpy as np
import options_V1
from skvideo import io

if __name__ == '__main__':

    args = options_V1.parse_arguments()
    checkpoint = torch.load(args.LOAD_CHECKPOINT, pickle_module=dill)
    start_epoch = checkpoint['epoch']

    G = Generator(3, args.IN_CHANNELS, args.OUT_CHANNELS, args.NGF, args.NUM_RESBLOCKS, args.SCHEME, args.SCALE, tanh=False)
    D = ProjectionDiscriminator(3, args.IN_CHANNELS, args.NDF, args.SCHEME, args.SCALE, logits=False)
    G.load_state_dict(checkpoint['G_state_dict']); G.to(args.DEVICE)
    D.load_state_dict(checkpoint['D_state_dict']); D.to(args.DEVICE)

    lr_filepath = "../../MedDiff/evaluation/V1/sampling_0"
    #lr_filepath = "../../MetaBreast/logs/video_diffusion/V2/sample"
    hr_filepath = os.path.join(lr_filepath, "_sr")
    if(not os.path.exists(f"{lr_filepath}_sr")): os.mkdir(f"{lr_filepath}_sr")
    sample_list = os.listdir(lr_filepath); print(len(sample_list))
    print(sample_list)

    for i in range(len(sample_list)):
        sample_img = torch.load(f"{lr_filepath}/{sample_list[i]}", map_location = torch.device(args.DEVICE))
        #sample_img = Image.open(f"{lr_filepath}/{sample_list[i]}")
        #sample_img = [frame.convert('L') for frame in ImageSequence.Iterator(sample_img)]
        #sample_img = sample_img.to(args.DEVICE)
        #sample_img
        #print(sample_img.shape)

        fake_img = G(sample_img).detach().cpu()
        print(fake_img.shape)
        if fake_img.shape[0] > 1:
            for s in range(fake_img.shape[0]):
                io.vwrite(f"{lr_filepath}_sr/sample_{i}_{s}.gif", fake_img[s])
        else: io.vwrite(f"{lr_filepath}_sr/sample_{i}.gif", fake_img[0])


    

