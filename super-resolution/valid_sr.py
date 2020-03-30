import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import SR_SliceDataDev
from models import TeacherVDSR,StudentVDSR
import h5py
from tqdm import tqdm

def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.
    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir.mkdir(exist_ok=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)


def create_data_loaders(args):

    
    data = SR_SliceDataDev(args.data_path)
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,)

    return data_loader


def load_model(model_type,checkpoint_file):

    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']

    if model_type == 'teacher':
        model = TeacherVDSR().to(args.device)
    else:
        model = StudentVDSR().to(args.device)

    model.load_state_dict(checkpoint['model'])

    return model


def run_model(args, model,data_loader):

    model.eval()
    reconstructions = defaultdict(list)

    with torch.no_grad():
        for (iter,data) in enumerate(tqdm(data_loader)):

            input,target,fnames,slices= data 
            input = input.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)
    
            input = input.float()
            target = target.float()
    
            recons = model(input)
            #print(recons)
            #recons = model(input.float())[-1]

            recons = recons[-1].to('cpu').squeeze(1)
            #print (recons.shape)

            #if args.dataset_type == 'cardiac':
            #    recons = recons[:,5:155,5:155]
            for i in range(recons.shape[0]):
                recons[i] = recons[i] 
                reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))

    #print (reconstructions.items())
    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }


    return reconstructions


def main(args):
    
    data_loader = create_data_loaders(args)
    model = load_model(args.model_type,args.checkpoint)

    reconstructions = run_model(args, model, data_loader)
    #print (reconstructions.keys(),reconstructions['0.h5'].shape)
    save_reconstructions(reconstructions, args.out_dir)


def create_arg_parser():

    parser = argparse.ArgumentParser(description="Valid setup for MR recon U-Net")
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    parser.add_argument('--data-path',type=str,help='path to validation dataset')
    parser.add_argument('--model_type',type=str,help='model type teacher student')
    	
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
