import argparse
from src import CNNPreProc, unet
from src import nmf
from src.utils import download_dataset
import train_frcnn_kitti as rcn

def main(args):
    parser = argparse.ArgumentParser(description='Execute Elder commands')
    parser.add_argument('-model', '--model', default='unet')
    
    # Download the dataset 
    dd=download_dataset()
    dd.download_codeneuro()    

    args = parser.parse_args()
    if args.model == 'unet':
        unet().extract_roi()
    if args.model == 'cnn':
        cn=CNNPreProc()
		# Remember to download the frcn code from https://github.com/Houchaoqun/keras_frcnn
		rcn.train_kitti()
    if args.model == 'nmf':
        from src import nmf
        
