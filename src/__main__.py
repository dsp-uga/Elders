import argparse
from src import CNNPreProc, unet

def main(args):
    parser = argparse.ArgumentParser(description='Execute Elder commands')
    parser.add_argument('-model', '--model', default='unet')

    args = parser.parse_args()
    if args.model == 'unet':
        unet().extract_roi()
    if args.model == 'cnn':
        cn=CNNPreProc()
