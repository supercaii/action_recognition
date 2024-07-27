'''
Description: 
Author: jackieysong
Email: jackieysong@tencent.com
Date: 2021-11-15 10:50:26
Acknowledgements: 
'''
import os, argparse
os.chdir(os.getcwd())


parser = argparse.ArgumentParser(description='Modify Configs')
parser.add_argument('--root_folder', '-r', type=str, default='', help='Your path to save numpy dataset')
parser.add_argument('--ntu60_path', '-n60p', type=str, default='', help='Your path to save original NTU 60 dataset (S001 to S017)')
parser.add_argument('--ntu120_path', '-n120p', type=str, default='', help='Your path to save original NTU 120 dataset (S018 to S032)')
parser.add_argument('--pretrained_path', '-pp', type=str, default='', help='Your path to save pretrained models')
parser.add_argument('--work_dir', '-wd', type=str, default='', help='Your path to save checkpoints and log files')
parser.add_argument('--gpus', '-g', type=str, default='', help='Your gpu device numbers')
args, _ = parser.parse_known_args()

for file in os.listdir('./configs'):
    with open(f'./configs/{file}', 'r') as f:
        lines = f.readlines()

    fr = open(f'./configs/{file}', 'w')
    for line in lines:
        if 'root_folder:' in line and args.root_folder != '':
            new_line = f'    root_folder: {args.root_folder}\n'
        elif 'ntu60_path:' in line and args.ntu60_path != '':
            new_line = f'    ntu60_path: {args.ntu60_path}\n'
        elif 'ntu120_path:' in line and args.ntu120_path != '':
            new_line = f'    ntu120_path: {args.ntu120_path}\n'
        elif 'pretrained_path:' in line and args.pretrained_path != '':
            new_line = f'pretrained_path: {args.pretrained_path}\n'
        elif 'work_dir:' in line and args.work_dir != '':
            new_line = f'work_dir: {args.work_dir}\n'
        elif 'gpus:' in line and args.work_dir != '':
            new_line = f'gpus: {args.gpus}\n'
        else:
            new_line = line
        fr.write(new_line)
    fr.close()
