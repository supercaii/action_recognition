import os, pickle, logging, numpy as np
from tqdm import tqdm

from .. import utils as U
from .transformer import pre_normalization


class H4AM_Reader():
    def __init__(self, args, root_folder, transform, h4am_path, **kwargs):
        self.max_channel = 3
        self.max_frame = 1200
        self.max_joint = 32
        self.max_person = 1
        self.select_person_num = 1
        self.dataset = args.dataset
        self.progress_bar = not args.no_progress_bar
        self.transform = transform

        # Set paths
        self.out_path = '{}/original/{}'.format(root_folder, self.dataset)
        U.create_folder(self.out_path)

        # Divide train and eval samples
        training_samples = dict()
        training_samples['H4AM'] = [
            1, 2, 3, 4, 5, 6, 8, 9, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 34, 35, 36, 38
        ]
        training_samples['ntu-xview'] = [2, 3]
        training_samples['ntu-xsub120'] = [
            1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
            38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
            80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
        ]
        training_samples['ntu-xset120'] = set(range(2, 33, 2))
        self.training_sample = training_samples[self.dataset]

        self.file_list = []
        for filename in os.listdir(h4am_path):
            self.file_list.append((h4am_path, filename))

    def read_file(self, file_path, frame_num):
        # M,T,V,C
        skeleton = np.zeros((self.max_person, self.max_frame, self.max_joint, self.max_channel), dtype=np.float32)
        with open(file_path, 'r') as fr:
            for frame in range(frame_num):
                for joint in range(32):
                    joint_info = fr.readline().strip().split()
                    skeleton[0, frame, joint, :] = np.array(joint_info[3:3+self.max_channel], dtype=np.float32)
        return skeleton[:,:frame_num,:,:]


    def gendata(self, phase):
        sample_data = []
        sample_label = []
        sample_path = []
        sample_length = []
        iterizer = tqdm(sorted(self.file_list), dynamic_ncols=True) if self.progress_bar else sorted(self.file_list)
        for folder, filename in iterizer:
            file_path = os.path.join(folder, filename)
            subject_loc = filename.find('P')
            action_loc = filename.find('A')
            frame_loc = filename.find('T')
            subject_id = int(filename[(subject_loc+1):(subject_loc+3)])
            action_class = int(filename[(action_loc+1):(action_loc+3)])
            frame_num = int(filename[(frame_loc+1):(frame_loc+5)])

            # Distinguish train or eval sample
            is_training_sample = (subject_id in self.training_sample)
            if (phase == 'train' and not is_training_sample) or (phase == 'eval' and is_training_sample):
                continue

            # Read one sample
            data = np.zeros((self.max_channel, self.max_frame, self.max_joint, self.select_person_num), dtype=np.float32)
            skeleton = self.read_file(file_path,frame_num)

            # C,T,V,M
            data[:,:frame_num,:,:] = skeleton.transpose(3, 1, 2, 0)

            sample_data.append(data)
            sample_path.append(file_path)
            sample_label.append(action_class - 1)  # to 0-indexed
            sample_length.append(frame_num)

        # Save label
        with open('{}/{}_label.pkl'.format(self.out_path, phase), 'wb') as f:
            pickle.dump((sample_path, list(sample_label), list(sample_length)), f, protocol=4)

        # Transform data
        sample_data = np.array(sample_data)
        if self.transform:
            sample_data = pre_normalization(sample_data, progress_bar=self.progress_bar)

        # Save data
        np.save('{}/{}_data.npy'.format(self.out_path, phase), sample_data)

    def start(self):
        for phase in ['train', 'eval']:
            logging.info('Phase: {}'.format(phase))
            self.gendata(phase)
