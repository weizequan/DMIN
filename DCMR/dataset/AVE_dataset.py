import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class AVEDataset(Dataset):
    def __init__(self, data_root, split='train'):
        super(AVEDataset, self).__init__()
        self.split = split
        self.visual_feature_path = os.path.join(data_root, 'visual_feature_vec.h5')
        self.audio_feature_path = os.path.join(data_root, 'audio_feature.h5')
        # Now for the supervised task
        self.labels_path = os.path.join(data_root, 'labels_closs.h5') 
        self.sample_order_path = os.path.join(data_root, f'{split}_order_match.h5')
        self.h5_isOpen = False

    def __getitem__(self, index):
        if not self.h5_isOpen:
            self.visual_feature = h5py.File(self.visual_feature_path, 'r')['avadataset']
            self.audio_feature = h5py.File(self.audio_feature_path, 'r')['avadataset']
            self.labels = h5py.File(self.labels_path, 'r')['avadataset']
            self.sample_order = h5py.File(self.sample_order_path, 'r')['order']
            self.l = h5py.File(self.sample_order_path, 'r')['order'][:]
            self.h5_isOpen = True
        sample_index = self.sample_order[index]
        visual_feat = self.visual_feature[sample_index]
        audio_feat = self.audio_feature[sample_index]
        label = self.labels[sample_index]
        
        '''
        x_audio = np.zeros((len(self.l) * 10, 128)) #[333910,128]
        x_video = np.zeros((len(self.l) * 10, 512)) #[333910,512]
        y = np.zeros((len(self.l) * 10)) #[33390]
        
        for i in range(len(l)):
            id = l[i]
            for j in range(10):
                x_audio[10 * i + j, :] = self.audio_features[id, j, :]
                x_video[10 * i + j, :] = self.video_features[id, j, :]
                y[10 * i + j] = self.labels[id, j]
        '''

        return visual_feat, audio_feat, label


    def __len__(self):
        f = h5py.File(self.sample_order_path, 'r')
        sample_num = len(f['order'])
        f.close()
        return sample_num

