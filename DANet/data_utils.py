import torch
from torch.utils.data import Dataset
import h5py

class WSJDataset(Dataset):
    """
    Wrapper for the WSJ Dataset.
    The dataset is saved in HDF5 binary data format,
    which contains the input feature, mixture magnitude
    spectrogram, wiener-filter like mask as training target,
    ideal binary mask as the oracle source assignment,
    and the weight threshold matrix for masking out low 
    energy T-F bins.
    """
    
    def __init__(self, path):
        super(WSJDataset, self).__init__()

        self.h5pyLoader = h5py.File(path, 'r')
        
        self.infeat = self.h5pyLoader['infeat']  # input feature, shape: (num_sample, time, freq)
        self.mixture = self.h5pyLoader['mix']  # mixture magnitude spectrogram, shape: (num_sample, time, freq)
        self.wf = self.h5pyLoader['wf']  # wiener-filter like mask, shape: (num_sample, time*freq, num_spk)
        self.ibm = self.h5pyLoader['ibm']  # ideal binary mask, shape: (num_sample, time*freq, num_spk)
        self.weight = self.h5pyLoader['weight']  # weight threshold matrix, shape: (num_sample, time*freq, 1)
        
        self._len = self.infeat.shape[0]
    
    def __getitem__(self, index):
        """
        Wrap the data to Pytorch tensors.
        """
        infeat_tensor = torch.from_numpy(self.infeat[index])    
        wf_tensor = torch.from_numpy(self.wf[index])
        mixture_tensor = torch.from_numpy(self.mixture[index])
        ibm_tensor = torch.from_numpy(self.ibm[index])
        weight_tensor = torch.from_numpy(self.weight[index])
        return infeat_tensor, wf_tensor, mixture_tensor, mask_tensor, weight_tensor
    
    def __len__(self):
        return self._len
