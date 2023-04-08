model = dict(type='MMEEGNet', nChan=64, nTime=480)

dataset = dict(type='EEGDataset', 
               data_root='/home/msai/anishmad001/codes/EEG-Conformer/data/phisionet', 
               subs=[1,2,3], 
               batch_size=1, 
               augment=False)

val_dataset = dict(type='EEGDataset', 
               data_root='/home/msai/anishmad001/codes/EEG-Conformer/data/phisionet', 
               subs=[4,5,6], 
               batch_size=1, 
               augment=False)