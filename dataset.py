from torch.utils.data import Dataset, DataLoader
from data_pre_process import load


class TreeDataSet(Dataset):
    def __init__(self, data_dir, max_size, skip=0):
        print('Loading data...')
        code, parent_matrix, brother_matrix, rel_par_ids, rel_bro_ids, comments = load(data_dir, max_size, skip)
        self.code = code
        self.par_matrix = parent_matrix
        self.bro_matrix = brother_matrix
        self.rel_par_ids = rel_bro_ids
        self.rel_bro_ids = rel_bro_ids
        self.comments = comments
        self.len = self.code.shape[0]
        print('Loading finished')

    def __getitem__(self, index):
        return self.code[index], self.par_matrix[index], self.bro_matrix[index], self.rel_par_ids[index], self.rel_bro_ids[index], self.comments[index]

    def __len__(self):
        return self.len


if __name__ == '__main__':
    data_set = TreeDataSet('./data/tree/train/', max_size=100)
    train_loader2 = DataLoader(dataset=data_set,
                               batch_size=32,
                               shuffle=True)
    for i, data in enumerate(train_loader2):
        code, par_matrix, bro_matrix, rel_par_ids, rel_bro_ids, comm = data
        print(code.shape[0])
