from torch.utils.data import Dataset
from data_pre_process import load


class TreeDataSet(Dataset):
    def __init__(self, data_dir):
        print('Loading data...')
        code, parent_matrix, brother_matrix, rel_par_ids, rel_bro_ids, comments = load(data_dir)
        self.code = code
        self.par_matrix = parent_matrix.view()
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
        data_set = TreeDataSet('')