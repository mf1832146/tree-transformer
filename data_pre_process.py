import os
import sys
from glob import glob
from tqdm import tqdm
from utils import Node, traverse_label, rebuild_tree, load_txt
import nltk
import re
from collections import Counter
import pickle
import torch
import numpy as np


def pre_process():
    """
    pre process the data
    1. generate the vocab from the training data set
    2. deal with tree
    3. save file
    """
    dirs = [
        "./data",
        "./data/tree",
        "./data/tree/train",
        "./data/tree/valid",
        "./data/tree/test",
    ]
    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)

    data_dir = sys.argv[1]
    max_size = int(sys.argv[2])
    max_comment_size = int(sys.argv[3])
    k = int(sys.argv[4])

    for path in [data_dir + "/" + s for s in ["train", "valid", "test"]]:
        set_name = path.split("/")[-1]
        if set_name == 'train':
            generate_vocab(path)
        code_tensor, parent_matrix_tensor, brother_matrix_tensor, parent_ids_tensor, brother_ids_tensor, relative_parent_ids_tensor, relative_brother_ids_tensor, comments_tensor = deal_with_tree(path, max_size, k, max_comment_size)

        save_path = './data/tree/' + set_name + '/'

        save_to_file(code_tensor, save_path + 'code.txt')
        save_to_file(parent_matrix_tensor, save_path + 'parent_matrix.txt')
        save_to_file(brother_matrix_tensor, save_path + 'brother_matrix.txt')
        save_to_file(parent_ids_tensor, save_path + 'parent_ids.txt')
        save_to_file(brother_ids_tensor, save_path + 'brother_ids.txt')
        save_to_file(relative_parent_ids_tensor, save_path + 'relative_parents.txt')
        save_to_file(relative_brother_ids_tensor, save_path + 'relative_brothers.txt')
        save_to_file(comments_tensor, save_path + 'comments.txt')


def load(save_path, max_size):
    code = load_txt(save_path + 'code.txt')
    parent_matrix = load_txt(save_path + 'parent_matrix.txt').view(-1, max_size, max_size)
    brother_matrix = load_txt(save_path + 'brother_matrix.txt').view(-1, max_size, max_size)
    rel_par_ids = load_txt(save_path + 'relative_parents.txt').view(-1, max_size, max_size)
    rel_bro_ids = load_txt(save_path + 'relative_brothers.txt').view(-1, max_size, max_size)
    comments = load_txt(save_path + 'comments.txt')

    return code, parent_matrix, brother_matrix, rel_par_ids, rel_bro_ids, comments


def deal_with_tree(data_dir, max_size, k, max_comment_size):
    files = sorted(glob(data_dir + "/*"))
    code_dic = pickle.load(open("./data/code_w2i.pkl", "rb"))
    comment_dic = pickle.load(open("./data/nl_w2i.pkl", "rb"))
    skip = 0

    code_data = []
    parent_matrix_data = []
    brother_matrix_data = []
    parent_ids_data = []
    brother_ids_data = []
    relative_parent_ids_data = []
    relative_brother_ids_data = []
    comments = []

    for file in tqdm(files, "traverse tree data from {}".format(data_dir)):
        tree, nl = parse(file)

        nl = clean_nl(nl)
        if is_invalid_com(nl):
            skip += 1
            continue
        if is_invalid_tree(tree):
            skip += 1
            continue
        seq = tokenize(nl)
        if is_invalid_seq(seq):
            skip += 1
            continue
        tree = rebuild_tree(tree, code_dic)
        code_seq, parent_matrix, brother_matrix, parent_ids, brother_ids, relative_parent_ids, relative_brother_ids = traverse(tree, max_size, k)

        code_data.append(code_seq)
        parent_matrix_data.append(parent_matrix)
        brother_matrix_data.append(brother_matrix)
        parent_ids_data.append(parent_ids)
        brother_ids_data.append(brother_ids)
        relative_parent_ids_data.append(relative_parent_ids)
        relative_brother_ids_data.append(relative_brother_ids)
        seq_tensor = convert_comment_to_ids(seq, comment_dic, max_comment_size)
        comments.append(seq_tensor)

    code_tensor = torch.stack(code_data, dim=0)
    parent_matrix_tensor = torch.stack(parent_matrix_data, dim=0).view(-1, max_size * max_size)
    brother_matrix_tensor = torch.stack(brother_matrix_data, dim=0).view(-1, max_size * max_size)
    parent_ids_tensor = torch.stack(parent_ids_data, dim=0)
    brother_ids_tensor = torch.stack(brother_ids_data, dim=0)
    relative_parent_ids_tensor = torch.stack(relative_parent_ids_data, dim=0).view(-1, max_size * max_size)
    relative_brother_ids_tensor = torch.stack(relative_brother_ids_data, dim=0).view(-1, max_size * max_size)
    comments_tensor = torch.stack(comments, dim=0)

    print('skip num', skip)

    return code_tensor, parent_matrix_tensor, brother_matrix_tensor, parent_ids_tensor, brother_ids_tensor, relative_parent_ids_tensor, relative_brother_ids_tensor, comments_tensor


def save_to_file(data_tensor, file_name):
    data_tensor = data_tensor.numpy()
    np.savetxt(file_name, data_tensor, fmt='%d')


def generate_vocab(path):
    files = sorted(glob(path + "/*"))
    set_name = path.split("/")[-1]

    nls = {}
    skip = 0

    for file in tqdm(files, "generate vocab from {}".format(path)):
        tree, nl = parse(file)
        nl = clean_nl(nl)
        if is_invalid_com(nl):
            skip += 1
            continue
        if is_invalid_tree(tree):
            skip += 1
            continue
        seq = tokenize(nl)
        if is_invalid_seq(seq):
            skip += 1
            continue

        nls[tree] = seq

    "comment vocab"
    comment_vocab = Counter([x for l in nls.values() for x in l])
    nl_i2w = {i: w for i, w in enumerate(
        ["<PAD>", "<UNK>"] + sorted([x[0] for x in comment_vocab.most_common(30000)]))}
    nl_w2i = {w: i for i, w in enumerate(
        ["<PAD>", "<UNK>"] + sorted([x[0] for x in comment_vocab.most_common(30000)]))}
    pickle.dump(nl_i2w, open("./data/nl_i2w.pkl", "wb"))
    pickle.dump(nl_w2i, open("./data/nl_w2i.pkl", "wb"))

    "code vocab"
    code_labels = [traverse_label(c) for c in nls.keys()]
    code_labels = [l for s in code_labels for l in s]
    non_terminals = set(
        [get_bracket(x) for x in tqdm(
            list(set(code_labels)), "collect non-tarminals")]) - set([None, "(SimpleName)"])
    non_terminals = sorted(list(non_terminals))
    ids = Counter(
        [y for y in [get_identifier(x) for x in tqdm(
            code_labels, "collect identifiers")] if y is not None])
    ids_list = [x[0] for x in ids.most_common(30000)]
    values = Counter(
        [y for y in [get_values(x) for x in tqdm(
            code_labels, "collect values")] if y is not None])
    values_list = [x[0] for x in values.most_common(1000)]

    vocab = ["<UNK>", "SimpleName_<UNK>", "Value_<NUM>", "Value_<STR>"]
    vocab += non_terminals + ids_list + values_list + ["(", ")"]

    code_i2w = {i: w for i, w in enumerate(vocab)}
    code_w2i = {w: i for i, w in enumerate(vocab)}

    pickle.dump(code_i2w, open("./data/code_i2w.pkl", "wb"))
    pickle.dump(code_w2i, open("./data/code_w2i.pkl", "wb"))


def get_values(s):
    if "value=" == s[:6]:
        return "Value_" + s[6:]
    else:
        return None


def get_identifier(s):
    if "identifier=" == s[:11]:
        return "SimpleName_" + s[11:]
    else:
        return None


def get_bracket(s):
    if "value=" == s[:6] or "identifier=" in s[:11]:
        return None
    p = "\(.+?\)"
    res = re.findall(p, s)
    if len(res) == 1:
        return res[0]
    return s


def parse(path):
    with open(path, 'r') as f:
        try:
            num_objects = f.readline()
            nodes = [Node(num=i, children=[]) for i in range(int(num_objects))]
            for i in range(int(num_objects)):
                label = " ".join(f.readline().split(" ")[1:])[:-1]
                nodes[i].label = label
            while 1:
                line = f.readline()
                if line == "\n":
                    break
                p, c = map(int, line.split(" "))
                nodes[p].children.append(nodes[c])
                nodes[c].parent = nodes[p]
            nl = f.readline()[:-1]
        except Exception as e:
            print(path)
            raise e
    return nodes[0], nl


def clean_nl(s):
    if s[-1] == ".":
        s = s[:-1]
    s = s.split(". ")[0]
    s = re.sub("[<].+?[>]", "", s)
    s = re.sub("[\[\]\%]", "", s)
    s = s[0:1].lower() + s[1:]
    return s


def is_invalid_com(s):
    return s[:2] == "/*" and len(s) > 1


def is_invalid_seq(s):
    return len(s) < 4


def get_method_name(root):
    for c in root.children:
        if c.label == "name (SimpleName)":
            return c.children[0].label[12:-1]


def is_invalid_tree(root):
    labels = traverse_label(root)
    if root.label == 'root (ConstructorDeclaration)':
        return True
    if len(labels) >= 1000:
        return True
    method_name = get_method_name(root)
    for word in ["test", "Test", "set", "Set", "get", "Get"]:
        if method_name[:len(word)] == word:
            return True
    return False


def convert_comment_to_ids(comment, dic, max_len):
    nl = torch.zeros(max_len)
    for i, word in enumerate(comment):
        if i >= max_len:
            break
        if word not in dic:
            nl[i] = dic['<UNK>']
        else:
            nl[i] = dic[word]
    return nl


def tokenize(s):
    return ["<s>"] + nltk.word_tokenize(s) + ["</s>"]


def traverse(tree, max_size, k):
    """
     广度优先遍历树， 生成广度优先遍历序列，兄弟，父子遮盖矩阵, 相对位置矩阵
    :param tree:
    :param max_size:
    :param k:
    :return:
    """
    root_id = tree.num
    queue = [tree]
    parent_map = {}
    brother_map = {}

    seq = torch.zeros(max_size)
    parent_matrix = torch.ones((max_size, max_size))
    brother_matrix = torch.ones((max_size, max_size))
    parent_ids = torch.zeros(max_size)
    brother_ids = torch.zeros(max_size)
    relative_parent_ids = torch.zeros((max_size, max_size))
    relative_brother_ids = torch.zeros((max_size, max_size))

    while queue:
        current_node = queue.pop()
        node_id = current_node.num

        if node_id >= max_size:
            continue

        seq[node_id] = current_node.label
        if node_id == root_id:
            parent_map[node_id] = [node_id]
            brother_map[node_id] = [node_id]

        if len(current_node.children) > 0:
            brother_node_ids = [x.num for x in current_node.children if x.num < max_size]
            for child in current_node.children:
                child_id = child.num
                if child_id >= max_size:
                    continue

                parent_map[child_id] = parent_map[node_id] + [child_id]
                brother_map[child_id] = brother_node_ids

            queue.extend(current_node.children)

    for node_id in parent_map:
        for parent_id in parent_map[node_id]:
            parent_matrix[node_id][parent_id] = 0
            parent_matrix[parent_id][node_id] = 0
        for brother_id in brother_map[node_id]:
            brother_matrix[node_id][brother_id] = 0
        # 位置顺序
        parent_ids[node_id] = parent_map[node_id].index(node_id) + 1
        brother_ids[node_id] = brother_map[node_id].index(node_id) + 1

    # 相对位置
    for i in range(max_size):
        for j in range(max_size):
            # 0为填充位置，不需要考虑
            if parent_ids[i] != 0 and parent_ids[j] != 0:
                # 将相对位置从-k~k调整到1～2k+1, 0仍然是填充向量
                relative_parent_ids[i][j] = max(-k, min(k, parent_ids[j]-parent_ids[i])) + k + 1
            if brother_ids[i] != 0 and brother_ids[j] != 0:
                relative_brother_ids[i][j] = max(-k, min(k, brother_ids[j]-brother_ids[i])) + k + 1

    return seq, parent_matrix, brother_matrix, parent_ids, brother_ids, relative_parent_ids, relative_brother_ids


if __name__ == '__main__':
    pre_process()
