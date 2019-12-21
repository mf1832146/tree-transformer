import re
import numpy as np
import torch
import os


def load_txt(file_path):
    data = np.loadtxt(file_path, dtype=np.long)
    data = torch.from_numpy(data)
    # if torch.cuda.is_available():
    #     data = data.cuda()
    return data


def make_save_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def remove_simple_name(root):
    """
    移除单个节点
    """
    for node in traverse(root):
        if '=' not in node.label and '(SimpleName)' in node.label:
            if node.children[0].label[:11] != "identifier=":
                raise Exception("ERROR!")
            node.label = "SimpleName_" + node.children[0].label[11:]
            node.children = []
        elif node.label[:11] == "identifier=":
            node.label = "SimpleName_" + node.label[11:]
        elif node.label[:6] == "value=":
            node.label = "Value_" + node.label[6:]
    return root


def get_bracket(s):
    if "value=" == s[:6] or "identifier=" in s[:11]:
        return None
    p = "\(.+?\)"
    res = re.findall(p, s)
    if len(res) == 1:
        return res[0]
    return s


def modifier(root, dic):
    """
    将节点转化为编号
    """
    for node in traverse(root):
        if is_simple_name(node.label):
            if node.label not in dic:
                node.label = "SimpleName_<UNK>"
        elif is_value(node.label):
            if node.label not in dic:
                if is_num(node.label):
                    node.label = "Value_<NUM>"
                else:
                    node.label = "Value_<STR>"
        else:
            node.label = get_bracket(node.label)
        if node.label not in dic:
            raise Exception("Unknown word", node.label)
        node.label = dic[node.label]

    return root


def rebuild_tree(root, dic):
    """
    1. remove simpleName node
    2. replace unknown word with type_UNK
    3. convert the type to id
    """
    root = remove_simple_name(root)
    root = modifier(root, dic)

    return root


class Node:
    def __init__(self, label="", parent=None, children=[], num=0):
        self.label = label
        self.parent = parent
        self.children = children
        self.num = num


def traverse_label(root):
    """return list of tokens"""
    li = [root.label]
    for child in root.children:
        li += traverse_label(child)
    return(li)


def traverse(root):
    """traverse all nodes"""
    res = [root]
    for child in root.children:
        res = res + traverse(child)
    return(res)


def is_simple_name(label):
    return "SimpleName_" == label[:11]


def is_value(label):
    return "Value_" == label[:6]


def is_num(label):
    try:
        float(label)
    except ValueError:
        return False
    else:
        return True
