# -*- coding: utf-8 -*-
from collections import Counter


class LexNode:
    def __init__(self, val):
        self.val = val
        self.children = []
        # extra field for properties, used like C-style enum.
        # 0: normal node
        # 1: start(root) node
        # 2: end-of-word node
        self.property = 0

    # pretty print tree in command line
    def pretty_str(self, level=0):
        ret = "\t" * level + repr(self.val) + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

    # only return the value of itself, in order to provide better idea what it is when debugging
    def __str__(self):
        return self.val

    def get_max_level(self, level=0):
        if len(self.children) == 0:
            return level
        child_levels = [child.get_max_level(level=level + 1) for child in self.children]
        return max(child_levels)


def append_lex_node(parent, child):
    assert type(parent) is LexNode and type(child) is LexNode
    parent.children.append(child)


def _build_lextree(node, words, i, max_word_len):
    if i >= max_word_len:
        return
    for w in words:
        # if current character is the end of a word, add it as a separate node even if the same node can be shared. This
        # is to ensure that every leaf represent exactly one word.
        if (i + 1 < max_word_len and w[i + 1] == ' ') or i == max_word_len - 1:
            child = LexNode(w[i])
            child.property = 2
            append_lex_node(node, child)
            words.remove(w)
    n_words = len(words)
    # get the characters in j-th position
    chars = [words[n][i] for n in range(n_words)]
    counts = Counter(chars)
    for ch, count in counts.items():
        child = LexNode(ch)
        append_lex_node(node, child)
        new_words = [x for _, x in enumerate(words) if x[i] == ch]
        _build_lextree(child, new_words, i + 1, max_word_len)


def lextree_from_words(words):
    """
    Construct a lexical tree using provided word list.
    :param words: a list of words.
    :return: constructed lexical tree, see lexnode data structure.
    """
    # dummy symbol for the root of the tree
    tree = LexNode('*')
    tree.property = 1
    n_words = len(words)
    word_lens = [len(w) for w in words]
    max_word_len = max(word_lens)
    # pad the words to the same length using ' ', in order to avoid index out of range
    for n in range(n_words):
        words[n] = words[n].ljust(max_word_len, ' ')
    _build_lextree(tree, words, 0, max_word_len)
    return tree
