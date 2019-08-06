# -*- coding: utf-8 -*-


class GraphNode:
    def __init__(self, val, extra=None):
        self.val = val
        self.extra = extra


class Graph:
    def __init__(self, nodes):
        self.nodes = nodes
        self.edges = []

    def add_edge(self, from_, to, val=0):
        if from_ not in self.nodes:
            self.nodes.append(from_)
        if to not in self.nodes:
            self.nodes.append(to)
        self.edges.append([from_, to, val])

    def get_dests(self, origin):
        res = []
        for o, d, val in self.edges:
            if o == origin:
                res.append(d)
        return res, val

    def get_origins(self, dest):
        res = []
        for o, d, val in self.edges:
            if d == dest:
                res.append(o)
        return res, val

    def __getitem__(self, item):
        return self.nodes[item]


class LayeredHMMGraph(Graph):
    def __init__(self, nodes=[]):
        super().__init__(nodes)
        self.curr_layer = []

    def add_non_emitting_state(self):
        nes = GraphNode(None, 'NES')
        if len(self.curr_layer) == 0:
            self.nodes.append(nes)  # NES stands for non-emitting state
        else:
            self.nodes.append(nes)
            for m in self.curr_layer:
                self.add_edge(m, nes)
        self.curr_layer = [nes]
        return nes

    def add_layer_from_models(self, models):
        # assert that the previous layer is a non-emitting state
        assert len(self.curr_layer) == 1 and self.curr_layer[0].extra == 'NES'
        new_layer = []
        for i in range(len(models)):
            m = models[i]
            # get gmm states of all models
            gmm_nodes = [GraphNode(gs, extra=i) for gs in m.gmm_states]
            n_segments = len(gmm_nodes)
            # connect previous layer to the current one
            self.add_edge(self.curr_layer[0], gmm_nodes[0])

            # build graph for all gmm states in each hmm model
            for j in range(0, n_segments):
                # self loop
                self.add_edge(gmm_nodes[j], gmm_nodes[j], val=m.transitions[j, j])
                if j < n_segments - 2:
                    # connect to the next and the one after the next node
                    self.add_edge(gmm_nodes[j], gmm_nodes[j + 1], val=m.transitions[j + 1, j])
                    self.add_edge(gmm_nodes[j], gmm_nodes[j + 2], val=m.transitions[j + 2, j])
                elif j < n_segments - 1:
                    # connect to the next
                    self.add_edge(gmm_nodes[j], gmm_nodes[j + 1], val=m.transitions[j + 1, j])
                else:
                    new_layer.append(gmm_nodes[j])
        # update self.curr_layer to the last gmm_states in models
        self.curr_layer = new_layer
        return new_layer
