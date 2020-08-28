import networkx as nx
from grid.simulation import Simulation
from grid.Servicepoint import Servicepoint


class Node(object):
    def __init__(self, id, data):
        self.id = id
        self.data = data


class Grid(nx.DiGraph):
    def __init__(self, incoming_graph_data=None, **attr):
        super(Grid, self).__init__(incoming_graph_data, **attr)

    def get_servicepoints(self):
        return [n for n, d in self.nodes(data=True) if 'label' in d and 'Service Point' in d['label']]

    def get_transformers(self):
        return [n for n, d in self.nodes(data=True) if 'label' in d and 'Transformer' in d['label']]

    def get_fuses(self):
        return [n for n, d in self.nodes(data=True) if 'label' in d and 'Fuse' in d['label']]

    def get_transformers_from_fuse(self, fuse):
        return [Node(x[0], self.nodes[x[0]]) for x in self.in_edges(fuse) if 'Transformer' in self.nodes[x[0]]['label']]

    def get_servicepoint_from_transformer(self, transformer):
        return [Node(x[0], self.nodes[x[0]]) for x in self.in_edges(transformer) if 'Service Point' in self.nodes[x[0]]['label']]

    def get_servicepoints_from_fuse(self, fuse):
        service= list()
        for transformer in self.get_transformers_from_fuse(fuse):
            service.extend(self.get_servicepoint_from_transformer(transformer.id))
        return service

    def simulation(self, fuse, conf, distribution, duration, n, years):
        service_points = [Servicepoint("A6", conf) for x in self.get_servicepoints_from_fuse(fuse)]
        sim = Simulation(
            configuration=conf,
            consumptions=service_points,
            distribution=distribution,
            duration=duration,
            n=n,
            years=years
        )
        return sim
