from graph_nets.data_structures.attribute import Attribute


class EdgeUpdate:

    def __call__(self, e: Attribute, v_r: Attribute, v_s: Attribute, u: Attribute) -> Attribute:
        """
        Edge update function phi^(e). Converts e into e' given the attributes of the two adjacent nodes and the global
        state u.
        :param e: Edge attribute
        :param v_r: Receiver node attribute
        :param v_s: Sender node attribute
        :param u: Global state
        :return: New edge attribute e'
        """
        raise NotImplementedError


class NodeUpdate:

    def __call__(self, aggr_e: Attribute, v: Attribute, u: Attribute) -> Attribute:
        """
        Node update function phi^(v). Converts v into v' given the aggregated, updated edge attributes of all adjacent
        edges and the global state u.
        :param aggr_e: Aggregated, updated edges
        :param v: Node attribute
        :param u: Global state
        :return: New node attribute v'
        """
        raise NotImplementedError


class GlobalStateUpdate:

    def __call__(self, aggr_e: Attribute, aggr_v: Attribute, u: Attribute) -> Attribute:
        """
        Global state update function phi^(u). Converts u into u' given the aggregated edges and aggregated nodes.
        :param aggr_e: Aggregated edges
        :param aggr_v: Aggregated nodes
        :param u: Global state
        :return: New global state u'
        """
        raise NotImplementedError
