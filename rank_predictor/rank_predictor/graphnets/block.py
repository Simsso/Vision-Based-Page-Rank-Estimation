from torch import nn, Tensor
from rank_predictor.data_structures.graph import Graph
from rank_predictor.graphnets.functions.aggregation import Aggregation
from rank_predictor.graphnets.functions.update import EdgeUpdate, NodeUpdate, GlobalStateUpdate


class GNBlock(nn.Module):

    def __init__(self,
                 phi_e: EdgeUpdate, phi_v: NodeUpdate, phi_u: GlobalStateUpdate,
                 rho_ev: Aggregation, rho_vu: Aggregation, rho_eu: Aggregation) -> None:
        """
        Graph network block initialization function.
        :param phi_e: Edge update function
        :param phi_v: Node update function
        :param phi_u: Global state update function
        :param rho_ev: Edge aggregation function for nodes
        :param rho_vu: Node aggregation function for the global state
        :param rho_eu: Edge aggregation function for the global state
        """
        super().__init__()

        self.phi_e, self.phi_v, self.phi_u = phi_e, phi_v, phi_u
        self.rho_ev, self.rho_vu, self.rho_eu = rho_ev, rho_vu, rho_eu

    def forward(self, g: Graph) -> Graph:
        g = self.forward_edge_block(g)
        g = self.forward_node_block(g)
        g = self.forward_global_block(g)
        return g

    def forward_edge_block(self, g: Graph) -> Graph:
        pass

    def forward_node_block(self, g: Graph) -> Graph:
        pass

    def forward_global_block(self, g: Graph) -> Graph:
        pass
