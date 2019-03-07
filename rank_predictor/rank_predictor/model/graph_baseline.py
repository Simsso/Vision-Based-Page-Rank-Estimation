from torch import nn
from graph_nets import Graph
from rank_predictor.model.screenshot_feature_extractor import DesktopScreenshotFeatureExtractor


class GraphBaseline(nn.Module):

    def __init__(self):
        super().__init__()

        self.desktop_screenshot_extractor = DesktopScreenshotFeatureExtractor()

    def forward(self, g: Graph) -> Graph:
        pass
