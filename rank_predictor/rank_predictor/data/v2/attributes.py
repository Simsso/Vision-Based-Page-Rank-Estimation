from typing import Optional, Dict
from graph_nets import Attribute, torch
import numpy as np


class PageAttribute(Attribute):

    def __init__(self, base_url: str, desktop_img: np.ndarray, mobile_img: np.ndarray, client_status: Optional,
                 https: bool, server_status: Optional[int], start_node: bool, title: str) -> None:
        assert base_url is not None, "'base_url' must not be None"
        if client_status is None:
            client_status = ''
        if https is None:
            https = False
        if server_status is None:
            server_status = 0
        if start_node is None:
            start_node = False
        if title is None:
            title = ''
        val = {
            'base_url': base_url,
            'desktop_img': torch.Tensor(desktop_img),
            'mobile_img': torch.Tensor(mobile_img),
            'client_status': client_status,
            'https': https,
            'server_status': server_status,
            'start_node': start_node,
            'title': title,
        }
        super().__init__(val)

    @staticmethod
    def from_json(json: Dict, desktop_img: np.ndarray, mobile_img: np.ndarray) -> 'PageAttribute':
        return PageAttribute(
            json['base_url'],
            desktop_img,
            mobile_img,
            json['client_status'],
            json['https'],
            json['server_status'],
            json['startNode'],
            json['title']
        )


class LinkAttribute(Attribute):

    def __init__(self, val: str) -> None:
        super().__init__(val)
