from typing import List
from torch import Tensor


def tensors_stackable(tensors: List[Tensor]) -> bool:
    """
    torch.stack requires "All tensors need to be of the same size."
    This function returns True if that is the case for all tensors in the given list. False otherwise.
    :param tensors: List of tensors to check the sizes of
    :return: True if the tensors can be stacked
    """

    tensor_shapes = [t.shape for t in tensors]

    if len(tensor_shapes) == 0:
        return True

    # batch dimension must not be regarded because "Sizes of tensors must match except in dimension 0."
    target_shape = tensor_shapes[0]

    for shape in tensor_shapes:
        if shape != target_shape:
            return False

    return True
