from typing import List


def get_co_axes(dim_total: int, axes: List[int]) -> List[int]:
    axes_co = list(set(range(dim_total)).difference(set(axes)))
    return axes_co
