"""this module defines some helper types that we can use 
for type checking in other modules"""

from typing import Annotated

import numpy as np

from beartype.vale import IsAttr, IsEqual, IsSubclass

# define a datatype for 2D numpy arrays of floats
NumpyFloat2DArray = Annotated[
    np.ndarray,
    IsAttr["ndim", IsEqual[2]]
    & IsAttr["dtype", IsAttr["type", IsSubclass[np.floating]]],
]


# define a datatype for 1D numpy arrays of floats
NumpyFloatVector = Annotated[
    np.ndarray,
    IsAttr["ndim", IsEqual[1]]
    & IsAttr["dtype", IsAttr["type", IsSubclass[np.floating]]],
]

# logical vector
NumpyBooleanVector = Annotated[
    np.ndarray,
    IsAttr["ndim", IsEqual[1]] & IsAttr["dtype", IsEqual["bool"]],
]

# a vector with any datatype
NumpyVector = Annotated[np.ndarray, IsAttr["ndim", IsEqual[1]]]


# define a datatype for 2D numpy arrays of floats
NumpyFloatArray = Annotated[
    np.ndarray,
    IsAttr["dtype", IsAttr["type", IsSubclass[np.floating]]],
]
