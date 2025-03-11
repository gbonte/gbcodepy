#!/usr/bin/env python3
import numpy as np
import sys

# Define the Table class to mimic R's table object
class Table:
    def __init__(self, data=None, dimnames=None, *args):
        # If data is provided, use it directly along with dimnames
        if data is not None and dimnames is not None:
            self.values = np.array(data, dtype=float)
            self.dimnames = dimnames
        else:
            # Build table from raw vectors passed in *args
            if len(args) == 0:
                raise ValueError("No input vectors provided to table()")
            n_dims = len(args)
            n = len(args[0])
            # Ensure all vectors have the same length
            for vec in args:
                if len(vec) != n:
                    raise ValueError("All input vectors must have the same length")
            # For each dimension, get sorted unique levels (as strings)
            self.dimnames = []
            for vec in args:
                # Convert each element to string and get unique values
                levels = sorted(set(str(x) for x in vec), key=lambda y: float(y) if y.replace('.','',1).isdigit() else y)
                self.dimnames.append(levels)
            # Create an array of zeros with the corresponding shape
            shape = tuple(len(levels) for levels in self.dimnames)
            self.values = np.zeros(shape, dtype=float)
            # Populate counts
            for i in range(n):
                indices = []
                for d, vec in enumerate(args):
                    val = str(vec[i])
                    index = self.dimnames[d].index(val)
                    indices.append(index)
                self.values[tuple(indices)] += 1

    @property
    def ndim(self):
        return len(self.dimnames)

    def _convert_keys(self, key):
        # Helper to convert keys from labels to indices
        # If key is not a tuple, treat it as for the first dimension.
        if not isinstance(key, tuple):
            key = (key,)
        # Extend key with slice(None) if fewer keys than dimensions
        if len(key) < self.ndim:
            key = key + (slice(None),) * (self.ndim - len(key))
        # Convert each key: if key is a string or number, convert to string and then to index;
        # if key is slice, keep it.
        new_key = []
        for d, k in enumerate(key):
            if isinstance(k, (str, int, float)):
                sk = str(k)
                try:
                    idx = self.dimnames[d].index(sk)
                except ValueError:
                    raise KeyError(f"Label '{sk}' not found in dimension {d}")
                new_key.append(idx)
            else:
                new_key.append(k)
        return tuple(new_key)

    def __getitem__(self, key):
        k = self._convert_keys(key)
        result = self.values[k]
        # Check if the result is a scalar
        if np.isscalar(result):
            return result
        # If the result is an array, try to rebuild a Table if possible
        # Determine the remaining dimnames based on the key
        # If key is an integer for a dimension then that dimension is dropped
        new_dimnames = []
        for d, sub in enumerate(key):
            if isinstance(sub, slice):
                new_dimnames.append(self.dimnames[d])
        # If no dimensions remain, return the scalar result
        if len(new_dimnames) == 0:
            return result
        return Table(data=result, dimnames=new_dimnames)

    def __setitem__(self, key, value):
        k = self._convert_keys(key)
        self.values[k] = value

# Function to mimic R's table() function
def table(*args):
    return Table(None, None, *args)

# Function to mimic R's prop.table() function: returns proportions over the entire table
def prop_table(tabe):
    total = np.sum(tabe.values)
    if total == 0:
        raise ValueError("Total sum of table is zero, cannot compute proportions")
    new_values = tabe.values / total
    return Table(data=new_values, dimnames=tabe.dimnames)

# Function to mimic R's margin.table() function
def margin_table(tabe, dims):
    # dims: list or tuple of R dimension indices (1-indexed) to keep in the marginal table
    dims_python = [d - 1 for d in dims]
    ndim = tabe.ndim
    # Determine the remaining axes to sum over and reorder axes accordingly
    remaining = [d for d in range(ndim) if d not in dims_python]
    # Permute the array: first the desired dims in the order specified, then the rest
    perm = dims_python + remaining
    transposed = np.transpose(tabe.values, axes=perm)
    # Sum over the remaining axes
    new_values = np.sum(transposed, axis=tuple(range(len(dims_python), ndim)))
    new_dimnames = [tabe.dimnames[d] for d in dims_python]
    return Table(data=new_values, dimnames=new_dimnames)

# The risksT function as defined in the R code
def risksT(tabe, z=None):
    # Check if tabe is an instance of Table
    if not isinstance(tabe, Table):
        raise ValueError("Non table inputs")
    
    # Compute proportions of the table
    P = prop_table(tabe)
    
    # If the table is 2-dimensional
    if P.ndim == 2:
        p11 = P["1", "1"]
        p10 = P["1", "0"]
        p01 = P["0", "1"]
        p00 = P["0", "0"]
        return {"RD": p11 / (p11 + p10) - p01 / (p01 + p00),
                "RR": (p11 / (p11 + p10)) / (p01 / (p01 + p00)),
                "OR": (p11 * p00) / (p10 * p01)}
    
    # For tables with more than 2 dimensions, use the z parameter to slice the table
    if isinstance(z, (int, float)):
        # Convert numeric z to string as done in R by as.character(z)
        z_str = str(z)
        P = prop_table(tabe[:, :, z_str])
    else:
        P = prop_table(tabe[:, :, z])
    
    p11 = P["1", "1"]
    p10 = P["1", "0"]
    p01 = P["0", "1"]
    p00 = P["0", "0"]
    return {"RD": p11 / (p11 + p10) - p01 / (p01 + p00),
            "RR": (p11 / (p11 + p10)) / (p01 / (p01 + p00)),
            "OR": (p11 * p00) / (p10 * p01)}

# rm(list=ls()) equivalent in Python:
# WARNING: Clearing globals can be dangerous. Here we simulate it by deleting names
# defined before this point, but we must leave the imported modules and built-ins.
for name in list(globals().keys()):
    if name not in ["np", "sys", "Table", "table", "prop_table", "margin_table", "risksT"]:
        if not name.startswith("__"):
            del globals()[name]

# ------------------------------
# First part of the R code: Confounder configuration
Z = [0, 1]  # variable G in the text
X = [0, 1]  # variable T in the text
Y = [0, 1]  # variable Y in the text

tabe = table(X, Y, Z)

# Section 16.2.1 of SFML
# Confounder configuration
# treatment x
# outcome y (1:recovery,0: non recovery)
# covariate z sex (0: Female, Male)

tabe["0", "1", "0"] = 1
tabe["0", "0", "0"] = 4
tabe["1", "1", "0"] = 2
tabe["1", "0", "0"] = 6

tabe["0", "1", "1"] = 6
tabe["0", "0", "1"] = 2
tabe["1", "1", "1"] = 4
tabe["1", "0", "1"] = 1

print("\n --\n SFML:  confounder case \n", " Impact of gender on outcome  =",
      risksT(margin_table(tabe, [2, 3]))["RD"], "\n")

print("\n SFML: mediator case \n", " Impact of gender on treatment  =",
      risksT(margin_table(tabe, [3, 1]))["RD"], "\n")

print("\n SFML example: confounder case \n", " Non conditioned risks=",
      risksT(margin_table(tabe, [1, 2]))["RD"], "\n",
      " Conditioned risk | Female =", risksT(tabe, "0")["RD"],
      " Conditioned risk | Male =", risksT(tabe, "1")["RD"])

# ------------------------------
# Second part of the R code: Mediator configuration
# Section 16.2.1 of SFML
# 2nd intepretation of the data
# Mediator configuration
# treatment x
# outcome y (1:recovery,0: non recovery)
# mediating variable (e.g. enzyme) whose rate has a positive impact on recovery (O: low, 1:high)

Z = ["0", "1"]
X = [0, 1]
Y = [0, 1]

tabe = table(X, Y, Z)
tabe["0", "1", "0"] = 1
tabe["0", "0", "0"] = 4
tabe["1", "1", "0"] = 2
tabe["1", "0", "0"] = 6

tabe["0", "1", "1"] = 6
tabe["0", "0", "1"] = 2
tabe["1", "1", "1"] = 4
tabe["1", "0", "1"] = 1

print("\n --\n SFML: mediator case \n", " Impact of treatment on mediating var =",
      risksT(margin_table(tabe, [1, 3]))["RD"], "\n")

print("\n SFML example: mediator case \n", " Non conditioned risks=",
      risksT(margin_table(tabe, [1, 2]))["RD"], "\n",
      " Conditioned risk | 0 =", risksT(tabe, 0)["RD"],
      " Conditioned risk | 1 =", risksT(tabe, 1)["RD"])
      
if __name__ == '__main__':
    pass
       
# End of Python translation of the provided R code

