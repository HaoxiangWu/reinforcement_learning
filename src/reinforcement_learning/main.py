# from mdp.exampleFiniteMDP import ExampleGridWorldEnv
from copy import deepcopy
import numpy as np
def main():
    a = np.array([1,2])
    b = deepcopy(a)
    b = b + 1
    print(type(a))
    print(type(b))
    pass


if __name__ == "__main__":
    main()
