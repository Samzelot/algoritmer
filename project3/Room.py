import numpy as np
import scipy.linalg as sclin
import enum
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

class Side(enum.Enum):
   LEFT = 1
   RIGHT = 2
   UPPER = 3
   LOWER = 4

SIDES_AXES = {
    #               --- start --- dir ------
    Side.UPPER: np.array([[-1,0], [-1,0]]),
    Side.LEFT: np.array([[0,0], [0,1]]),     # start_ and direction
    Side.LOWER: np.array([[0,-1], [1,0]]),
    Side.RIGHT: np.array([[-1,-1],[0,-1]]),
}

class Room:
    def __init__(self, width, height, h, initial_boundaries):

        self.width = width
        self.height = height
        self.h = h
        self.N = self.width*self.height

        self.K = self.K_matrix()
        self.f = np.zeros(self.N)
        self.set_boundaries(self.K, self.f, initial_boundaries)
    
    def set_boundaries(self, K, f, boundaries):
        boundary_type = {
            "neumann": self.add_neumann,
            "dirichlet": self.add_dirichlet,
        }
        #type, side, start, end, values
        for b in boundaries:
            boundary_type[b["type"]](K, f, **b)      # Runs add_neuman or add_dirichlet 

    def solve(self, temp_boundaries):
        K_temp = self.K.copy()
        f_temp = self.f.copy()
        self.set_boundaries(K_temp, f_temp, temp_boundaries)
        return spsolve(sparse.csr_matrix(K_temp), f_temp).reshape(self.height, self.width)

    def add_neumann(self, K, f, side, start, end, values, **kwargs):
        self.add_boundary_cond(side, K, f, start, end, values, [1, -1, -self.h])
    
    def add_dirichlet(self, K, f, side, start, end, values, **kwargs):
        self.add_boundary_cond(side, K, f, start, end, values, [1, -2, -1])

    def add_boundary_cond(self, side, K, f, start, end, values, kernel):
        ax_start, dir = SIDES_AXES[side]
        normal = np.array([-dir[1], dir[0]])
        for i in range(start, end):
            n = self.v_index(*(ax_start + i*dir)) #n, gives v_index on the axis we are stepping in (dir) (boundary node)
            n_adjacent = self.v_index(*(ax_start - normal + i*dir)) #finds adjacant v_index around current boundary node (n)
            K[n,n_adjacent] += kernel[0] # add element to adjacent v_indeces, dependent on condition
            K[n,n] += kernel[1] #Diagonal element add, finite diff approx coefficient
            f[n] += values[i - start]*kernel[2]

    def v_index(self, x,y):
        """Transforms coordinates to index of v"""
        return x % self.width + self.width*(y % self.height)
        
    def K_matrix(self):  # Creates the K matrix.
        K = sparse.lil_matrix((self.N, self.N))
        for i in range(1, self.width - 1):
            for j in range(self.height):
                v = [self.v_index(i + m, j) for m in [-1, 0, 1]]
                K[v[1],v] += np.array([1, -2, 1])
        for i in range(self.width):
            for j in range(1, self.height - 1):
                v = [self.v_index(i, j + m) for m in [-1, 0, 1]]
                K[v[1],v] += np.array([1, -2, 1])
        return K
        
def plot_heatmap(room1, room2, room3): #task 3, plot the heatmap
    A=np.zeros((40, 60))

    A[20:,:20]=room1
    A[:,20:40]=room2
    A[:20,40:]=room3

    plt.imshow(A, cmap='hot')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    room = Room(4, 4, 1/3, 1)
    print(room.K)