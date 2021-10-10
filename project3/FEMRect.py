import numpy as np
import scipy.linalg as sclin
import enum
import matplotlib.pyplot as plt

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
    def __init__(self, width, height, h):

        self.width = width
        self.height = height
        self.h = h
        self.N = self.width*self.height

        self.K = self.K_matrix()
        self.f = np.zeros(self.N)
        

    def solve(self, boundaries):
        boundary_type = {
            "neumann": self.add_neumann,
            "dirichlet": self.add_dirichlet,
        }

        #type, side, start, end, values
        for b in boundaries:
            boundary_type[b["type"]](**b)      # Runs add_neuman or add_dirichlet 
        return np.linalg.solve(self.K, self.f).reshape(self.height, self.width)

    def add_neumann(self, side, start, end, values, **kwargs):
        ax_start, dir = SIDES_AXES[side]
        normal = np.array([-dir[1], dir[0]])
        for i in range(start, end):
            n = self.v_index(*(ax_start + i*dir))
            n_adjacent = [self.v_index(*(ax_start + y*normal + (x + i)*dir)) for x, y in [(-1, 0), (1, 0), (0, -1)]]
            self.K[n,n] = -3
            self.K[n,n_adjacent] = 1
            self.f[n] = values[i]*self.h

    def add_dirichlet(self, side, start, end, values, **kwargs):
        ax_start, dir = SIDES_AXES[side]
        for i in range(start, end):
            n = self.v_index(*(ax_start + i*dir))
            self.K[n,:] = 0
            self.K[n,n] = 1
            self.f[n] = values[i]

    def v_index(self, x,y):
        """Transforms coordinates to index of v"""
        return x % self.width + self.width*(y % self.height)
        
        
    def K_matrix(self):  # Creates the K matrix.
        c=np.zeros(self.N)
        c[0]=-4             # (i, j)   
        c[1]=1              # (i+1, j)
        c[self.width]=1     # (i, j+1)
        c[(self.height-1)*self.width]=1  # (i-1,j)
        c[-1]=1             # (i,j-1)

        K=sclin.circulant(c)    # Fills all rows.

        boundary_indexes = np.concatenate((   # Boundary indexes
            np.arange(0, self.width),
            -np.arange(0, self.width) - 1,
            np.arange(0, self.width*self.height, self.width),
            np.arange(self.width - 1, self.width*self.height, self.width)
        ))
        K[boundary_indexes,:] = 0   # Sets all "boundary rows" to zero
        return K
        
def plot_heatmap(f): #task 3, plot the heatmap
    plt.imshow(f, cmap='hot')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    room = Room(4, 4, 1/3, 1)
    print(room.K)