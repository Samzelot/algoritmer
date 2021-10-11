#from mpi4py import MPI
from FEMRect import Room, Side, plot_heatmap
import numpy as np
import mpi4py as MPI

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def aa(width,height,h):
    comm = MPI.Comm.Clone(MPI.COMM_WORLD)
    rank = comm.Get_rank()

    #Initialize
    room_1 = Room(width,height,h)
    room_2 = Room(width,height*2,h)
    room_3 = Room(width,height,h)

    #TODO: Create 3 discrete vectors u_1,u_2,u_3 as initial vectors
    u_1,u_2,u_3 = np.array([]), np.array([]), np.array([])

    for i in range(10):# 10 iterations
        if rank == 0:
            """
            Solve next iterate u_2, k+1 on room Ohm_2, Dirichlet conditions on Gamma_1 and Gamma_2 given by inital u_1,u_3, k
            """
            #TODO: solve room 2

            comm.send(solution,dest=1)
            comm.send(solution,dest=2)
        if rank == 1:
            """
            Recieve data from 0 solve next iterate u_1 with Neumann conditions on Gamma_1 given by u_2, k + 1
            """
            data = comm.recv(source=0)

            #send data back to rank0
        if rank == 2:
            """
            Recieve data from 0 solve next iterate u_3 with Neumann conditions on Gamma_2 given by u_2, k+1
            """
            data = comm.recv(source=0)
            #send data back to rank0 
            
def main():

    w = 20
    h = 20
    room = Room(w, h, 1/(h - 1))

    lower, upper, left, right = np.zeros(w), np.zeros(w), np.zeros(h), np.zeros(h)
    lower[10:15] = 3
    left[10:20] = -1
    left[30:40] = -1
    right[10:40] = -1

    boundaries = [
        {"type": "neumann", "side": Side.UPPER, "start": 0, "end": w, "values": 20*np.ones(w)},
        {"type": "dirichlet", "side": Side.LEFT, "start": 0, "end": h, "values": 40*np.ones(h)},
        {"type": "dirichlet", "side": Side.RIGHT, "start": 0, "end": h, "values": 40*np.ones(h)},
        {"type": "dirichlet", "side": Side.LOWER, "start": 0, "end": w, "values": 40*np.ones(w)},
    ]
    res = room.solve(boundaries)
    plot_heatmap(res)

if __name__ == "__main__":
    main()