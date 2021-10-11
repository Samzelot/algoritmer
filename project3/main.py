#from mpi4py import MPI
from Room import Room, Side, plot_heatmap
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def aa(room, boundaries):
    comm = MPI.Comm.Clone(MPI.COMM_WORLD)
    rank = comm.Get_rank()

    width = 20
    height = 20

    #Initialize
    room_1 = Room(width,height,h)
    room_2 = Room(width,height*2,h)
    room_3 = Room(width,height,h)

    boundaries1 = [
        {"type": "dirichlet", "side": Side.UPPER, "start": 0, "end": width, "values": 40*np.ones(width)},
        {"type": "dirichlet", "side": Side.LOWER, "start": 0, "end": width, "values": 5*np.ones(width)},
        {"type": "dirichlet", "side": Side.LEFT, "start": 0, "end": height, "values": 15*np.ones(height)}, #upper part left side
    ]
    room_1.set_boundaries(boundaries1)

    boundaries2 = [
        {"type": "dirichlet", "side": Side.UPPER, "start": 0, "end": width, "values": 40*np.ones(width)},
        {"type": "dirichlet", "side": Side.LOWER, "start": 0, "end": width, "values": 5*np.ones(width)},
        {"type": "dirichlet", "side": Side.LEFT, "start": 0, "end": height, "values": 15*np.ones(height)}, #upper part left side
        #{"type": "neumann", "side": Side.LEFT, "start": 20, "end": h - 1, "values": np.zeros(h - 21)}, #lower part left side
        {"type": "dirichlet", "side": Side.RIGHT, "start": 0, "end": height, "values": 15*np.ones(height)}, #lower part right side
        #{"type": "neumann", "side": Side.RIGHT, "start": 20, "end": h - 1, "values": np.zeros(h - 21)}, #upper part right side
    ]
    room_2.set_boundaries(boundaries2)

    boundaries3 = [
        {"type": "dirichlet", "side": Side.UPPER, "start": 0, "end": width, "values": 40*np.ones(width)},
        {"type": "dirichlet", "side": Side.LOWER, "start": 0, "end": width, "values": 5*np.ones(width)},
        {"type": "dirichlet", "side": Side.RIGHT, "start": 0, "end": height, "values": 15*np.ones(height)}, #upper part left side
    ]
    room_3.set_boundaries(boundaries3)

    #TODO: Create 3 discrete vectors u_1,u_2,u_3 as initial vectors
    # u_1,u_2,u_3 = np.array([]), np.array([]), np.array([])

    # relationships = {
    #     0: {"adjacent": [1], "room": Room()},
    #     1: {"adjacent": [0, 2], "room": Room()},
    #     2: {"adjacent": [1], "room": right_room},
    # }


    for i in range(10):# 10 iterations
    
        if rank == 0: # room 1

            left_border_room_2 = comm.recv(source=1) # get u_1 from room_2
            

            boundaries1 = [{"type": "dirichlet", "side": Side.RIGHT, "start": 0, "end": height, "values": left_border_room_2}]
            room_1.set_boundaries(boundaries1)

            border_room_1 = room_1.solve()[:,-1][1:-2]

            comm.send(border_room_1,dest=1)

        if rank == 1: #room 2
            if i == 0:
                temperatures = room_2.solve()
                room_2_boarder_to_1 = None
                room_2_boarder_to_3 = None

            room_1_boarder = comm.recv(source=0)    # boarder of room 1
            room_3_boarder = comm.recv(source=2)    # boarder of room 2
            temperatures = room_2.solve()           # TODO
            room_2_boarder_to_1 = None              # TODO
            room_2_boarder_to_3 = None              # TODO
            
            
            comm.send(room_2_boarder_1, dest = 0)
            comm.send(room_2_boarder_2, dest = 2)
            
        if rank == 2: # room 3
            room_2_boarder_2 = comm.recv(source=1)
            #room_3_boarder = ? - run algorithm
            data = comm.recv(source=0)
            

def left_room(border, comm):
    pass
    
def center_room():
    pass
def right_room():
    pass
   
def main():
    
    w = 20
    h = 40
    boundaries = [
        {"type": "dirichlet", "side": Side.UPPER, "start": 0, "end": w, "values": 40*np.ones(w)},
        {"type": "dirichlet", "side": Side.LOWER, "start": 0, "end": w, "values": 5*np.ones(w)},
        {"type": "dirichlet", "side": Side.LEFT, "start": 0, "end": 20, "values": 15*np.ones(20)}, #upper part left side
        {"type": "dirichlet", "side": Side.RIGHT, "start": 0, "end": 20, "values": 15*np.ones(20)}, #lower part right side
    ]
    room = Room(w, h, 1/(w - 1), boundaries)
    temp = [
        {"type": "neumann", "side": Side.LEFT, "start": 20, "end": h, "values": np.zeros(h - 20)}, #lower part left side
        {"type": "neumann", "side": Side.RIGHT, "start": 20, "end": h, "values": np.zeros(h - 20)}, #upper part right side
    ]

    res = room.solve(temp)
    plt.imshow(res, cmap='hot')
    plt.colorbar()
    plt.show()

#run from command line "mpiexec -n numprocs python -m mpi4py pyfile "
def test(): 
    comm = MPI.Comm.Clone(MPI.COMM_WORLD)
    rank = comm.Get_rank()

    width = 20
    height = 20

    #Initialize
    room_1 = Room(width,height,h)
    room_2 = Room(width,height*2,h)
    room_3 = Room(width,height,h)

    boundaries1 = [
        {"type": "dirichlet", "side": Side.UPPER, "start": 0, "end": width, "values": 40*np.ones(width)},
        {"type": "dirichlet", "side": Side.LOWER, "start": 0, "end": width, "values": 5*np.ones(width)},
        {"type": "dirichlet", "side": Side.LEFT, "start": 0, "end": height, "values": 15*np.ones(height)}, #upper part left side
    ]
    room_1.set_boundaries(boundaries1)

    boundaries2 = [
        {"type": "dirichlet", "side": Side.UPPER, "start": 0, "end": width, "values": 40*np.ones(width)},
        {"type": "dirichlet", "side": Side.LOWER, "start": 0, "end": width, "values": 5*np.ones(width)},
        {"type": "dirichlet", "side": Side.LEFT, "start": 0, "end": height, "values": 15*np.ones(height)}, #upper part left side
        #{"type": "neumann", "side": Side.LEFT, "start": 20, "end": h - 1, "values": np.zeros(h - 21)}, #lower part left side
        {"type": "dirichlet", "side": Side.RIGHT, "start": 0, "end": height, "values": 15*np.ones(height)}, #lower part right side
        #{"type": "neumann", "side": Side.RIGHT, "start": 20, "end": h - 1, "values": np.zeros(h - 21)}, #upper part right side
    ]
    room_2.set_boundaries(boundaries2)

    boundaries3 = [
        {"type": "dirichlet", "side": Side.UPPER, "start": 0, "end": width, "values": 40*np.ones(width)},
        {"type": "dirichlet", "side": Side.LOWER, "start": 0, "end": width, "values": 5*np.ones(width)},
        {"type": "dirichlet", "side": Side.RIGHT, "start": 0, "end": height, "values": 15*np.ones(height)}, #upper part left side
    ]
    room_3.set_boundaries(boundaries3)
    for i in range(10):
        if rank == 0: #room 1 and 3
            if i == 0: # first iteration, guess initial values
                left_border_room_2=40*np.ones(height)
            else: # use solution from room_2
                left_border_room_2 = comm.recv(source=1)
            
            boundaries1 = [{"type": "dirichlet", "side": Side.RIGHT, "start": 0, "end": height, "values": left_border_room_2}] 
            room_1.set_boundaries(boundaries1) # set guess as boundaries
            border_room_1 = room_1.solve() #solution
            comm.send(border_room_1[:,-1][1:-2], dest=2) #send solution to room_2 core

        if rank ==1:
            if i == 0:
                right_border_room_2 = 40*np.ones(height)
            else:
                right_border_room_2 = comm.recv(source=1)
            
            boundaries3 = [{"type": "dirichlet", "side": Side.LEFT, "start": 0, "end": height, "values": right_border_room_2}]
            room_3.set_boundaries(boundaries3)
            border_room_3 = room_3.solve()
            comm.send(border_room_3[:,1][1:-2], dest=2)

        if rank==2:
            dir_room_1_border = comm.recv(source=0)
            dir_room_3_border = comm.recv(source=1)

            boundaries2 = [ #TODO: use neumann
                {"type": "dirichlet", "side": Side.LEFT, "start": 20, "end": height - 1, "values": dir_room_1_border},
                {"type": "dirichlet", "side": Side.RIGHT, "start": 20, "end": height - 1, "values": dir_room_3_border}, #upper part right side#lower part left side
            ]
            room_2.set_boundaries(boundaries2) 
            border_room_2=room_2.solve()
            
            comm.send(border_room_2[:,1][20:height-1], dest=0)
            comm.send(border_room_2[:,-1][20:height-1], dest=1)
    
    return border_room_1, border_room_2, border_room_3

if __name__ == "__main__":
    main()


