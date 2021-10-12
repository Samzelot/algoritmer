#from mpi4py import MPI
from Room import Room, Side, plot_heatmap
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    
def test():
    
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
    plt.imshow(res, cmap='hot', vmin=0, vmax=50)
    plt.colorbar()
    plt.show()

#run from command line "mpiexec -n numprocs python -m mpi4py pyfile "

def main(omega = 0.8): 
    comm = MPI.Comm.Clone(MPI.COMM_WORLD)
    rank = comm.Get_rank()

    width = 40
    height = 40
    h = 1/(width - 1)

    #Initialize
    boundaries1 = [
        {"type": "dirichlet", "side": Side.UPPER, "start": 0, "end": width, "values": 15*np.ones(width)},
        {"type": "dirichlet", "side": Side.LOWER, "start": 0, "end": width, "values": 15*np.ones(width)},
        {"type": "dirichlet", "side": Side.LEFT, "start": 0, "end": height, "values": 40*np.ones(height)}, #upper part left side
    ]
    room_1 = Room(width,height,h, boundaries1)

    boundaries2 = [
        {"type": "dirichlet", "side": Side.UPPER, "start": 0, "end": width, "values": 40*np.ones(width)},
        {"type": "dirichlet", "side": Side.LOWER, "start": 0, "end": width, "values": 5*np.ones(width)},
        {"type": "dirichlet", "side": Side.LEFT, "start": 0, "end": height, "values": 15*np.ones(height)}, #upper part left side
        {"type": "dirichlet", "side": Side.RIGHT, "start": 0, "end": height, "values": 15*np.ones(height)}, #lower part right side
    ]
    room_2 = Room(width,height*2,h, boundaries2)

    boundaries3 = [
        {"type": "dirichlet", "side": Side.UPPER, "start": 0, "end": width, "values": 15*np.ones(width)},
        {"type": "dirichlet", "side": Side.LOWER, "start": 0, "end": width, "values": 15*np.ones(width)},
        {"type": "dirichlet", "side": Side.RIGHT, "start": 0, "end": height, "values": 40*np.ones(height)}, #upper part left side
    ]
    room_3 = Room(width,height,h,boundaries3)
    
    iters = 10
    for i in range(iters):
        if rank == 0: #room 1
            left_border_room_2 = np.flip(comm.recv(source=2))
            
            boundaries1 = [{"type": "neumann", "side": Side.RIGHT, "start": 0, "end": height, "values":left_border_room_2}] 

            room_1_temps = room_1.solve(boundaries1) #solution

            #u_knext1 = border_room_1[:,-1][1:-2] #get gamma
            #u_sol1 = omega*u_knext1 + (1-omega)*left_border_room_2[:,-1][1:-2] # relaxation, uses next iterate and previous
            comm.send(room_1_temps[:,-1], dest=2) if i != iters-1 else comm.send(room_1_temps, dest=2) #send solution to room_2 core

        if rank ==1: # room 3
            right_border_room_2 = comm.recv(source=2)
            
            boundaries3 = [{"type": "neumann", "side": Side.LEFT, "start": 0, "end": height, "values": right_border_room_2}]
            room_3_temps = room_3.solve(boundaries3)

            #u_knext3 = border_room_3[:,-1][1:-2] #get gamma
            #u_sol3 = omega*u_knext3 + (1-omega)*right_border_room_2[:,-1][1:-2] # relaxation, uses next iterate and previous
            comm.send(room_3_temps[:,0], dest=2) if i != iters-1 else comm.send(room_3_temps, dest=2) #send solution to room_2 core

        if rank==2: # room 2
            dir_room_1_border = 10*np.ones(height) if i == 0 else comm.recv(source=0)
            dir_room_3_border = np.flip(10*np.ones(height) if i == 0 else comm.recv(source=1))
            boundaries2 = [
                {"type": "dirichlet", "side": Side.LEFT, "start": height, "end": height*2, "values": dir_room_1_border}, #lower part left side
                {"type": "dirichlet", "side": Side.RIGHT, "start": height, "end": height*2, "values": dir_room_3_border}, #upper part right side
            ]
            room_2_prev=np.ones((height*2, width))*15

            if i > 0:
                room_2_prev=room_2_temps

            room_2_temps=room_2.solve(boundaries2)
            

            u_sol = omega*room_2_temps + (1-omega)*room_2_prev # relaxation, uses next iterate and previous
            
            comm.send((u_sol[height:,0]-u_sol[height:,1])/-h, dest=0) #lower left wall to room 1
            comm.send((u_sol[:height,-1] - u_sol[:height,-2])/-h, dest=1) #upper right wall to room 3

            if i == iters-1:
                room_1_temps = comm.recv(source=0)
                room_3_temps = comm.recv(source=1)
                plot_heatmap(room_1_temps, room_2_temps, room_3_temps)
                

if __name__ == "__main__":
    main()


