from data_farm_constants import HOST  
import os 
from car_env.car_client import PiCarEnv 
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

fig, ax = plt.subplots(1) 

env = PiCarEnv(HOST, memory_length=200, memory_write_location=os.getcwd()+'/data_without_rl') 

i = 0 
no_kill = True 
while no_kill:
    i = i+1 
    print(i) 
    img, x, y, r = env.auto_action() 
    ## render
    ax.imshow(img)
    circ = None
    if r > 0.:
        circ = Circle((x,y), r, fill=False)
        ax.add_patch(circ)
        pass
    plt.pause(.1)
    if circ is not None:
        circ.remove()
        pass
    env.memorize() 
    pass


