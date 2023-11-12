from car_env import car_client
from data_farm_constants import HOST 
from matplotlib import pyplot as plt
from matplotlib.patches import Circle 

fig, ax = plt.subplots(1) 

while True: 
    ## get image 
    img, x, y, r = car_client.img(HOST)  
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
    pass 

