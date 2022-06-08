import numpy as np
import matplotlib.pyplot as plt
from scanner import Scanner
from time import sleep

centers = [0,0]
spans = [-2*np.pi,-2*np.pi]
steps = [20,20]
snake = [1]
labels=['x','y']

def func(x,y):
    sleep(50E-3)
    return np.sin(x) * np.sin(y)

scan = Scanner(func,centers,spans,steps,snake,[],float,labels)

buffer = np.zeros(steps)
X,Y = np.meshgrid(*scan.positions)
fig = plt.figure()
imobj = plt.pcolormesh(X,Y,buffer,shading='auto')
plt.show(block=False)

def init(*args):
    print("Starting")

def progress(i,imax,idx,pos,res):
    print("                                                           ",end='\r')
    print(f"{i+1}/{imax}, {pos}",end='\r')
    if i+1 == imax:
        print()
    buffer[idx] = res
    if not (i+1)%10:
        imobj.set_array(buffer.ravel())
        imobj.autoscale()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

def finish(*args):
    print("Done")

scan._init_func = init
scan._prog_func = progress
scan._finish_func = finish
scan.run_async()