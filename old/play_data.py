import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

INDEX = [1,0,4,2,9,12,6]

# Reorder x and y according to their physical location
# X will be an array [1:n] where the numbers relate to location, not sensor
def reorder(y):
    y = np.array(y)[INDEX]
    x = range(len(y))
    return (x,y)


fig, ax = plt.subplots()


fd = open('data.csv', 'rb')
reader = csv.reader(fd, delimiter=',')
y = reader.next()
x,y = reorder(y)

line, = ax.plot(x, y)
ax.set_ylim(-0.006, 0.006) 

# Read the first 7000 lines
for i in range(7000):
    reader.next()


def animate(i):
    if i % 100==0:
        print "Line {i}".format(i=i+7000)
    y = reader.next()
    x,y = reorder(y)
    line.set_ydata(y)
    return line,


# Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

try:
    ani = animation.FuncAnimation(fig, animate, np.arange(1, 20000), init_func=init, interval=100, blit=False)
    plt.show()
finally:
    fd.close()