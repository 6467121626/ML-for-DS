import numpy as np

a = np.arange(6).reshape((2,3));

palette = np.array( [ [0,0,0],                # black
                       [255,0,0],              # red
                      [0,255,0],              # green
                      [0,0,255],              # blue
                      [255,255,255] ] )       # white
image = np.array( [ [ 0, 1, 2, 0 ],           # each value corresponds to a color in the palette
                    [ 0, 3, 4, 0 ]  ] )


b = np.array([[1,2,3],[4,5,6]])

c = np.array([[-1,-2,-3],[-4,-5,-6]])

p = np.array([[0,0,0,0.4,0.6],
              [0,0,0,0.6,0.4],
              [0.45,0.55,0,0,0],
              [0.6,0.1,0.3,0,0],
              [0.4,0.55,0.05,0,0]])

r = np.array([[2],
              [-2],
              [5],
              [10],
              [-10]])

print(np.dot(np.linalg.inv(np.eye(5) - 0.9*p),r))