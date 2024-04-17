import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import random 


def forwardKinematics(theta1, theta2):

  xForward = np.cos(theta1) * link1 + np.cos(theta1+theta2) * link2
  yForward = np.sin(theta1) * link1 + np.sin(theta1+theta2) * link2
  
  return xForward, yForward


def equations(thetas, desiredPos, lengths):

  theta1, theta2 = thetas
  link1, link2 = lengths

  xForward = np.cos(theta1) * link1 + np.cos(theta1+theta2) * link2
  yForward = np.sin(theta1) * link1 + np.sin(theta1+theta2) * link2
  
  return [xForward - desiredPos[0], yForward - desiredPos[1]]

 
desiredPos = [random.uniform(4,8), np.random.uniform(4,8)]



link1 = 1
link2 = 1
lengths = [link1,link2]




'''
desiredPos = [4,5]
initialGuess = [0.1,0.1]

JointAngles = fsolve(equations, initialGuess, args=(desiredPos, lengths))

print('Joint angles are: ', JointAngles)

pos1 = (1,1)
pos2 = (4,5)



# plt.plot(desiredPos)
# plt.plot(initialGuess)

plt.plot(pos1)
plt.plot(pos2)

plt.show()

'''

