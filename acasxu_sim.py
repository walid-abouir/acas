# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:29:16 2024

@author: fperotto
"""

import math
import os
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from itertools import product as iterprod

import numpy as np

from scipy import ndimage
from scipy.linalg import expm

from matplotlib import patches, animation
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.lines import Line2D

#mpl.use('QT5Agg')


if __name__ == '__main__' :
    from aidge_nn import AidgeNN
    from dubins_nn import DubinsNN
    from lut_model import NearestLUT
    from lut_model import Encounter
    from lut_constants import get_act_idx, ACTION_NAMES, ACTION_CODES, ACT_IDX_COC
else:
    from acas import AidgeNN
    from acas import DubinsNN
    from acas import NearestLUT
    from acas.lut_model import Encounter
    from acas.lut_constants import get_act_idx, ACTION_NAMES, ACTION_CODES, ACT_IDX_COC


# 0: rho, distance (m) [0, 60760]
# 1: theta, angle to intruder relative to ownship heading (rad) [-pi,+pi]
# 2: psi, heading of intruder relative to ownship heading (rad) [-pi,+pi]
# 3: v_own, speed of ownship (ft/sec) [100, 1145? 1200?] 
# 4: v_int, speed in intruder (ft/sec) [0? 60?, 1145? 1200?] 


###############################################################################

def rad_mod(angle):
    """return angle between -pi and +pi"""
    return ( (angle + np.pi) % (np.pi*2) ) - np.pi

###############################################################################

def plot_best_actions_cartesian(psi, v_own, v_int, last_cmd=0, max_dist=20000, dist_incr=1000, model=None, fig=None, show=True):

    if fig is None:    
        fig, ax = plt.subplots(figsize=(8,8))
    else:
        ax = fig.gca()
    
    #own displacement in m in 10 seconds
    d_own = v_own/0.3281
    
    #int displacement in m in 10 seconds
    d_int = v_int/0.3281
    
    #intruder displacement
    # psi = 0 means intruder flying in the same sense than own
    dx = d_int * np.cos(psi)
    dy = d_int * np.sin(psi)
    
    # draw own speed
    ax.annotate('', xytext=(0,0), xy=(d_own,0), arrowprops=dict(arrowstyle='-|>'))

    # draw intruder speed
    ax.annotate('', xytext=(-dx,-dy), xy=(0,0), arrowprops=dict(arrowstyle='-|>', linestyle='--', alpha=0.3))
    
    
    colors = ['whitesmoke', 'lightgreen', 'lightblue', 'darkgreen', 'darkblue']
    
    if model is None:
        model = DubinsNN()

    x_values = range(-max_dist, +max_dist, dist_incr)
    y_values = range(-max_dist, +max_dist, dist_incr)
    coords_x_y = np.array(list(iterprod(x_values, y_values)))
    rho_theta = np.array([(np.hypot(x, y), np.arctan2(y, x)) for (x, y) in coords_x_y])
    A = np.array([model.predict(Encounter(last_a=last_cmd, v_own=v_own, v_int=v_int, theta=theta, psi=psi, rho=rho)).argmin() for (rho, theta) in rho_theta])
    C = np.array([colors[a] for a in A])    
    
#    #intruder position
#    for x in range(-max_dist, +max_dist, dist_incr):
#        for y in range(-max_dist, +max_dist, dist_incr):
#            
#            #rho = np.sqrt(x**2 + y**2)
#            rho = np.hypot(x, y)
#            theta = np.arctan2(y, x)
#            
#            action = model.predict(last_cmd, v_own, v_int, theta, psi, rho).argmin()
#    
#            #if action != 0:
#            # draw intruder position
#            ax.scatter(x, y, color=colors[action])  #, label=names[action])
#
#            # draw intruder speed
#            #if theta == 0:
#            #    ax.annotate('', xytext=(x,y), xy=(x+dx,y+dy), arrowprops=dict(arrowstyle='->', alpha=0.2))
    X = np.array(coords_x_y)[:,0]
    Y = np.array(coords_x_y)[:,1]
    ax.scatter(X, Y, color=C)  #, label=names[action])

    custom = [Line2D([], [], marker='.', markersize=20, c=color, linestyle='None') for color in colors]
    ax.legend(custom, ACTION_NAMES, loc='upper left')
    
    if show:
        plt.show()
            
    
###############################################################################


def plot_best_actions(psi, v_own, v_int, last_cmd=0, model=None, fig=None, show=True):

    if fig is None:    
        fig, ax = plt.subplots(figsize=(8,8))
    else:
        ax = fig.gca()
    
    #own displacement in m in 10 seconds
    d_own = v_own/0.3281
    
    #int displacement in m in 10 seconds
    d_int = v_int/0.3281
    
    #intruder displacement
    # psi = 0 means intruder flying in the same sense than own
    dx = d_int * np.cos(psi)
    dy = d_int * np.sin(psi)
    
    # draw own speed
    ax.annotate('', xytext=(0,0), xy=(d_own,0), arrowprops=dict(arrowstyle='-|>'))
    
    # draw intruder speed
    ax.annotate('', xytext=(-dx,-dy), xy=(0,0), arrowprops=dict(arrowstyle='-|>', linestyle='--', alpha=0.3))
    
    #41 angles (rad) from -pi to +pi, linearly disposed
    #theta_values = [-3.1416, -2.9845, -2.8274, -2.6704, -2.5133, -2.3562, -2.1991, -2.042, -1.885, -1.7279, -1.5708, -1.4137, -1.2566, -1.0996, -0.9425, -0.7854, -0.6283, -0.4712, -0.3142, -0.1571, 0.0, 0.1571, 0.3142, 0.4712, 0.6283, 0.7854, 0.9425, 1.0996, 1.2566, 1.4137, 1.5708, 1.7279, 1.885, 2.042, 2.1991, 2.3562, 2.5133, 2.6704, 2.8274, 2.9845, 3.1416]
    theta_values = np.linspace(-3.1416, 3.1416, 41)   #rounded at 4 dec positions
    
    #39 distances
    rho_values = [   499.,    800.,   2508.,   4516.,   6525.,   8534.,  10543.,  12551.,  14560.,  
      16569.,  18577.,  20586.,  22595.,  24603.,  26612.,  28621.,  30630.,  32638.,
      34647.,  36656.,  38664., 40673.,  42682.,  44690.,  46699. , 48708.,  50717.,
      52725.,  54734.,  56743.,  58751.,  60760.,  75950.,  94178., 112406., 130634.,
     148862., 167090., 185318.]
    
    colors = ['whitesmoke', 'lightgreen', 'lightblue', 'darkgreen', 'darkblue']
    names = ['clear-of-conflict', 'weak-left', 'weak-right', 'strong-left', 'strong-right']

    if model is None:
        model = DubinsNN()
    
    A = np.array([[model.predict(Encounter(last_a=last_cmd, v_own=v_own, v_int=v_int, theta=theta, psi=psi, rho=rho)).argmin() for rho in rho_values] for theta in theta_values]).flat
    X = np.array([[rho * math.cos(theta) for rho in rho_values] for theta in theta_values]).flat
    Y = np.array([[rho * math.sin(theta) for rho in rho_values] for theta in theta_values]).flat
    C = np.array([colors[a] for a in A])    
    
#    for rho in rho_values:
#        for theta in theta_values:
#            #intruder position
#            x = rho * math.cos(theta)
#            y = rho * math.sin(theta)
#    
#            action = model.predict(last_cmd, v_own, v_int, theta, psi, rho).argmin()
#    
#            #if action != 0:
#            # draw intruder position
#            ax.scatter(x, y, color=colors[action])  #, label=names[action])
#
#            # draw intruder speed
#            #if theta == 0:
#            #    ax.annotate('', xytext=(x,y), xy=(x+dx,y+dy), arrowprops=dict(arrowstyle='->', alpha=0.2))

    ax.scatter(X, Y, color=C)  #, label=names[action])

    custom = [Line2D([], [], marker='.', markersize=20, c=color, linestyle='None') for color in colors]
    ax.legend(custom, names, loc='upper left')
    
    
    if show:
        plt.show()
            
    
###############################################################################


def plot_state(encounter, fig=None, animate=False, frames=12, show=True):

    #own displacement in m in 10 seconds
    d_own = encounter.v_own/0.3281
    
    #int displacement in m in 10 seconds
    d_int = encounter.v_int/0.3281
    
    #intruder position
    x = encounter.rho * math.cos(encounter.theta)
    y = encounter.rho * math.sin(encounter.theta)
    
    #intruder displacement
    # psi = 0 means intruder flying in the same sense than own
    dx = d_int * np.cos(encounter.psi)
    dy = d_int * np.sin(encounter.psi)

    #lim = 2.3 * max(x, y)

    if fig is None:    
        fig, ax = plt.subplots()
    else:
        ax = fig.gca()
    
    #ax.set_xlim(-lim, lim)
    #ax.set_ylim(-lim, lim)

    # draw distance
    ax.annotate('', xytext=(x,y), xy=(0,0), arrowprops=dict(arrowstyle="-", linestyle="--", shrinkA=0, shrinkB=0, alpha=0.5))
    ax.annotate(f'$\\rho = {round(encounter.rho/1000,1)}$ Km', xy=(x//2 + encounter.v_own, y//2), xycoords='data')
    
    # draw own speed
    ax.annotate('', xytext=(0,0), xy=(d_own,0), arrowprops=dict(arrowstyle='->'))
    ax.text(encounter.v_own, -2*encounter.v_own, str(round(d_own/1000,1)) + "Km in 10s")
    # draw own linear trajectory
    ax.annotate('', xy=(12*encounter.v_own,0), xytext=(0,0), arrowprops=dict(arrowstyle="-", linestyle=":", color='b', shrinkA=0, shrinkB=0, alpha=0.5))
    # draw intruder speed
    ax.annotate('', xytext=(x,y), xy=(x+dx,y+dy), arrowprops=dict(arrowstyle='->'))
    # draw own position
    ax.scatter(0, 0)
    ax.scatter(d_own, 0, alpha=0)
    # draw intruder position
    ax.scatter(x, y)
    ax.scatter(x+dx, y+dy, alpha=0)
    
    # draw intruder angle
    r = encounter.v_int
    arc_angles = np.linspace(0, encounter.psi, 20)
    arc_xs = r * np.cos(arc_angles)
    arc_ys = r * np.sin(arc_angles)
    ax.plot(arc_xs+x, arc_ys+y, lw = 1, color='k', alpha=0.5)
    ax.annotate(f'$\\psi = {round(np.degrees(encounter.psi),1)}\\degree$', xy=(x+encounter.v_int, y+encounter.v_int//2), xycoords='data')
    
    # draw intruder reference own heading
    ax.annotate('', xy=(x+2*encounter.v_int,y), xytext=(x,y), arrowprops=dict(arrowstyle="-", linestyle=":", color='g', shrinkA=0, shrinkB=0, alpha=0.5))

    # draw theta angle and annotation
    r = encounter.v_own
    arc_angles = np.linspace(0, encounter.theta, 20)
    arc_xs = r * np.cos(arc_angles)
    arc_ys = r * np.sin(arc_angles)
    ax.plot(arc_xs, arc_ys, lw = 1, color='k', alpha=0.5)
    ax.annotate(f'$\\theta = {round(np.degrees(encounter.theta),1)}\\degree$', xy=(encounter.v_own, encounter.v_own//2), xycoords='data')
    #plt.gca().annotate('<----- r = 1.5 ---->', xy=(0 - 0.2, 0 + 0.2), xycoords='data', fontsize=15, rotation = 45)
    
    ax.set_aspect('equal')
    
    ax.set_title("displacement (interval = $10s$)")
    
    if animate:
       # draw own final position
       ax.scatter(frames*d_own, 0)
       # draw intruder final position
       ax.scatter(x+frames*dx, y+frames*dy)
    
       def animate(n):
          own_traj_coordinates = np.array([(0,0), ((n+1)*d_own, 0)])
          own_lines, = ax.plot(own_traj_coordinates[:,0], own_traj_coordinates[:,1], alpha=0.4, color='b')
          intruder_traj_coordinates = np.array([(x,y), (x+(n+1)*dx, y+(n+1)*dy)])
          intruder_lines, = ax.plot(intruder_traj_coordinates[:,0], intruder_traj_coordinates[:,1], alpha=0.4, color='g')
          return own_lines, intruder_lines
       
       anim = FuncAnimation(fig, animate, frames=frames, interval=500, repeat=False, blit=True)
    
    if show:
        plt.show()
    
    if animate:
        return anim
    else:
        return None
    

###############################################################################

class Airplane():
    
    def __init__(self, x=0.0, y=0.0, head=0.0, speed=1000.0, name='airplane'):
        self.x = x  
        self.y = y   
        self.head = head  # in rad
        self.speed = speed  #distance per unit of time (on heading direction)
        self.name=name
   
    def __str__(self):
        return f"x: {self.x}, y: {self.y}, head: {self.head}, speed: {self.speed}"
    


###############################################################################
    
class AcasEnv():

    # todo: make this a parameter
    #nn_update_rate = 1.0  #decision time interval
    #dt = 1.0   #time interval of one step in the simulation (can be smaller than the decision time interval)

    #self.vec = step_state(self.vec, self.v_own, self.v_int, time_elapse_mat, State.dt)
    #act_to_angle = [0, np.radians(1.5), np.radians(-1.5), np.radians(3.), np.radians(-3.)]
    act_to_angle = [0, np.radians(-1.5), np.radians(+1.5), np.radians(-3.), np.radians(+3.)]


    def __init__(self, 
                 #render_mode: str=None,
                 #x_int, y_int, head_int, v_int=500,          # in m, m, rad, ft/sec
                 #x_own=0., y_own=0., head_own=0., v_own=800,  # in m, m, rad, ft/sec
                 airplanes = [Airplane(name='own'), Airplane(x=10000.0, y=5000.0, head=-3.0, name='intruder')],
                 save_states=False,
                 #decision_freq=1.0,     # in s
                 #update_freq=0.1,       # in s
                 #init_command=0
                 ):      # initial command is COC


        #self.t = 0.0
        
        #x1, y1, _theta1, x2, y2, _theta2, _
        #self.vec = np.array(init_vec, dtype=float) # current state

        self.airplanes = airplanes
        
        #self.x_int=x_int
        #self.y_int=y_int
        #self.head_int=head_int

        #self.x_own=x_own
        #self.y_own=y_own 
        #self.head_own=head_own

        #self.command = init_command 
        
        #self.v_own = v_own
        #self.v_int = v_int

        #self.d2 = (x_own - x_int)**2 + (y_own - y_int)**2
        #self.rho = np.sqrt((x_own - x_int)**2 + (y_own - y_int)**2)
        #self.rho = np.sqrt((airplanes[0].x - airplanes[1].x)**2 + (airplanes[0].y - airplanes[1].y)**2)


        #self.next_nn_update = 0
        
        # these are set when simulation() if save_states=True
        self.save_states = save_states
        self.states_history = [] # state history
        self.commands_history = [] # commands history

        # assigned by simulate()
        #self.u_list = []  #intruder commands
        #self.u_list_index = None
        self.min_dist = np.inf
        
        #self.reset()


    def __str__(self):
        return "\n".join([f"x: {airplane.x}, y: {airplane.y}, head: {airplane.head_rad}" for airplane in self.airplanes])


    def update_relations(self):

        #could be a triangular matrix representation
        self.relative_distances = np.array([[np.hypot(own.x-intruder.x, own.y-intruder.y) for intruder in self.airplanes] for own in self.airplanes])
        for i in range(len(self.airplanes)):
            self.relative_distances[i,i] = np.inf
        self.nearest_intruder_index = self.relative_distances.argmin(axis=-1)
        #self.nearest_intruder_index = np.argmin(self.relative_distances > 0.0, axis=-1)
        self.relative_angles =    np.array([[rad_mod(np.arctan2(intruder.y-own.y, intruder.x-own.x)) for intruder in self.airplanes] for own in self.airplanes])
        self.relative_heads =     np.array([[rad_mod(intruder.head-own.head) for intruder in self.airplanes] for own in self.airplanes])
        self.rho = np.array([self.relative_distances[i, self.nearest_intruder_index[i]] for i in range(len(self.airplanes))])
        self.theta = np.array([self.relative_angles[i, self.nearest_intruder_index[i]] for i in range(len(self.airplanes))])
        self.psi = np.array([self.relative_heads[i, self.nearest_intruder_index[i]] for i in range(len(self.airplanes))])
        self.v_int = np.array([self.airplanes[self.nearest_intruder_index[i]].speed for i in range(len(self.airplanes))])
        self.min_dist = self.rho.min()
        #self.min_dist.min()


    def get_observation(self):
        #return [[self.rho[i], self.theta[i], self.psi[i], own.speed, self.v_int[i]] for i, own in enumerate(self.airplanes)]
        return [Encounter(rho=self.rho[i], theta=self.theta[i], psi=self.psi[i], v_own=own.speed, v_int=self.v_int[i]) for i, own in enumerate(self.airplanes)]
        
        
    #def reset(self, *, seed:int=None, initial_state=None, options:dict=None) -> tuple:
    def reset(self):

        """
        Resets the environement to an initial state. required before calling the step() function and make actions. Returns the airplane observation, 
        for an episode and information (the distance between the two)
        """
        
        self.update_relations()
        
        #own = self.airplanes[0]
        #self.rho = min([np.sqrt((own.x - intruder.x)**2 + (own.y - intruder.y)**2) for intruder in self.airplanes[1:]])
        #self.rho = min([np.hypoten((own.x - intruder.x), (own.y - intruder.y)) for intruder in self.airplanes[1:]])
        #self.min_dist = self.rho

        self.states_history = [] # state history
        self.commands_history = [] # commands history
        self.dist_history = [] # rho history

        return self.get_observation()

        
    def step(self, actions):
        """
        Make an action in the environnement. The action is chosen by the policy. 
        @type action : 
        @return : observation, truncated, terminated, info
        @rtype : obsType, Float, bool, bool, dict[str, Any]
        The observation space is a list of : rho, ownship's speed, intruder's speed, ownship's angle, Intruder's angle, minimum distance 
        between the two airplanes. 
        """

        #self.command = action
        #intruder_cmd = self.u_list[self.u_list_index]

        if self.save_states:
            #self.commands.append(self.command)
            #self.int_commands.append(intruder_cmd)
            self.commands_history.append(actions)
            self.states_history.append([[airplane.x, airplane.y, airplane.head, airplane.speed] for airplane in self.airplanes])
            self.dist_history.append(self.min_dist)

        #time_elapse_mat = State.time_elapse_mats[self.command][intruder_cmd] #get_time_elapse_mat(self.command, State.dt, intruder_cmd)

        for i, own in enumerate(self.airplanes):
            own.head = rad_mod(own.head + self.act_to_angle[actions[i]])
            own.x += np.cos(own.head) * own.speed
            own.y += np.sin(own.head) * own.speed

        self.update_relations()
        
            
        #cur_dist_sq = (self.vec[0] - self.vec[3])**2 + (self.vec[1] - self.vec[4])**2
        #self.rho = np.sqrt((self.x_own - self.x_int)**2 + (self.y_own - self.y_int)**2)
        #own = self.airplanes[0]
        #rho = min([np.sqrt((own.x - intruder.x)**2 + (own.y - intruder.y)**2) for intruder in self.airplanes[1:]])

        #rho = self.relative_distances[self.nearest_intruder_index]
        #theta = self.relative_angles[self.nearest_intruder_index]
        #self.psi = self.relative_heads[self.nearest_intruder_index]
        #self.v_int = self.airplanes[self.nearest_intruder_index].speed
        
        return self.get_observation()
        #return observation, self.r, self.terminated, self.truncated, info

    
    def simulate(self, tmax=200):
        '''simulate system

        saves result in self.vec_list
        also saves self.min_dist
        '''

        #assert isinstance(cmd_list, list)
        #self.int_commands = cmd_list
        #self.u_list = cmd_list  #intruder commands
        #self.u_list_index = None
        #tmax = len(cmd_list) * State.nn_update_rate
        #tmax = min(tmax, len(cmd_list))

        t = 0

        if self.save_states:
            #rv = [self.vec.copy()]
            #rv = [ [self.x_own, self.y_own, self.head_own, self.x_int, self.y_int, self.head_int] ]
            rv = [[[a.x, a.y, a.head_rad] for a in self.airplanes]]

        self.min_dist = self.rho
        #self.min_dist = 0, math.sqrt((self.vec[0] - self.vec[3])**2 + (self.vec[1] - self.vec[4])**2), self.vec.copy()
        #prev_dist_sq = (self.vec[0] - self.vec[3])**2 + (self.vec[1] - self.vec[4])**2
        #prev_dist = self.rho

        for t in range(1, tmax):
        #while t + 1e-6 < tmax:
            
            self.step()

            if self.save_states:
                #rv.append(self.vec.copy())
                rv.append([[a.x, a.y, a.head_rad] for a in self.airplanes])

            #t += State.dt
            #t += 1

            #if cur_dist_sq > prev_dist_sq:
            #    #print(f"Distance was increasing at time {round(t, 2)}, stopping simulation. Min_dist: {round(prev_dist, 1)}ft")
            #    break

            #prev_dist_sq = cur_dist_sq
            self.min_dist = min(self.min_dist, self.rho)
            
        ##because break when distance increases, min distance is the last one
        #self.min_dist = math.sqrt(prev_dist_sq)

        if self.save_states:
            self.states_history = rv

        #if not self.save_states:
        #    assert not self.vec_list
        #    assert not self.commands
        #    assert not self.int_commands

    
    # def update_command(self):
    #     'update command based on current state'''


    #     if self.rho > 60760:  # distance > 60Km
    #         self.command = 0
    #     else:
    #         last_command = self.command

    #         net = State.nets[last_command]

    #         state = [rho, theta, psi, v_own, v_int]

    #         res = run_network(net, state)
    #         self.command = np.argmin(res)

    #         #names = ['clear-of-conflict', 'weak-left', 'weak-right', 'strong-left', 'strong-right']

    #     if self.u_list_index is None:
    #         self.u_list_index = 0
    #     else:
    #         self.u_list_index += 1

    #         # repeat last command if no more commands
    #         self.u_list_index = min(self.u_list_index, len(self.u_list) - 1)


#################################################

class AbstractAgent():

    def __init__(self, name:str=None):
        self.name = name

    def get_name(self):
        if self.name is not None:
            return self.name
        else:
            return "Agent"
            
    def __str__(self):
        return self.get_name()

#################################################

class RandomAgent(AbstractAgent):

    def __init__(self, default_seed=None, name:str="Random"):
        super().__init__(name)
        self.default_seed = default_seed
        self.rndgen = np.random.default_rng(default_seed)
        
    def reset(self, obs):
        return self.rndgen.choice(range(5))
        
    def react(self, obs):
        return self.rndgen.choice(range(5))

#################################################

class ConstantAgent(AbstractAgent):

    def __init__(self, command=0, name:str=None):

        super().__init__(name)

        self.command = get_act_idx(command)
        if self.command is None:
            self.command = ACT_IDX_COC

        if self.name is None:
            self.name = "Constant-" + ACTION_CODES[self.command]

    def reset(self, obs):
        return self.command
        
    def react(self, obs):
        return self.command
    

#################################################
    
class ListAgent(AbstractAgent):
    
    def __init__(self, commands=[0,0,0,0,0,1,1], mode='cycle', name:str="Fixed-Behavior-Agent"):
        super().__init__(name)
        self.commands = commands
        self.mode = mode
        self.t = None
        
    def reset(self, obs):
        self.t = 0
        return self.commands[self.t]   
        
    def react(self, obs):
        self.t += 1
        if self.mode=='cycle':
            return self.commands[self.t % len(self.commands)]
        else:
            return self.commands[max(self.t, len(self.commands)-1)]
        
    
#################################################

class UtilityModelAgent(AbstractAgent):

    def __init__(self, name:str="Utility-Agent"):
        super().__init__(name)
        self.model = None
        self.obs = None
        
    def reset(self, obs):
        self.obs = obs
        self.action = 0     #default initial action COC
        return self.action   
        
    def react(self, obs):
        self.obs = obs
        #self.obs = [rho, theta, psi, v_own, v_int]
        ### v_own, v_int, theta, psi, rho = self.obs
        #rho, theta, psi, v_own, v_int = self.obs
        if obs.rho > 60760:  # distance > 60Km
            self.action = 0
        else:
            values = self.model.predict(obs)
            self.action = np.argmin(values)
        return self.action   


class LutAgent(UtilityModelAgent):
    
    def __init__(self, name:str="LUT-Agent"):
        super().__init__(name)
        self.model = NearestLUT()
        self.obs = None
        
    
class DubinsAgent(UtilityModelAgent):
    
    def __init__(self, name:str="Dubins-ONNX-Agent"):
        super().__init__(name)
        self.model = DubinsNN()
        self.obs = None
        
class AidgeAgent(UtilityModelAgent):
    
    def __init__(self, name:str="AIDGE-Dubins-Agent"):
        super().__init__(name)
        self.model = AidgeNN()
        self.obs = None

        
#rho, theta, psi, v_own, v_int = state7_to_state5(self.vec, self.v_own, self.v_int)

# 0: rho, distance
# 1: theta, angle to intruder relative to ownship heading
# 2: psi, heading of intruder relative to ownship heading
# 3: v_own, speed of ownship
# 4: v_int, speed in intruder

# min inputs: 0, -3.1415, -3.1415, 100, 0
# max inputs: 60760, 3.1415, 3,1415, 1200, 1200

#'Valid range" [100, 1145]'
#v_own = 800 # ft/sec

#'Valid range: [60, 1145]'
#v_int = 500

        
#################################################

class AcasRender():

    def __init__(self, envs):

       self.envs = envs
       #verify that envs is a list

       if not isinstance(self.envs, list): 
           self.envs = [self.envs]
           
       airplane_img_filepath = os.path.dirname(os.path.realpath(__file__)) + '/img/airplane.png'
       self.img = plt.imread(airplane_img_filepath)

       self.anim = None


    def plot(self, fig=None, interval=10, show=True, save_mp4=False, title="ACAS Xu Simulation"):
 
        #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
        if fig is None:
            fig, axes = plt.subplots(len(self.envs))
            if len(self.envs) == 1:
                axes = [axes]
            fig.tight_layout()
        else:
            axes = [fig.gca()]
        
        for env, ax in zip(self.envs, axes):
            
            ax.set_aspect('equal')
            #ax.axis('equal')
        
            ax.set_title(title)
            ax.set_xlabel('X Position (ft)')
            ax.set_ylabel('Y Position (ft)')
        
            palette = np.array(['gray', 'green', 'blue', 'magenta', 'red'])
            custom_lines = [Line2D([0], [0], color=c, lw=1) for c in palette]
            ax.legend(custom_lines, ACTION_NAMES, fontsize=12)#, loc='lower left')
    
            num_airplanes = len(env.airplanes)
            total_time = len(env.states_history)
            H = np.array(env.states_history)
            A = np.array(env.commands_history)
            C = palette[A]
    
            time_text = ax.text(0.02, 0.98, 'Time: 0 s', horizontalalignment='left', fontsize=12, verticalalignment='top', transform=ax.transAxes)
            time_text.set_visible(True)
            
            lines = []
            lcs = []
            figs = []
    
            #lc = LineCollection([], lw=2, animated=True, color='k', zorder=1)
            #ax.add_collection(lc)
            
            for i in range(num_airplanes):
                #initial state
                x, y, theta, speed = H[0,i]
                x_f, y_f, theta_f, speed_f = H[-1,i]
                #total displacement
                d = speed * total_time
                dx = d * np.cos(theta)
                dy = d * np.sin(theta)
                # draw linear initial trajectory
                ax.scatter([x, x_f], [y, y_f], zorder=3, alpha=0.3)
                ax.plot([x, x+dx], [y, y+dy], ls=':', lw=1, color='lightgray', zorder=2)
                l = ax.plot(x, y, ls='-', lw=2, zorder=1)[0]
                #l = Line2D(x, y, lc='c', lw=1, ls='-', zorder=1)
                lines.append(l)
                lc = LineCollection([], linestyle='solid', lw=2, animated=True, color='k', zorder=1)
                ax.add_collection(lc)
                lcs.append(lc)
                #planes
                size=1500 #5000
                box = Bbox.from_bounds(x - size/2, y - size/2, size, size)
                tbox = TransformedBbox(box, ax.transData)
                box_image = BboxImage(tbox, zorder=2)
                theta_deg = (theta - np.pi / 2) / np.pi * 180 # original image is facing up, not right
                img_rotated = ndimage.rotate(self.img, theta_deg, order=1)
                box_image.set_data(img_rotated)
                figs.append(box_image)
                ax.add_artist(box_image)
                
            artists = figs + lines + lcs + [time_text]

        def init():
            time_text.set_text('Time: 0 s')
            for i, lc in enumerate(lcs):
                lc.get_paths().clear()
            return artists
        
        def animate(n):
            time_text.set_text(f'Time: {n} s')
            #for i, l in enumerate(lines):
            for i, lc in enumerate(lcs):
                #airplane image
                fig=figs[i]
                #trajectory using lines
                #l.set_xdata(H[:n+1,i,0])
                #l.set_ydata(H[:n+1,i,1])
                #trajectory using paths
                paths = lc.get_paths()
                paths.append(Path([H[n,i,:2], H[n+1,i,:2]]))
                lc.set_colors(C[:n+1,i])
                x = H[n+1,i,0]
                y = H[n+1,i,1]
                theta = H[n+1,i,2]
                #airplane figure
                theta_deg = (theta - np.pi / 2) / np.pi * 180 # original image is facing up, not right
                original_size = list(self.img.shape)
                img_rotated = ndimage.rotate(self.img, theta_deg, order=1)
                rotated_size = list(img_rotated.shape)
                ratios = [r / o for r, o in zip(rotated_size, original_size)]
                fig.set_data(img_rotated)
                size = 1500 #5000
                width = size * ratios[0]
                height = size * ratios[1]
                box = Bbox.from_bounds(x - width/2, y - height/2, width, height)
                tbox = TransformedBbox(box, ax.transData)
                fig.bbox = tbox
            return artists
        
        self.anim = FuncAnimation(fig, animate, init_func=init, frames=total_time-1, interval=interval, blit=True, repeat=True, repeat_delay=100)
        
        def on_click(event):
           if self.anim is not None:
              self.anim.pause()
           plt.close(fig)


        #fig.canvas.mpl_connect('button_press_event', on_click)
        
#        def on_close(event):
#            plt.close()
#            if anim is not None:
#                #self.my_anim.pause()
#                anim = None    #setting to None allows garbage collection and closes loop
#                del(anim)
#        fig.canvas.mpl_connect('close_event', on_close)
        
        if save_mp4:
            writer = animation.writers['ffmpeg'](fps=1000//interval, bitrate=1800)
            self.anim.save('sim.mp4', writer=writer)
        
        if show:
            plt.show()
        
        self.anim = None

#################################################
  
def run_single_sim(total_time=80, 
                   airplanes = [Airplane(x=0.0, y=0.0, head=0.0, speed=1080.0), 
                                Airplane(x=25000.0, y=50000.0, head=-np.pi/3, speed=1000.0)],
                   agents=['dubins', 'dubins'],
                   ):
   
   env  = AcasEnv(airplanes=airplanes, save_states=True)
   
   agents = consolidate_agents(agents)

   obs = env.reset()
   actions = [agent.reset(obs[i]) for i, agent in enumerate(agents)]

   print("====================================================================")
   print("RUNNING SIMULATION:")
   print("--------------------------------------------------------------------")
   print(f"rho (distance) = {env.rho[0]} ft")
   print(f"theta (relative position angle) = {env.theta[0]} rad , = {env.theta[0]/np.pi} pi rad , = {np.degrees(env.theta[0])} deg")
   print(f"psi (relative intrusion angle) = {env.psi[0]} rad , = {env.psi[0]/np.pi} pi rad , = {np.degrees(env.psi[0])} deg)")
   print(f"v_own (own speed) = {env.airplanes[0].speed} ft/s")
   print(f"v_int (intruder speed) = {env.airplanes[1].speed} ft/s.")
   print(f"coords_own (x, y) = ({env.airplanes[0].x},{env.airplanes[0].y})")
   print(f"heading_own (own absolute angle) = {env.airplanes[0].head} rad , = {env.airplanes[0].head/np.pi} pi rad , = {np.degrees(env.airplanes[0].head)} deg)")
   print(f"coords_int (x, y) = ({env.airplanes[1].x},{env.airplanes[1].y})")
   print(f"heading_int (intruder absolute angle) = {env.airplanes[1].head} rad , = {env.airplanes[1].head/np.pi} pi rad , = {np.degrees(env.airplanes[1].head)} deg)")
   print("====================================================================")

   for t in range(total_time):
       obs = env.step(actions=actions)
       actions = [m.react(obs[i]) for i, m in enumerate(agents)]

   fig, ax = plt.subplots(figsize=(8,8))
   renderer = AcasRender(env)
   renderer.plot(fig=fig)

###############################################################################

def consolidate_agents(agents):

   consolidated_agents = []
   
   for i, a in enumerate(agents):
      if get_act_idx(a) is not None:
         consolidated_agents.append(ConstantAgent(command=a))
      elif a in ['dubins', 'DUBINS', 'Dubins', 'onnx', 'ONNX']:
         consolidated_agents.append(DubinsAgent())
      elif a in ['aidge', 'AIDGE', 'Aidge']:
         consolidated_agents.append(AidgeAgent())
      elif a in ['lut', 'LUT']:
         consolidated_agents.append(LutAgent())
      elif a in ['random', 'rnd', 'rand', 'RANDOM', 'RAND', 'RND']:
         consolidated_agents.append(RandomAgent())
      else:
         consolidated_agents.append(a)
         
   return consolidated_agents

###############################################################################

def smart_random_run(seed=None, min_d=0, max_d=5000, interest_time=60, total_time=120, agents=[['dubins', 'coc']], plot=True, plot_same_figure=True, save_mp4=False, verbose=False):
   
   #verify that agents is a list of lists
   if len(np.array(agents).shape) == 1: 
       agents = [agents]
       
   envs = []
   
   if plot and plot_same_figure:
       n = len(agents)
       fig, axes = plt.subplots(n, figsize=(10,10))
       if n == 1:
           axes = [axes]
   
   for i, list_agents in enumerate(agents):
       
       if seed is not None:
           np.random.seed(seed)
            
       airplanes = [] 

       #retrogradation de la position par rapport à la collision. 
       head = np.random.uniform(-np.pi, +np.pi)
       speed = np.random.uniform(100, 1200)
       x_t = 0
       y_t = 0
       x_0 = x_t - (speed * interest_time * np.cos(head))
       y_0 = y_t - (speed * interest_time * np.sin(head))
        
       airplanes.append(Airplane(x=x_0, y=y_0, head=head, speed=speed))
        
       for j in range(1, len(list_agents)):                 
         
            d = np.random.uniform(min_d, max_d) #distance at interest_time
            theta = np.random.uniform(-np.pi, +np.pi) #intruder relative angle at interest_time
         
            head = np.random.uniform(-np.pi, +np.pi)
            speed = np.random.uniform(100, 1200)
            x_t = d * np.cos(theta)
            y_t = d * np.sin(theta)
            x_0 = x_t - (speed * interest_time * np.cos(head))
            y_0 = y_t - (speed * interest_time * np.sin(head))
         
            airplanes.append(Airplane(x=x_0, y=y_0, head=head, speed=speed))

       list_agents = consolidate_agents(list_agents)
        
       env = AcasEnv(airplanes=airplanes, save_states=True)
    
       obs = env.reset()
       actions = [m.reset(obs[j]) for j, m in enumerate(list_agents)]
    
       if verbose:
           print("RUNNING SIMULATION:")
           print(f"rho (distance) = {env.rho[0]} ft")
           print(f"theta (relative position angle) = {env.theta[0]} rad , = {env.theta[0]/np.pi} pi rad , = {np.degrees(env.theta[0])} deg")
           print(f"psi (relative intrusion angle) = {env.psi[0]} rad , = {env.psi[0]/np.pi} pi rad , = {np.degrees(env.psi[0])} deg)")
           print(f"v_own (own speed) = {env.airplanes[0].speed} ft/s")
           print(f"v_int (intruder speed) = {env.airplanes[1].speed} ft/s.")
           print(f"coords_own (x, y) = ({env.airplanes[0].x},{env.airplanes[0].y})")
           print(f"heading_own (own absolute angle) = {env.airplanes[0].head} rad , = {env.airplanes[0].head/np.pi} pi rad , = {np.degrees(env.airplanes[0].head)} deg)")
           print(f"coords_int (x, y) = ({env.airplanes[1].x},{env.airplanes[1].y})")
           print(f"heading_int (intruder absolute angle) = {env.airplanes[1].head} rad , = {env.airplanes[1].head/np.pi} pi rad , = {np.degrees(env.airplanes[1].head)} deg)")
    
       for t in range(total_time):
           obs = env.step(actions=actions)
           actions = [m.react(obs[j]) for j, m in enumerate(list_agents)]
    
       if plot:
           if plot_same_figure:
               ax = axes[i]
           else:
               fig, ax = plt.subplots(figsize=(8,8))
           renderer = AcasRender(env)
           title = "ACAS Xu Simulation - " + " / ".join(agent.get_name() for agent in list_agents)
           renderer.plot(fig=fig, title=title, show = not plot_same_figure)
       
       envs.append(env) 
 
       if plot and plot_same_figure:
           plt.show(block=True)

   return envs
    

###############################################################################

def random_run(seed=None, intruder_can_turn=False, total_time=120, save_mp4=False):

   if seed is not None:
       np.random.seed(seed)
       
   
   airplanes = [Airplane(x=0.0, y=0.0, head=0.0, speed=1080.0), 
                Airplane(x=np.random.uniform(-150000, +150000), 
                         y=np.random.uniform(-150000, +150000), 
                         head=np.random.uniform(-np.pi, +np.pi), 
                         speed=np.random.uniform(0, 1200))]
   
   env  = AcasEnv(airplanes=airplanes, save_states=True)

   agents=[ DubinsAgent(), ConstantAgent(command=0)] #, DubinsAgent()]
   #agents=[ LutAgent(), ConstantAgent(command=0)] #, DubinsAgent()]
   #agents=[DubinsAgent(), DubinsAgent()] #, DubinsAgent()]
   #agents=[DubinsAgent(), ConstantAgent()]
   #agents=[DubinsAgent(), RandomAgent()]
   #agents=[LutAgent(), RandomAgent()]
   #agents=[ConstantAgent(command=0), ConstantAgent(command=1)]
   #agents=[DubinsAgent(), ConstantAgent(command=2)]

   obs = env.reset()
   actions = [m.reset(obs[i]) for i, m in enumerate(agents)]

   print("RUNNING SIMULATION:")
   print(f"rho (distance) = {env.rho[0]} ft")
   print(f"theta (relative position angle) = {env.theta[0]} rad , = {env.theta[0]/np.pi} pi rad , = {np.degrees(env.theta[0])} deg")
   print(f"psi (relative intrusion angle) = {env.psi[0]} rad , = {env.psi[0]/np.pi} pi rad , = {np.degrees(env.psi[0])} deg)")
   print(f"v_own (own speed) = {env.airplanes[0].speed} ft/s")
   print(f"v_int (intruder speed) = {env.airplanes[1].speed} ft/s.")
   print(f"coords_own (x, y) = ({env.airplanes[0].x},{env.airplanes[0].y})")
   print(f"heading_own (own absolute angle) = {env.airplanes[0].head} rad , = {env.airplanes[0].head/np.pi} pi rad , = {np.degrees(env.airplanes[0].head)} deg)")
   print(f"coords_int (x, y) = ({env.airplanes[1].x},{env.airplanes[1].y})")
   print(f"heading_int (intruder absolute angle) = {env.airplanes[1].head} rad , = {env.airplanes[1].head/np.pi} pi rad , = {np.degrees(env.airplanes[1].head)} deg)")

   for t in range(total_time):
       obs = env.step(actions=actions)
       actions = [m.react(obs[i]) for i, m in enumerate(agents)]

   fig, ax = plt.subplots(figsize=(8,8))
   renderer = AcasRender(env)
   renderer.plot(fig=fig)
   
###############################################################################
   
def multiple_smart_random_runs(num_sims=5, seed=None, min_d=0, max_d=5000, interest_time=60, total_time=120, agents=['dubins','coc'], plot=False, save_mp4=False):

    #interesting_seed = -1
    #interesting_state = None

    #num_sims = 10000
    # with 10000 sims, seed 671 has min_dist 4254.5ft

    #start = time.perf_counter()

    agents = consolidate_agents(agents)

    min_d_evolution = [] 
    
    seed_i = None
    for i in range(num_sims):
    #for seed in range(num_sims):
    #    if seed % 1000 == 0:
    #        print(f"{(seed//1000) % 10}", end='', flush=True)
    #    elif seed % 100 == 0:
    #        print(".", end='', flush=True)
    
        if seed is not None:
            seed_i = seed+i

        envs = smart_random_run( seed=seed_i, min_d=min_d, max_d=max_d, interest_time=interest_time, total_time=total_time, agents=agents, plot=plot, save_mp4=save_mp4)
        
        min_d_evolution.append(envs[0].dist_history)
        

        #init_vec, cmd_list, init_velo = make_random_input(seed, intruder_can_turn=intruder_can_turn)
        #
        #v_own = init_velo[0]
        #v_int = init_velo[1]
        #
        ## reject start states where initial command is not clear-of-conflict
        #state5 = state7_to_state5(init_vec, v_own, v_int)
        #
        #if state5[0] > 60760:
        #    command = 0 # rho exceeds network limit
        #else:
        #    res = run_network(State.nets[0], state5)
        #    command = np.argmin(res)
        #
        #if command != 0:
        #    continue
        #
        ## run the simulation
        #s = State(init_vec, v_own, v_int, save_states=False)
        #s.simulate(cmd_list)
        #
        ## save most interesting state based on some criteria
        #if interesting_state is None or s.min_dist < interesting_state.min_dist:
        #    interesting_seed = seed
        #    interesting_state = s

    return min_d_evolution
   
###############################################################################   

if __name__ == '__main__' :
   
   #print(mpl.get_backend())
   
   run_single_sim()
   
   random_run()
   
   smart_random_run(plot_same_figure=False)
   
   num_sims=30
   
   seed=3
   
   min_d_evolution = multiple_smart_random_runs(seed=seed, num_sims=num_sims, max_d=0, agents=['dubins', 'coc'], plot=False)
   fig, ax = plt.subplots(figsize=(8,8))
   for d_evo in min_d_evolution:
       plt.plot(d_evo)
   plt.grid()
   plt.show(block=True)

   min_d_evolution = multiple_smart_random_runs(seed=seed, num_sims=num_sims, max_d=0, agents=['lut', 'coc'], plot=False)
   fig, ax = plt.subplots(figsize=(8,8))
   for d_evo in min_d_evolution:
       plt.plot(d_evo)
   plt.grid()
   plt.show()
   
   