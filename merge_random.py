import os
import sys
import optparse
import numpy as np

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary 
import traci
import pandas as pd
from DQN import Agent


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    opt_parser.add_option("--testing", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options

def get_state():
    v_0_x = traci.vehicle.getPosition("v_0")[0]
    v_0_y = traci.vehicle.getPosition("v_0")[1]
    v_1_x = traci.vehicle.getPosition("v_1")[0]
    v_1_y = traci.vehicle.getPosition("v_1")[1]
    v_2_x = traci.vehicle.getPosition("v_2")[0]
    v_2_y = traci.vehicle.getPosition("v_2")[1]
    v_3_x = traci.vehicle.getPosition("v_3")[0]
    v_3_y = traci.vehicle.getPosition("v_3")[1]
    v_4_x = traci.vehicle.getPosition("v_4")[0]
    v_4_y = traci.vehicle.getPosition("v_4")[1]
    v_5_x = traci.vehicle.getPosition("v_5")[0]
    v_5_y = traci.vehicle.getPosition("v_5")[1]
    v_0_v = traci.vehicle.getSpeed("v_0")
    v_1_v = traci.vehicle.getSpeed("v_1")
    v_2_v = traci.vehicle.getSpeed("v_2")
    v_3_v = traci.vehicle.getSpeed("v_3")
    v_4_v = traci.vehicle.getSpeed("v_4")
    v_5_v = traci.vehicle.getSpeed("v_5")
    return [v_0_x, v_0_y, v_1_x, v_1_y, \
            v_2_x, v_2_y, v_3_x, v_3_y, \
            v_4_x, v_4_y, v_5_x, v_5_y, \
            v_0_v, v_1_v, v_2_v, v_3_v, \
            v_4_v, v_5_v]

def distance(state, id1, id2):
    return np.sqrt((state[2*id1] - state[2*id2])**2 + (state[2*id1 + 1] - state[2*id2 + 1])**2)

def distance2closest(state):
    distances = [distance(state, 0, 1), \
                 distance(state, 0, 2), \
                 distance(state, 0, 3), \
                 distance(state, 0, 4), \
                 distance(state, 0, 5)]
    distances.sort()
    return distances[0], distances[1]


def reward(state, action, minGap = 10):
    r = 0
    if action != 1:
        r -= 1
    if traci.simulation.getCollidingVehiclesIDList() != ():
        r -= 100
        return r
    if traci.simulation.getMinExpectedNumber() == 6:
        firstClosest, secondClosest = distance2closest(state)
        if max(firstClosest, secondClosest) < minGap:
            r -= 30
        elif min(firstClosest, secondClosest) < minGap:
            r -= 15
        return r

# def safe_speed_change(state, id1, id2):
#     speed_change = np.random.randn()
#     v_name1 = 'v_' + str(id1)
#     new_speed = state[id1 + 8] + speed_change
    
    
#     while new_speed > state[id2 + 8]:
#         speed_change = 1 + np.random.randn()
#         new_speed = state[id1 + 8] + speed_change
#     traci.vehicle.setSpeed(v_name1, new_speed)
#     # if distance(state, id1, id2) > 2*(np.abs(state[id2 + 8] - state[id1 + 8])):
#     #     new_speed = state[id1 + 8] + np.abs(state[id2 + 8] - state[id1 + 8])*0.2
#     #     traci.vehicle.setSpeed(v_name1, new_speed)
    
         

def move(state, action):
    speed = traci.vehicle.getSpeed("v_0") + 1*(action - 1)
    traci.vehicle.setSpeed("v_0", min(max(5, speed), 24))
    traci.simulationStep()
    done = traci.simulation.getMinExpectedNumber() < 6
    if not done:
        nstate = get_state()
        r = reward(state, action)
    else:
        nstate = np.zeros((1, 18))
        r = 0
        # traci.close()
        # sys.stdout.flush()
    # print(traci.simulation.getMinExpectedNumber())
    
    return nstate, r, done

def run(training = True):
    info = []
    done = False
    step = 0
    inserted = 0
    newlyinserted = 0
    score = 0
    traci.start([sumoBinary, "-c", "merging.sumocfg",
                            "--tripinfo-output", "tripinfo.xml"])
    v_1_speed = np.random.uniform(low= 12, high= 15)
    first = True
    while inserted < 6:
        traci.simulationStep()
        newlyinserted = traci.simulation.getDepartedNumber()
        if newlyinserted:
            insertedID = traci.simulation.getDepartedIDList()[0]
            traci.vehicle.setSpeedMode(insertedID, 0)
            if first:
                traci.vehicle.setSpeed("v_0", np.random.uniform(11, 14)) 
                traci.vehicle.setSpeed("v_1", v_1_speed) 
                traci.vehicle.setSpeedMode("v_0", 0)
                traci.vehicle.setSpeedMode("v_1", 0)
                first = False
            else:
                traci.vehicle.setSpeed(insertedID, v_1_speed)
            inserted += newlyinserted
            newlyinserted = 0
    traci.simulationStep()
    

    
    state = get_state()   
    collision = 0   
    while not done :
        # action = agent.choose_action(state)
        action = np.random.choice(3)
        nstate, reward, done = move(state, action)
        if training:
            if not done:
                agent.store_transition(state, action, reward, 
                                            nstate, done)
            else:
                agent.terminal_memory[agent.index] = True
            agent.learn()
        state = nstate
        score += reward
        step += 1
        if traci.simulation.getCollidingVehiclesIDList() != ():
            collision = 1
    traci.close()
    sys.stdout.flush()
    
    return collision, score

# main entry point
if __name__ == "__main__":
    options = get_options()
    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    if options.testing:
        training = False
    else:
        training = True
    # traci starts sumo as a subprocess and then this script connects and runs
    agent = Agent(gamma=0.99, epsilon=0.01, batch_size=64, n_actions=3, eps_end=0.0,
                  input_dims=[18], lr=0.001, eps_dec=1e-5)
    
    scores = []
    best = - np.inf
    epoch = 300
    version = '4'
    collisions = 0
    if not training:
        agent.load('Qv' + version + '.pth')
    for i in range(epoch):
        collision, score = run(training)
        collisions += collision
        scores.append(score)
        avg_score = np.mean(scores[-100:])

        print('episode ', i + 1, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'best score %.2f' % best, 
                'epsilon %.2f' % agent.epsilon) 
        if training:
            if i % 100 == 0 and avg_score > best and i > 0:
                print('saving the model ....')        
                agent.save('Qv' + version + 'pth')
                best = avg_score
            if i == epoch - 1:
                agent.save('Qv' + version + '_last.pth')
if training:        
    np.save('scores' + version + '.npy', scores)    
success_rate = (epoch-collisions)/epoch*100
print('Success rate: %.3f' % success_rate)   