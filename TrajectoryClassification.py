import numpy as np
from scipy.stats import maxwell
import pandas as pd

######TESTING#######

trajectories = pd.read_excel('test_trajectory.xlsx')
y_min = 149597870691 #meters
y_max = 149597870691 + 100000 #meters
z_length = 1600000*1000 #meters
beta = z_length/2
alpha = y_min + 218*1000

#####################

# A particle escapes if it travels below alpha while outside of beta or ends the simulation outside of beta
# A particle is recaptured if it ends the simulation below alpha and within beta or if it colides with beta below alpha
# A particle is resimulated if it ends the simulation above alpha but within beta (it may leave beta and then reenter)

def classify_trajectory(alpha,beta,trajectories):

    recaptured = False
    escaped = False
    resimulate = False
    beta_crossings = 0
    
    for i in list(range(1,len(trajectories))):
        x = trajectories.iloc[i,0]
        y = trajectories.iloc[i, 1]
        z = trajectories.iloc[i, 2]
        
        x_prev = trajectories.iloc[i-1,0]
        y_prev = trajectories.iloc[i-1,1]
        z_prev = trajectories.iloc[i-1,2]
            
        if abs(z) >= beta and y < alpha and recaptured != True and escaped != True: #check if particle has entered the side of ringworld
            if abs(z_prev) < beta: #if this is true, it hit the side
                recaptured = True
            else: #otherwise, it came from above
                escaped = True

        #this logs betacrossings by seing if the particle crossed beta between the two current timesteps
        if abs(z) >= beta and abs(z_prev) < beta:
            beta_crossings += 1
        if abs(z) <= beta and abs(z_prev) > beta:
            beta_crossings += 1

    #This accounts for particles that did not hit one of the initial ending conditions during the simulation by looking at
    #their ending locations
    if recaptured == False and escaped == False:
        if abs(z) <= beta and y > alpha:
            resimulate = True
        elif abs(z) <= beta and y < alpha:
            recaptured = True
        else:
            escaped = True

    if recaptured == True:
        result = 'recaptured'
    elif escaped == True:
        result = 'escaped'
    else:
        result = 'resimulate'

    return beta_crossings, result

[beta_crossings, result] = classify_trajectory(alpha,beta,trajectories)

    


