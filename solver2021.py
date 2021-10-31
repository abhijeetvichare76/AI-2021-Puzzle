#!/usr/local/bin/python3
# solver2021.py : 2021 Sliding tile puzzle solver
#
# Code by: Abhijeet 2000933720
#
# Based on skeleton code by D. Crandall & B551 Staff, September 2021
#

import sys
import numpy as np

ROWS=5
COLS=5

def printable_board(board):
    return [ ('%3d ')*COLS  % board[j:(j+COLS)] for j in range(0, ROWS*COLS, COLS) ]
# return a list of possible successor states
def successors(state):
    # taket the board as input and return all the new successors and the move played 
    # return : list of tuples where each tuple (successor, move)
    #The successors include: Oc, Occ, Ic, ICC, D1-D5, U1-U5, L1-L5, R1-R5 
    
    succs = []
    #new sucessor for each row rotate
    for row in range(5):
        temp_state = [i[:] for i in state]
        #same row go right
        for col in range(5):
            temp_state[row][(col+1)%5] = state[row][col]
        succs.append((temp_state,f'R{row+1}'))
        temp_state = [i[:] for i in state]
        # same row go left
        for col in range(5):
            temp_state[row][(col-1)%5] = state[row][col] 
        succs.append((temp_state,f'L{row+1}'))
    # new successor for each col
    for col in range(5):
        temp_state = [i[:] for i in state]
        # same col go down
        for row in range(5):
            temp_state[(row+1)%5][col] = state[row][col] 
        succs.append((temp_state,f'D{col+1}'))
        temp_state = [i[:] for i in state]
        # same col go up
        for row in range(5):
            temp_state[(row-1)%5][col] = state[row][col] 
        succs.append((temp_state,f'U{col+1}'))
    
    state = np.array(state)
    #OUTER
    temp_state = np.array(state)
    # ckwise
    #1st row
    temp_state[0,1:] = state[0,:-1] 
    #last row
    temp_state[-1,:-1] = state[-1,1:]
    #1st col
    temp_state[:-1,0] = state[1:,0]
    #Last col
    temp_state[1:,-1] = state[:-1,-1]
    succs.append((temp_state.tolist(),'Oc'))
    #counter ckwise
    temp_state = np.array(state)
    #1st row
    temp_state[0,:-1] = state[0,1:] 
    #last row
    temp_state[-1,1:] = state[-1,:-1]
    #1st col
    temp_state[1:,0] = state[:-1,0]
    #Last col
    temp_state[:-1,-1] = state[1:,-1]
    succs.append((temp_state.tolist(),'Occ'))

    # INNER
    #clockwise
    temp_state = np.array(state)
    #2st row
    temp_state[1,1:-2] = state[1,2:-1] 
    #2nd last row
    temp_state[-2,2:-1] = state[-2,1:-2]
    #2st col
    temp_state[2:-1,1] = state[1:-2,1]
    #2nd Last col
    temp_state[1:-2,-2] = state[2:-1,-2]
    succs.append((temp_state.tolist(),'Ic'))
    #ckwise
    temp_state = np.array(state)
    #2st row
    temp_state[1,2:-1] = state[1,1:-2] 
    #2nd last row
    temp_state[-2,1:-2] = state[-2,2:-1]
    #2st col
    temp_state[1:-2,1] = state[2:-1,1]
    #Last col
    temp_state[2:-1,-2] = state[1:-2,-2]
    succs.append((temp_state.tolist(),'Icc'))
    return succs

def is_goal(state):
    # check if we've reached the goal
    flag = True
    total_checks = 5*5
    for i in range(total_checks):
        if state[i//5][i%5] != i+1:
            flag = False
    return flag

def places_from_target(board):
    # The breakup of each cell with its directions its away from its ideal place. 
    # horizontal: + means right and - means left
    # vertical: + means down and - means up
    # Return: a 2d array where each element is the form of tuple (Vertical direction, horizontal direction)
    moves_list = [[0 for i in range(5)] for j in range(5)]
    for row in range(5):
        for col in range(5):
            curr_num = board[row][col]
            tar_row = (curr_num-1)//5
            tar_col = (curr_num-1)%5 
            # for col
            to_move_h = tar_col - col
            if to_move_h!= 0:
                raw_direction = int(to_move_h/abs(to_move_h))
            #take the shortest dist
            if abs(to_move_h) > 5 - abs(to_move_h):
                to_move_h = (5 - abs(to_move_h))*raw_direction*(-1)

            # for row
            to_move_v = tar_row - row
            if to_move_v!= 0:
                raw_direction = int(to_move_v/abs(to_move_v))
            #take the shortest dist
            if abs(to_move_v) > 5 - abs(to_move_v):
                to_move_v = (5 - abs(to_move_v))*raw_direction*(-1)
            moves_list[row][col] = (to_move_v,to_move_h)
    return moves_list

def costs_unidirection(moves_to_make,operation):
    #Calculate the cost of each row/column operation. 
    # Take the max of the unidirections move per row/column.
    # return list of costs for each cell.
    dir_costs = []
    r_c = int(operation[1])-1
    direction = operation[0]
    if direction in ['L','U']:
        sign = -1
    else:
        sign = 1

    if direction in ['L','R']:
        dir_moves = moves_to_make[r_c,:,1]
    elif direction in ['U','D']:
        dir_moves = moves_to_make[:,r_c,0]      

    dir_moves = [sign*i for i in dir_moves if i*sign >0]
    dir_costs.extend(dir_moves)

    return dir_costs

def heuristic(state):
    #IDEA: Check cells that can be moved together in a move and take max of the moves needed in that group. Then move the individual cells that can't be grouped.
    # heiarchy of grouping: OC -> OCC -> IC -> ICC -> 3U -> 3R -> 3D -> 3L ->
    #  2R(1,4,5) -> 2L(1,2,5) -> 4R (1,4,5) -> 4L (1,2,5) -> 2U (1,2,5) -> 
    #  2D(1,4,5) -> 4U(1,2,5) -> 4D(1,4,5) ->
    #  1L(1) -> 1U(1) -> 1R(5) -> 5U(1) -> 1D(5) -> 5L(1) -> 5D(5) -> 5R(5)
    #reminder: right is +ve and down is +ve and -ve sign for the opp direction
    moves_to_make = np.array(places_from_target(state))
    # state = np.array(state)
    cost = 0
    #Group 1 with merging costs: OC + R1 + D5 + L5 + U1
    #OC     
    oc_group = []
    # 0R,0-4C (R)
    r_moves = moves_to_make[0,:-1,1]
    r_moves = [i for i in r_moves if i >0]
    oc_group.extend(r_moves)
    #0-4R,4C (D)
    d_moves = moves_to_make[:-1,-1,0]
    d_moves = [i for i in d_moves if i >0]
    oc_group.extend(d_moves)
    #4R,1-4C (L)
    l_moves = moves_to_make[-1,1:,1]
    l_moves = [-i for i in l_moves if i <0]
    oc_group.extend(l_moves)
    #1-4R,0C (U)
    u_moves = moves_to_make[1:,0,0]
    u_moves = [-i for i in u_moves if i <0]
    oc_group.extend(u_moves)
    oc_group.append(0)
    cost += max(oc_group)
    #OC complete

    #OCC
    occ_group = []
    # 0R,1:C (L)
    l_moves = moves_to_make[0,1:,1]
    l_moves = [-i for i in l_moves if i <0]
    occ_group.extend(l_moves)
    #1:R,4C (U)
    u_moves = moves_to_make[1:,-1,0]
    u_moves = [-i for i in u_moves if i <0]
    occ_group.extend(u_moves)
    #4R,:-1C (R)
    r_moves = moves_to_make[-1,:-1,1]
    r_moves = [i for i in r_moves if i >0]
    occ_group.extend(r_moves)
    # :4R,0C (D)
    d_moves = moves_to_make[:-1,0,0]
    d_moves = [i for i in d_moves if i >0]
    occ_group.extend(d_moves) 
    occ_group.append(0)
    cost += max(occ_group)
    #OCC complete

    #IC
    ic_group = []
    # 1R,1:-2C (R)
    r_moves = moves_to_make[1,1:-2,1]
    r_moves = [i for i in r_moves if i >0]
    ic_group.extend(r_moves)
    #1:-2R,-2C (D)
    d_moves = moves_to_make[1:-2,-2,0]
    d_moves = [i for i in d_moves if i >0]
    ic_group.extend(d_moves)
    #-2R,2:-2C (L)
    l_moves = moves_to_make[-2,2:-2,1]
    l_moves = [-i for i in l_moves if i <0]
    ic_group.extend(l_moves)
    #2:-2R,1C (U)
    u_moves = moves_to_make[2:-2,1,0]
    u_moves = [-i for i in u_moves if i <0]
    ic_group.extend(u_moves)
    ic_group.append(0)
    cost += max(ic_group)
    #IC complete

    #ICC
    icc_group = []
    # 1R,2:-2C (L)
    l_moves = moves_to_make[1,2:-2,1]
    l_moves = [-i for i in l_moves if i <0]
    icc_group.extend(l_moves)
    # 2:-2R,-2C (U)
    u_moves = moves_to_make[2:-2,-2,0]
    u_moves = [-i for i in u_moves if i <0]
    icc_group.extend(u_moves)
    #-2R,1:-2C (R)
    r_moves = moves_to_make[-2,1:-2,1]
    r_moves = [i for i in r_moves if i >0]
    icc_group.extend(r_moves)
    # 1:-2R,1C (D)
    d_moves = moves_to_make[1:-2,1,0]
    d_moves = [i for i in d_moves if i >0]
    icc_group.extend(d_moves) 
    icc_group.append(0)
    cost += max(icc_group)
    #ICC complete

    # 3U -> 3R -> 3D -> 3L
    move_3u = moves_to_make[:,2,0]
    move_3u = [-i for i in move_3u if i<0]
    move_3u.append(0)
    cost+=max(move_3u)
    move_3d = moves_to_make[:,2,0]
    move_3d = [i for i in move_3d if i>0]
    move_3d.append(0)
    cost+=max(move_3d)
    move_3r = moves_to_make[2,:,1]
    move_3r = [i for i in move_3r if i>0]
    move_3r.append(0)
    cost+=max(move_3r)
    move_3l = moves_to_make[2,:,1]
    move_3l = [-i for i in move_3l if i<0]
    move_3l.append(0)
    cost+=max(move_3l)
        
    #  2R(1,4,5) -> 2L(1,2,5) -> 4R (1,4,5) -> 4L (1,2,5) ->
    #  2U(1,2,5) -> 2D(1,4,5) -> 4U(1,2,5) -> 4D(1,4,5)
    move_2r = moves_to_make[1,(0,3,4),1]
    move_2r = [i for i in move_2r if i>0]
    move_2r.append(0)
    cost+=max(move_2r)
    move_2l = moves_to_make[1,(0,1,4),1]
    move_2l = [-i for i in move_2l if i<0]
    move_2l.append(0)
    cost+=max(move_2l)
    move_4r = moves_to_make[3,(0,3,4),1]
    move_4r = [i for i in move_4r if i>0]
    move_4r.append(0)
    cost+=max(move_4r)
    move_4l = moves_to_make[3,(0,1,4),1]
    move_4l = [-i for i in move_4l if i<0]
    move_4l.append(0)
    cost+=max(move_4l)
    #L/R DONE. ONTO U/D
    move_2d = moves_to_make[(0,3,4),1,0]
    move_2d = [i for i in move_2d if i>0]
    move_2d.append(0)
    cost+=max(move_2d)
    move_2u = moves_to_make[(0,1,4),1,0]
    move_2u = [-i for i in move_2u if i<0]
    move_2u.append(0)
    cost+=max(move_2u)
    move_4d = moves_to_make[(0,3,4),3,0]
    move_4d = [i for i in move_4d if i>0]
    move_4d.append(0)
    cost+=max(move_4d)
    move_4u = moves_to_make[(0,1,4),3,0]
    move_4u = [-i for i in move_4u if i<0]
    move_4u.append(0)
    cost+=max(move_4u)

    #  1L(1) -> 1U(1) -> 1R(5) -> 5U(1) -> 1D(5) -> 5L(1) -> 5D(5) -> 5R(5)
    move_1l = moves_to_make[0,0,1]
    cost+=abs(move_1l)
    move_1u = moves_to_make[0,0,0]
    cost+=abs(move_1u)
    move_1r = moves_to_make[0,4,1]
    cost+=abs(move_1r)
    move_1d = moves_to_make[4,0,0]
    cost+=abs(move_1d)
    #1st ones done. now last ones
    move_5l = moves_to_make[4,0,1]
    cost+=abs(move_5l)
    move_5u = moves_to_make[0,4,0]
    cost+=abs(move_5u)
    move_5r = moves_to_make[4,4,1]
    cost+= abs(move_5r)
    move_5d = moves_to_make[4,4,0]
    cost+= abs(move_5d)

    return cost

def heuristic_1(state):
    #IDEA: Continue on original heuristic fn.
    # Create a more simplistic heuristic fn that groups together rows and columns that mimic the movements of Oc,Occ, Ic, and Icc.
    # For EG: OC will include 1R, 5D, 5L, 1U.
    # New steps: 
    # 1. OC : R1 + D5 + L5 + U1
    # 2.OCC : L1 + D1 + R5 + U5
    # 3. IC : R2 + D4 + L4 + U2
    # 4.ICC : L2 + D2 + R4 + U4 
    # 5. ADD(3L + 3R + 3U + 3D)
    cost = 0
    moves_to_make = np.array(places_from_target(state))
    #All steps:
    all_steps = [
        ['R1' , 'D5' , 'L5' , 'U1'],
        ['L1' , 'D1' , 'R5' , 'U5'],
        ['R2' , 'D4' , 'L4' , 'U2'],
        ['L2' , 'D2' , 'R4' , 'U4'],
        ['L3'],['R3'],['D3'],['U3']
    ]
    hits = 0
    for one_step in all_steps:
        step_costs = []
        for operation in one_step:
            step_costs.extend(costs_unidirection(moves_to_make,operation))
        if step_costs:
            hits+=1
            cost += max(step_costs)

    # The group overlap across groups. To tackle that reduced a factor of num_matches_group from the cost function.
    cost = cost - (hits-1)
    return cost

def heuristic_2(state):
    # heurestic new was causing issues when the move to be made was Oc and family. Trying to solve that issue.
    # New formula :
    # MIN(MAX(OC)+ 4 CORNERS EXTRPOLATING THE OC DIRECTIONS,  MAX (OC DIRECTIONS i.e R1 + D5 + L5 + U1)
    cost = 0
    moves_to_make = np.array(places_from_target(state))
    # OC STEP:
    oc_group = []
    # 0R,0-4C (R)
    r_moves = moves_to_make[0,:-1,1]
    r_moves = [i for i in r_moves if i >0]
    oc_group.extend(r_moves)
    #0-4R,4C (D)
    d_moves = moves_to_make[:-1,-1,0]
    d_moves = [i for i in d_moves if i >0]
    oc_group.extend(d_moves)
    #4R,1-4C (L)
    l_moves = moves_to_make[-1,1:,1]
    l_moves = [-i for i in l_moves if i <0]
    oc_group.extend(l_moves)
    #1-4R,0C (U)
    u_moves = moves_to_make[1:,0,0]
    u_moves = [-i for i in u_moves if i <0]
    oc_group.extend(u_moves)
    oc_group.append(0)
    
    oc_cost = max(oc_group)
    #Add the R1(5) + D5(5) + L5(1) + U1(1)
    cell_cost = moves_to_make[0,4,1]
    if cell_cost >0:
        oc_cost += cell_cost
    cell_cost = moves_to_make[4,4,0]
    if cell_cost >0:
        oc_cost += cell_cost
    cell_cost = moves_to_make[4,0,1]
    if cell_cost <0:
        oc_cost += abs(cell_cost)
    cell_cost = moves_to_make[0,0,0]
    if cell_cost <0:
        oc_cost += abs(cell_cost)
    alternate_costs = 0
    for operation in ['R1' , 'D5', 'L5', 'U1']:
        alternate_costs += max(costs_unidirection(moves_to_make,operation))
    cost += min(oc_cost,alternate_costs)
    #OC Complete

    #OCC
    occ_group = []
    # 0R,1:C (L)
    l_moves = moves_to_make[0,1:,1]
    l_moves = [-i for i in l_moves if i <0]
    occ_group.extend(l_moves)
    #1:R,4C (U)
    u_moves = moves_to_make[1:,-1,0]
    u_moves = [-i for i in u_moves if i <0]
    occ_group.extend(u_moves)
    #4R,:-1C (R)
    r_moves = moves_to_make[-1,:-1,1]
    r_moves = [i for i in r_moves if i >0]
    occ_group.extend(r_moves)
    # :4R,0C (D)
    d_moves = moves_to_make[:-1,0,0]
    d_moves = [i for i in d_moves if i >0]
    occ_group.extend(d_moves) 
    occ_group.append(0)
    
    occ_cost = max(occ_group)
    #Add the L1(1) + D1(5) + R5(5) + U5(1)
    cell_cost = moves_to_make[0,0,1] #L1(1)
    if cell_cost <0:
        occ_cost += abs(cell_cost)
    cell_cost = moves_to_make[4,0,0] #D1(5)
    if cell_cost >0: 
        occ_cost += abs(cell_cost)
    cell_cost = moves_to_make[4,4,1] # R5(5)
    if cell_cost >0:
        occ_cost += abs(cell_cost)
    cell_cost = moves_to_make[0,4,0] # U5(1)
    if cell_cost <0:
        occ_cost += abs(cell_cost)
    alternate_costs = 0
    for operation in ['L1' , 'D1' , 'R5' , 'U5']:
        alternate_costs += max(costs_unidirection(moves_to_make,operation))
    cost += min(occ_cost,alternate_costs)
    #OCC complete
    
    #IC
    ic_group = []
    # 1R,1:-2C (R)
    r_moves = moves_to_make[1,1:-2,1]
    r_moves = [i for i in r_moves if i >0]
    ic_group.extend(r_moves)
    #1:-2R,-2C (D)
    d_moves = moves_to_make[1:-2,-2,0]
    d_moves = [i for i in d_moves if i >0]
    ic_group.extend(d_moves)
    #-2R,2:-2C (L)
    l_moves = moves_to_make[-2,2:-2,1]
    l_moves = [-i for i in l_moves if i <0]
    ic_group.extend(l_moves)
    #2:-2R,1C (U)
    u_moves = moves_to_make[2:-2,1,0]
    u_moves = [-i for i in u_moves if i <0]
    ic_group.extend(u_moves)
    ic_group.append(0)
    ic_cost = max(ic_group)
    #Add the R2(1,4,5) + D4(1,45) + L5(1,2,5) + U2(1,2,5)

    r_moves = moves_to_make[1,(0,3,4),1]
    r_moves = [i for i in r_moves if i >0]
    r_moves.append(0)
    ic_cost+= max(r_moves)
    
    d_moves = moves_to_make[(0,3,4),3,0]
    d_moves = [i for i in d_moves if i >0]
    d_moves.append(0)
    ic_cost+=max(d_moves)
    
    l_moves = moves_to_make[3,(0,1,4),1]
    l_moves = [-i for i in l_moves if i <0]
    l_moves.append(0)
    ic_cost+=max(l_moves)
    
    u_moves = moves_to_make[(0,1,4),1,0]
    u_moves = [-i for i in u_moves if i <0]
    u_moves.append(0)
    ic_cost+=max(u_moves)
    alternate_costs = 0
    for operation in ['R2' , 'D4' , 'L4' , 'U2']:
        alternate_costs += max(costs_unidirection(moves_to_make,operation))
    cost += min(ic_cost,alternate_costs)
    #IC complete

    #ICC
    icc_group = []
    # 1R,2:-2C (L)
    l_moves = moves_to_make[1,2:-2,1]
    l_moves = [-i for i in l_moves if i <0]
    icc_group.extend(l_moves)
    # 2:-2R,-2C (U)
    u_moves = moves_to_make[2:-2,-2,0]
    u_moves = [-i for i in u_moves if i <0]
    icc_group.extend(u_moves)
    #-2R,1:-2C (R)
    r_moves = moves_to_make[-2,1:-2,1]
    r_moves = [i for i in r_moves if i >0]
    icc_group.extend(r_moves)
    # 1:-2R,1C (D)
    d_moves = moves_to_make[1:-2,1,0]
    d_moves = [i for i in d_moves if i >0]
    icc_group.extend(d_moves) 
    icc_group.append(0)
    icc_cost = max(icc_group)
    #ICC complete

    #Add the R2(1,4,5) + D4(1,45) + L5(1,2,5) + U2(1,2,5)
    r_moves = moves_to_make[3,(0,3,4),1]
    r_moves = [i for i in r_moves if i >0]
    r_moves.append(0)
    icc_cost+= max(r_moves)
    
    d_moves = moves_to_make[(0,3,4),1,0]
    d_moves = [i for i in d_moves if i >0]
    d_moves.append(0)
    icc_cost+=max(d_moves)
    
    l_moves = moves_to_make[1,(0,1,4),1]
    l_moves = [-i for i in l_moves if i <0]
    l_moves.append(0)
    icc_cost+=max(l_moves)
    
    u_moves = moves_to_make[(0,1,4),3,0]
    u_moves = [-i for i in u_moves if i <0]
    u_moves.append(0)
    icc_cost+=max(u_moves)

    alternate_costs = 0
    for operation in ['L2' , 'D2' , 'R4' , 'U4']:
        alternate_costs += max(costs_unidirection(moves_to_make,operation))
    cost += min(icc_cost,alternate_costs)

    all_steps = [ ['L3'],['R3'],['D3'],['U3']    ]
    for one_step in all_steps:
        step_costs = []
        for operation in one_step:
            step_costs.extend(costs_unidirection(moves_to_make,operation))
        if step_costs:
            cost += max(step_costs)
    return cost
def heuristic_3(state):
    #IDEA: Continue on original heuristic fn.
    # Create a more simplistic heuristic fn that groups together rows and columns that mimic the movements of Oc,Occ, Ic, and Icc.
    # For EG: OC will include 1R, 5D, 5L, 1U.
    # New steps: 
    # 1. OC : R1 + D5 + L5 + U1
    # 2.OCC : L1 + D1 + R5 + U5
    # 3. IC : R2 + D4 + L4 + U2
    # 4.ICC : L2 + D2 + R4 + U4 
    # 5. ADD(3L + 3R + 3U + 3D)
    cost = 0
    moves_to_make = np.array(places_from_target(state))
    #All steps:
    all_steps = [
        ['R1' , 'D5' , 'L5' , 'U1'],
        ['L1' , 'D1' , 'R5' , 'U5'],
        ['R2' , 'D4' , 'L4' , 'U2'],
        ['L2' , 'D2' , 'R4' , 'U4'],
        ['L3'],['R3'],['D3'],['U3']
    ]
    for one_step in all_steps:
        step_costs = []
        for operation in one_step:
            step_costs.extend(costs_unidirection(moves_to_make,operation))
        if step_costs:
            cost += max(step_costs)

    # The group overlap across groups. To tackle that reduced a factor of num_matches_group from the cost function.
    return cost
def create_a_problem(n_operations):
    # Helper function to create random problems for the board to solve.
    operations = []
    initial_board = [[0 for i in range(5)] for j in range(5)]
    initial_state = [i+1 for i in range(5*5)]
    for i in range(len(initial_state)):
        initial_board[i//5][i%5] = initial_state[i]    
    
    inter_board = initial_board
    for i in range(n_operations):
        succs = successors(inter_board)
        index = np.random.randint(low = 0, high = len(succs)-1)
        inter_board, inter_operation = succs[index]
        operations.append(inter_operation)
    return (inter_board,operations)

def solve(initial_state):
    """
    Implementation of A start search to solve the 2021 problem.
    Steps followed in the algorithm:
    1. Initial state: The input board provided. 
    2. Initialize fringe with the initial state and the initial A* cost.
    3. If the initial_board is the goal_state then end
    4. While Fringe not empty:
        take the most promising state form the fringe and take its successors;
            find successors of the state:
            for each successor:
                if goal_state;
                end
            if successor already visited: 
                skip 
            append fringe in the sucessor.

    return format: the moves taken to reach the goal state. 
    """

    initial_board = [[0 for i in range(5)] for j in range(5)]

    for i in range(len(initial_state)):
        initial_board[i//5][i%5] = initial_state[i]
    
    moves = []
    initial_cost = heuristic_3(initial_board)
    initial_function_cost = initial_cost + len(moves)
    fringe = [(initial_board,moves,initial_function_cost)]

    if is_goal(initial_board):
        return moves
    
    closed = [fringe[0]]
    while fringe:
        all_costs = [i[-1] for i in fringe]
        least_index = np.argmin(all_costs)
        (curr_board,curr_moves,curr_fn_cost) = fringe.pop(least_index)
        closed.append((curr_board,curr_moves,curr_fn_cost))
        for new_board,next_path in successors(curr_board):
            new_cost = heuristic_3(new_board)
            new_fn_cost = new_cost + (len(curr_moves) + 1)
            if is_goal(new_board):
                return curr_moves + [next_path]
            elif new_board in [i[0] for i in closed]:
                continue
            else:
                fringe.append((new_board,curr_moves + [next_path],new_fn_cost))
    return -1

# Please don't modify anything below this line
#
if __name__ == "__main__":
    if(len(sys.argv) != 2):
        raise(Exception("Error: expected a board filename"))

    start_state = []
    with open(sys.argv[1], 'r') as file:
        for line in file:
            start_state += [ int(i) for i in line.split() ]

    if len(start_state) != ROWS*COLS:
        raise(Exception("Error: couldn't parse start state file"))

    print("Start state: \n" +"\n".join(printable_board(tuple(start_state))))

    print("Solving...")
    route = solve(tuple(start_state))
    
    print("Solution found in " + str(len(route)) + " moves:" + "\n" + " ".join(route))
