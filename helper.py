# ========================= HELPER FUNCTIONS =========================
import os, sys, copy

import numpy as np
import pandas as pd

from tabulate import tabulate
from fractions import Fraction

# visualizing the tabulations
def visualize_tabulation(tabulation, all_vars, basic_vars, big_M=False):

    tabulation = copy.deepcopy(tabulation)
    # tabulation = np.round(tabulation, 4)

    header = np.concatenate(([], all_vars, ["sol"]))

    if big_M:
        # if big-M method is implemented, combine the first two rows to create the objective function row
        obj_func_row = []
        for i in range(tabulation.shape[1]):
            entry = ''
            if tabulation[0][i] != 0:
                entry += str(tabulation[0][i])
            if tabulation[1][i] != 0:
                if tabulation[1][i] > 0 and entry != '': entry += '+'
                entry += str(tabulation[1][i]) + 'M'
            if entry == '': entry = 0
            obj_func_row.append(entry)
        obj_func_row = np.reshape(obj_func_row, (1, -1))
        tabulation = np.concatenate((obj_func_row, tabulation[2: ]), axis=0)
        basic_vars = np.concatenate(([basic_vars[0]], basic_vars[2: ]))

    print(tabulate(np.concatenate((np.array(basic_vars).reshape(-1, 1), tabulation), axis=1), headers=header, stralign='center', tablefmt='simple_grid'))

# all slack starting method
def all_slack_starting(tabulation, all_vars, basic_vars, opt_type):

    iteration_no = 0
    tabulation = copy.deepcopy(tabulation)

    for _ in range(10): # *****

        # ====================================== ENTERING VARIABLE ======================================
        # check the optimality condition and the entering variable accordingly
        if opt_type == "MAX":
            if not (tabulation[0][1:-1] < 0).any():
                print(f"*****there are no more negative values in the objective row; tabulation is optimal...*****\n")
                break
            pivot_col_value = np.min(tabulation[0][1:-1]) # highest negative value
        else: # opt_type == "MIN"
            if not (tabulation[0][1:-1] > 0).any():
                print(f"*****there are no more positive values in the objective row; tabulation is optimal...*****\n")
                break
            pivot_col_value = np.max(tabulation[0][1:-1]) # highest positive value

        iteration_no += 1
        print("="*20 + f" iteration no.: {iteration_no} " + "="*20)

        pivot_col_idx = list(tabulation[0][1:-1]).index(pivot_col_value) + 1

        entering_var = all_vars[pivot_col_idx]
        print(f"entering var: {entering_var}")
        pivot_col = tabulation[:, pivot_col_idx]
        print(f"pivot col: {pivot_col}")

        # find the ratio column
        solution  = tabulation[:, -1]
        print(f"solution : {solution}")
        ratio_col = np.float64(solution) / np.float64(pivot_col)
        print(f"ratio col: {ratio_col}")

        # ====================================== LEAVING VARIABLE =======================================
        # find the lowest positive value in the ratio column
        if not (ratio_col[1: ] > 0).any():
            raise Exception(f"there is no positive value in the ratio column; therefore, NO FEASIBLE SOLUTION EXISTS...")

        pivot_row_value = np.min(ratio_col[1: ][ratio_col[1: ] > 0])
        pivot_row_idx = list(ratio_col[1: ]).index(pivot_row_value) + 1

        leaving_var = basic_vars[pivot_row_idx]
        print(f"leaving var : {leaving_var}")
        pivot_row = tabulation[pivot_row_idx]
        print(f"pivot row: {pivot_row}")

        # replace the leaving variable with the entering variable
        basic_vars[pivot_row_idx] = entering_var
        print(f"new basic variables: {basic_vars}")

        # ------------------------- convert the pivot column to a pivot vector --------------------------

        # verify that the pivot column is convertible to a pivot vector
        if pivot_row[pivot_col_idx] == 0:
            raise Exception(f"the pivot column (var '{entering_var}') at index {pivot_col_idx} is not convertible to a pivot vector...")

        # transform the pivot row
        pivot_row_ = pivot_row / pivot_row[pivot_col_idx]
        tabulation[pivot_row_idx] = pivot_row_

        # transform all the other rows
        for i in range(col_size := tabulation.shape[0]):

            if i == pivot_row_idx:
                continue

            row = tabulation[i]
            tabulation[i] = row - row[pivot_col_idx] * pivot_row_

        visualize_tabulation(tabulation, all_vars=all_vars, basic_vars=basic_vars)

        # the values in the solution column except the one in the objective function row must be non-negative
        is_feasible = (tabulation[:, -1][1: ] >= 0).all()

        if not is_feasible:
            raise Exception(f"the tabulation is not feasible...")
        print(f"the tabulation is feasible.")

        print('\n')

    # check for many solutions
    non_basic_vars = list(set(all_vars) - set(basic_vars))
    non_basic_vars.sort()
    for non_basic_var in non_basic_vars:
        # find the objective row value for each non-basic variable
        print(non_basic_var, tabulation[0][all_vars.index(non_basic_var)])
        if tabulation[0][all_vars.index(non_basic_var)] == 0:
            print(f"WARNING: non-basic variable '{non_basic_var}' has a zero value in the objective row; therefore, MANY SOLUTIONS EXISTS...")
        
    return tabulation, basic_vars

# dual simplex method
def dual_simplex_to_all_slack_starting(tabulation, all_vars, basic_vars):

    # take a deepcopy of the input tabulation not to alter existing tabulation
    tabulation = copy.deepcopy(tabulation)

    iteration_no = 0
    while (tabulation[:, -1][1: ] < 0).any():
        # until there are negative values in the solution column
        iteration_no += 1
        print("="*20 + f"DUAL SIMPLEX - iteration no.: {iteration_no}" + "="*20)

        # ------------------------------ find the leaving variable ------------------------------
        pivot_row_val = np.min(tabulation[:, -1][1: ]) # max negative solution value
        if list(tabulation[:, -1][1: ]).count(pivot_row_val) > 1:
            print(f"WARNING: there are multiple instances of the maximum negative solution value {pivot_row_val}...")
        pivot_row_idx = list(tabulation[:, -1][1: ]).index(pivot_row_val) + 1
        pivot_row_var = basic_vars[pivot_row_idx]
        print(f"leaving variale : {pivot_row_var}")

        # ----------------------------- find the entering variable ------------------------------
        ratio_row = np.float64(np.abs(tabulation[0])) / np.float64(np.abs(tabulation[pivot_row_idx]))
        print(f"ratio row       : {ratio_row}")
        pivot_col_val = np.min(ratio_row[1:-1][tabulation[pivot_row_idx][1:-1] < 0]) # surely positive

        # there can be multiple instances of `pivot_col_value`; must find a correct one
        pivot_col_idx = 0
        for _ in range(list(ratio_row[1:-1]).count(pivot_col_val)):
            pivot_col_idx = list(ratio_row[pivot_col_idx+1:-1]).index(pivot_col_val) + 1
            if tabulation[pivot_row_idx][pivot_col_idx] < 0: break
        pivot_col_var = all_vars[pivot_col_idx]
        print(f"entering variale: {pivot_col_var}")
        basic_vars[pivot_row_idx] = pivot_col_var

        # --------------------- convert the pivot column to a pivot vector ----------------------
        tabulation[pivot_row_idx] /= tabulation[pivot_row_idx][pivot_col_idx]
        pivot_row = tabulation[pivot_row_idx]

        for row_idx in range(tabulation.shape[0]):
            if row_idx == pivot_row_idx: continue

            row = tabulation[row_idx]
            tabulation[row_idx] = row - row[pivot_col_idx] * pivot_row

        visualize_tabulation(tabulation, all_vars=all_vars, basic_vars=basic_vars)
        print('\n')
    
    return tabulation, basic_vars

# tabulation saving and loading functions
def save_tabulation(tabulation, all_vars, basic_vars, file_name):

    if not file_name.endswith('.csv'): file_name += '.csv'

    columns_labels = copy.deepcopy(all_vars)
    columns_labels.append("sol")

    DF = pd.DataFrame(tabulation, index=basic_vars, columns=columns_labels)
    DF.to_csv(file_name)

def load_tabulation(file_name):

    if not file_name.endswith('.csv'): file_name += '.csv'

    DF = pd.read_csv(file_name, index_col=0)  # Assuming the row labels are stored in the first column
    tabulation = DF.values
    for i in range(tabulation.shape[0]):
        tabulation[i] = np.array(list(map(Fraction, tabulation[i])))

    basic_vars = DF.index.tolist()
    all_vars = DF.columns.tolist()

    return tabulation, all_vars[ :-1], basic_vars

# functions for integer programming
def add_constraint(tabulation, constraint, new_var, all_vars, basic_vars):

    tabulation = copy.deepcopy(tabulation)
    all_vars = copy.deepcopy(all_vars)
    basic_vars = copy.deepcopy(basic_vars)

    # adding a new row for the basic slack variable
    tabulation = np.concatenate((tabulation, constraint.reshape(1, -1)))

    # adding a new column
    column = np.zeros(tabulation.shape[0])
    column[-1] = 1
    column = np.array(list(map(Fraction, column)))
    temp = tabulation[:, range(tabulation.shape[1]-1)]
    temp = np.concatenate((temp, column.reshape(-1, 1)), axis=1)
    tabulation = np.concatenate((temp, tabulation[:, -1].reshape(-1, 1)), axis=1)

    # add the new variable
    all_vars.append(new_var)
    basic_vars.append(new_var)

    return tabulation, all_vars, basic_vars

def integer_program(tabulation, all_vars, basic_vars, opt_type):

    def is_integer(tabulation):
        value = True
        for sol in tabulation[:, -1][1: ]:
            if sol.denominator != 1: value = False; break
        return value
    
    def extract_frac(arr):

        frac_arr = np.zeros([len(arr)], dtype=object)

        for i, real_num in enumerate(arr):
            frac_arr[i] = real_num - Fraction(int(real_num))
        
        return frac_arr

    tabulation = copy.deepcopy(tabulation)
    all_vars = copy.deepcopy(all_vars)
    basic_vars = copy.deepcopy(basic_vars)

    iteration_no = 0
    while not is_integer(tabulation):

        iteration_no += 1
        print("#"*30 + f" ITERATION NO. {iteration_no} " + "#"*30)

        # find the largest fraction in the solution column
        solution_frac_arr = extract_frac(tabulation[:, -1][1: ])
        max_frac = np.max(solution_frac_arr)
        print(f"max. frac: {max_frac}")

        # find the corresponding resource row (there can be multiple instances of `max_frac`)
        resource_row_idx = list(solution_frac_arr).index(max_frac) + 1
        print(f"resource var: {basic_vars[resource_row_idx]}")

        # create the constraint
        constraint = -extract_frac(tabulation[resource_row_idx])
        tabulation, all_vars, basic_vars = add_constraint(tabulation, constraint, new_var="G"+str(iteration_no), all_vars=all_vars, basic_vars=basic_vars)
        visualize_tabulation(tabulation, all_vars, basic_vars)
        
        # apply dual simplex method
        print("\nAPPLYING DUAL SIMPLEX...")
        tabulation, basic_vars = dual_simplex_to_all_slack_starting(tabulation, all_vars, basic_vars)
        
        # apply all-slack starting to make the tabulation optimal
        print("\nAPPLYING ALL-SLACK STARTNG...")
        tabulation, basic_vars = all_slack_starting(tabulation, all_vars, basic_vars, opt_type)

        visualize_tabulation(tabulation, all_vars, basic_vars)

        if iteration_no > 5: 
            print("the maximum no. of iteration is surpassed...")
            break
