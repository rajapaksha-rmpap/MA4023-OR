# ========================= HELPER FUNCTIONS =========================
import os, sys, copy

import numpy as np
from tabulate import tabulate

# visualizing the tabulations
def visualize_tabulation(tabulation, all_vars, basic_vars, big_M=False):

    tabulation = np.round(tabulation, 4)

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
        ratio_col = solution / pivot_col
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
            raise Exception(f"non-basic variable '{non_basic_var}' has a zero value in the objective row; therefore, MANY SOLUTIONS EXISTS...")
        
    return tabulation, all_vars, basic_vars
