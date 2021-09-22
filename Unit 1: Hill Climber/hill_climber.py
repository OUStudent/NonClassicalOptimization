import numpy as np


def steepest_ascent_hill_climber(objective_function, lower_bound, upper_bound, max_iter,
                                 step_size=0.01, initial_point=None, stag_count=20, info=False):
    """
    Performs Steepest Ascent Hill Climber using 2*n possible neighbors at each position

    :param objective_function: A function pointer to the objective function
    :param lower_bound: The lower bound of the domain of the variables - list
    :param upper_bound: The upper bound of the domain of the variables - list
    :param max_iter: The maximum number of iterations the algorithm will take - int
    :param step_size: The percentage of the total domain for creating the steps - float range between [0, 1]
    :param initial_point: The random initial point used to start the algorithm, if None it will be created
                            randomly from the domain space
    :param stag_count: The number of iterations allowed for stagnation, where f(x) does not change - int
    :param info: Boolean for whether or not to print out information
    :return:
    """

    total_bound = np.asarray(upper_bound) - np.asarray(lower_bound)

    # create initial point if none
    if initial_point is None:
        initial_point = np.empty(shape=(len(lower_bound),))
        for i in range(0, len(lower_bound)):
            initial_point[i] = np.random.uniform(lower_bound[i], upper_bound[i], 1)[0]

    # create steps for each respective variable
    steps = step_size * total_bound

    current = initial_point
    current_f = objective_function(initial_point)

    func_eval = 1  # count number of function evaluations
    stag = 0  # count number of stagnation - times the f(x) did not change
    min_f = [current_f]  # keep track of best neighbor f(x) value for each iteration

    for i in range(0, max_iter):
        if info:
            print("Iteration: {}".format(i))
            print("F(x): {}".format(current_f))

        visited_points = [current]  # visited neighbors, include current position
        visited_f = [current_f]  # visited neighbor f(x) values

        # loop over all variables
        for j in range(0, len(total_bound)):
            # create first child/neighbor by adding step size
            child1 = np.copy(current)
            child1[j] += steps[j]
            if child1[j] > upper_bound[j]:
                child1[j] = upper_bound[j]
            visited_points.append(child1)
            visited_f.append(objective_function(child1))

            # create second child/neighbor by subtracting step size
            child2 = np.copy(current)
            child2[j] -= steps[j]
            if child2[j] < lower_bound[j]:
                child2[j] = lower_bound[j]
            visited_points.append(child2)
            visited_f.append(objective_function(child2))

            func_eval += 2  # objective function was called twice

        # find best neighbor by finding min f(x) value - assuming minimization
        bst = np.argmin(visited_f)
        min_f.append(visited_f[bst])

        current = visited_points[bst]
        current_f = visited_f[bst]

        if bst == 0:  # if best index is 0 then it is current position, so stagnated
            stag += 1
            if stag == stag_count:  # stagnated for stag_count iterations
                if info:
                    print("Stagnation")
                break

        else:
            stag = 0

    if info:
        print("Function Eval: {}".format(func_eval))
    return current, np.asarray(min_f)  # return current position and array of best f(x) per iteration for plotting


def stochastic_variable_and_uniform_step_hill_climber(objective_function, lower_bound, upper_bound, max_iter,
                                                      num_variables, step_size=0.01, initial_point=None,
                                                      stag_count=20, info=True):
    """
    Performs Steepest Ascent Hill Climber using 2*n possible neighbors at each position

    :param objective_function: A function pointer to the objective function
    :param lower_bound: The lower bound of the domain of the variables - list
    :param upper_bound: The upper bound of the domain of the variables - list
    :param max_iter: The maximum number of iterations the algorithm will take - int
    :param num_variables: The number of variables to sample from the possible set of variables - int
    :param step_size: The percentage of the total domain for creating the steps - float range between [0, 1]
    :param initial_point: The random initial point used to start the algorithm, if None it will be created
                            randomly from the domain space
    :param stag_count: The number of iterations allowed for stagnation, where f(x) does not change - int
    :param info: Boolean for whether or not to print out information
    :return:
    """

    total_bound = np.asarray(upper_bound) - np.asarray(lower_bound)

    # create initial point if none
    if initial_point is None:
        initial_point = np.empty(shape=(len(lower_bound),))
        for i in range(0, len(lower_bound)):
            initial_point[i] = np.random.uniform(lower_bound[i], upper_bound[i], 1)[0]

    # create steps for each respective variable
    steps = step_size * total_bound

    current = initial_point
    current_f = objective_function(initial_point)

    func_eval = 1  # count number of function evaluations
    stag = 0  # count number of stagnation - times the f(x) did not change
    min_f = [current_f]  # keep track of best neighbor f(x) value for each iteration

    for i in range(0, max_iter):
        if info:
            print("Iteration: {}".format(i))
            print("F(x): {}".format(current_f))

        visited_points = [current]  # visited neighbors, include current position
        visited_f = [current_f]  # visited neighbor f(x) values


        chosen_variables = np.random.choice(range(0, len(total_bound)), num_variables, replace=False)
        # loop over sampled variable indices
        for j in chosen_variables:
            child = np.copy(current)
            # only create one child by adding random uniform value between the step sizes
            child[j] += np.random.uniform(-steps[j], steps[j], 1)[0]

            # check bounds
            if child[j] > upper_bound[j]:
                child[j] = upper_bound[j]
            elif child[j] < lower_bound[j]:
                child[j] = lower_bound[j]

            visited_points.append(child)
            visited_f.append(objective_function(child))
            func_eval += 1

        # find best neighbor by finding min f(x) value - assuming minimization
        bst = np.argmin(visited_f)
        min_f.append(visited_f[bst])

        current = visited_points[bst]
        current_f = visited_f[bst]

        if bst == 0:  # if best index is 0 then it is current position, so stagnated
            stag += 1
            if stag == stag_count:  # stagnated for stag_count iterations
                if info:
                    print("Stagnation")
                break

        else:
            stag = 0

    if info:
        print("Function Eval: {}".format(func_eval))
    return current, np.asarray(min_f)  # return current position and array of best f(x) per iteration for plotting
  
def sphere_function(x):  # [5.12,-5.12] min at 0 vector
    if len(x.shape) == 1:
        return np.sum(np.power(x, 2))
    else:
        t = [0] * len(x)
        for i in range(0, len(x)):
            t[i] = np.sum(x[i,] ** 2)
        return np.asarray(t)

def eggholder_function(x):  # [512,-512] min at -959.6407 -  only supports n=2
    if len(x.shape) == 1:
        return -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[0] / 2 + (x[1] + 47)))) - x[0] * np.sin(
        np.sqrt(np.abs(x[0] - (x[1] + 47))))
    else:
        return -(x[:, 1] + 47) * np.sin(np.sqrt(np.abs(x[:, 0] / 2 + (x[:, 1] + 47)))) - x[:, 0] * np.sin(
            np.sqrt(np.abs(x[:, 0] - (x[:, 1] + 47))))

def shubert_function(x):  # [-5, 5] min changes depending on n
    if len(x.shape) == 1:
        t = [1, 2, 3, 4, 5]
        loc_sum = 0
        for i in range(0, len(t)):
            loc_sum += np.sum(t[i] * np.cos(x * (t[i] + 1) + t[i]))
        return loc_sum
    else:
        z = []
        t = [1, 2, 3, 4, 5]
        for row in x:
            loc_sum = 0
            for i in range(0, len(t)):
                loc_sum += np.sum(t[i] * np.cos(row * (t[i] + 1) + t[i]))
            z.append(loc_sum)
        return np.asarray(z)

      
# example use:
d = 1000
lower_bound = [-5]*d
upper_bound = [5]*d

step_sizes = [1, 0.5, 0.1, 0.01]

objective_function = shubert_function

for step in step_sizes:
    fx_values = []
    for i in range(0, 30):
        point, chart2 = stochastic_variable_and_uniform_step_hill_climber(objective_function, lower_bound=lower_bound,
                                                                          upper_bound=upper_bound, max_iter=10000,
                                                                          num_variables=10, step_size=step, info=False,
                                                                          stag_count=100)
        fx_values.append(objective_function(point))
    print("Step: {} - Min: {}, Median: {}, stdev: {}".format(step, np.min(fx_values), np.median(fx_values),
                                                             np.std(fx_values)))

    fx_values = []
    for i in range(0, 1):
        point, chart2 = stochastic_variable_and_uniform_step_hill_climber(objective_function, lower_bound=lower_bound,
                                                                          upper_bound=upper_bound, max_iter=100000,
                                                                          num_variables=10, step_size=step, info=False,
                                                                          stag_count=100)
        fx_values.append(objective_function(point))
    print("2 Mill Function Eval - Min: {}".format(np.min(fx_values)))


