import numpy as np
import random as rd
import matplotlib.pyplot as plt
import GA_utils as genf
import time

# hyperparameters (user inputted parameters)
prob_crsvr = 1 # probablity of crossover
prob_mutation = 0.3 # probablity of mutation
population = 120 # population number
generations = 80 # generation number

# x and y decision variables' encoding
# 13 genes for x and 13 genes for y (arbitrary number)
x_y_string = np.array([0,1,0,0,0,1,0,0,1,0,1,1,1,
                       0,1,1,1,0,0,1,0,1,1,0,1,1])

# create an empty array to put initial population
pool_of_solutions = np.empty((0,len(x_y_string)))

# create an empty array to store a solution from each generation
best_of_a_generation = np.empty((0,len(x_y_string)+1))

# shuffle n times, where n is the no. of the desired population
for i in range(population):
    rd.shuffle(x_y_string)
    pool_of_solutions = np.vstack((pool_of_solutions,x_y_string))

start_time = time.time() # start time (timing purposes)
gen = 1 # we start at generation no.1 (tracking purposes)

for i in range(generations): # do it n (generation) times
    
    # empty array for saving the new generation
    new_population = np.empty((0,len(x_y_string)))
    
    # empty array for saving the new generation plus its obj func val
    new_population_with_obj_val = np.empty((0,len(x_y_string)+1))
    
    # empty array for saving the best solution (chromosome)
    sorted_best_for_plotting = np.empty((0,len(x_y_string)+1))
    
    print()
    print()
    print("--> Generation: #", gen)
    family = 1 
    
    for j in range(int(population/2)): # population/2 because each gives 2 parents
        print()
        print("--> Family: #", family) # tracking purposes
             
        # selecting 2 parents using tournament selection
        parent_1 = genf.find_parents_ts(pool_of_solutions)[0]
        parent_2 = genf.find_parents_ts(pool_of_solutions)[1]
    
        # crossover the 2 parents to get 2 children
        child_1 = genf.crossover(parent_1,parent_2, prob_crsvr=prob_crsvr)[0]
        child_2 = genf.crossover(parent_1,parent_2, prob_crsvr=prob_crsvr)[1]
        
        # mutating the 2 children to get 2 mutated children
        mutated_child_1 = genf.mutation(child_1,child_2, prob_mutation=prob_mutation)[0]  
        mutated_child_2 = genf.mutation(child_1,child_2, prob_mutation=prob_mutation)[1] 
        
        # getting the obj val (fitness value) for the 2 mutated children
        obj_val_mutated_child_1 = genf.objective_value(mutated_child_1)[2]
        obj_val_mutated_child_2 = genf.objective_value(mutated_child_2)[2]
        
        # track each mutant child and its obj val
        print()
        print("Obj Val for Mutated Child #1 at Generation #{} : {}".
              format(gen,obj_val_mutated_child_1))
        print("Obj Val for Mutated Child #2 at Generation #{} : {}".
              format(gen,obj_val_mutated_child_2))

        # for each mutated child, put its obj val next to it
        mutant_1_with_obj_val = np.hstack((obj_val_mutated_child_1, mutated_child_1))
        
        mutant_2_with_obj_val = np.hstack((obj_val_mutated_child_2, mutated_child_2))
        
        
        # create the new population for the next generation
        new_population = np.vstack((new_population, 
                                    mutated_child_1, 
                                    mutated_child_2))
        
        new_population_with_obj_val = np.vstack((new_population_with_obj_val,
                                                 mutant_1_with_obj_val,
                                                 mutant_2_with_obj_val))
        
        family = family+1
        
    # this new pool of solutions becomes the starting population of the next generation
    pool_of_solutions = new_population
    sorted_best_for_plotting = np.array(sorted(new_population_with_obj_val,
                                               key=lambda x:x[0]))
    
    # the best in that generation would be the first solution in the array
    best_of_a_generation = np.vstack((best_of_a_generation,
                                      sorted_best_for_plotting[0]))
    gen = gen+1       
end_time = time.time() # end time (timing purposes)

# sort them from best to worst
sorted_last_population = np.array(sorted(new_population_with_obj_val,
                                         key=lambda x:x[0]))
sorted_best_of_a_generation = np.array(sorted(best_of_a_generation,
                                         key=lambda x:x[0]))

# the best would be the first solution in the array
best_string_convergence = sorted_last_population[0]
best_string_overall = sorted_best_of_a_generation[0]

print()
print()
print("------------------------------")
print()
print("Execution Time in Seconds:",end_time - start_time) # exec. time
print()
print("Final Solution (Convergence):",best_string_convergence[1:]) # final solution entire chromosome
print("Encoded X (Convergence):",best_string_convergence[1:14]) # final solution x chromosome
print("Encoded Y (Convergence):",best_string_convergence[14:]) # final solution y chromosome
print()
print("Final Solution (Best):",best_string_overall[1:]) # final solution entire chromosome
print("Encoded X (Best):",best_string_overall[1:14]) # final solution x chromosome
print("Encoded Y (Best):",best_string_overall[14:]) # final solution y chromosome

# to decode the x and y chromosomes to their real values
final_solution_convergence = genf.objective_value(best_string_convergence[1:])
final_solution_overall = genf.objective_value(best_string_overall[1:])

print()
print("Decoded X (Convergence):",round(final_solution_convergence[0],5)) # real value of x
print("Decoded Y (Convergence):",round(final_solution_convergence[1],5)) # real value of y
print("Obj Value - Convergence:",round(final_solution_convergence[2],5)) # obj val of final chromosome
print()
print("Decoded X (Best):",round(final_solution_overall[0],5)) # real value of x
print("Decoded Y (Best):",round(final_solution_overall[1],5)) # real value of y
print("Obj Value - Best in Generations:",round(final_solution_overall[2],5)) # obj val of final chromosome
print()
print("------------------------------")

### FOR PLOTTING THE BEST SOLUTION FROM EACH GENERATION ###
best_obj_val_convergence = (best_string_convergence[0]) 
best_obj_val_overall = best_string_overall[0]
plt.plot(best_of_a_generation[:,0]) 
plt.axhline(y=best_obj_val_convergence,color='r',linestyle='--')
plt.axhline(y=best_obj_val_overall,color='r',linestyle='--')
plt.title("Z Reached Through Generations",fontsize=20,fontweight='bold')
plt.xlabel("Generation",fontsize=18,fontweight='bold')
plt.ylabel("Z",fontsize=18,fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')

if sorted_best_of_a_generation[-1][0] > 2:
    k = 0.8
elif sorted_best_of_a_generation[-1][0] > 1:
    k = 0.5
elif sorted_best_of_a_generation[-1][0] > 0.5:
    k = 0.3
elif sorted_best_of_a_generation[-1][0] > 0.3:
    k = 0.2
else:
    k = 0.1

xyz1 = (generations/2.4,best_obj_val_convergence)
xyzz1 = (generations/2.2,best_obj_val_convergence+k)

plt.annotate("At Convergence: %0.5f" % best_obj_val_convergence,xy=xyz1,xytext=xyzz1,
             arrowprops=dict(facecolor='black',shrink=1,width=1,headwidth=5),
             fontsize=12,fontweight='bold')
xyz2 = (generations/6,best_obj_val_overall)
xyzz2 = (generations/5.4,best_obj_val_overall+(k/2))
plt.annotate("Minimum Overall: %0.5f" % best_obj_val_overall,xy=xyz2,xytext=xyzz2,
             arrowprops=dict(facecolor='black',shrink=1,width=1,headwidth=5),
             fontsize=12,fontweight='bold')
plt.show()
