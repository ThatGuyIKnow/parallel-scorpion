# mutual_info.pyx

# Enable C++ support in Cython
# distutils: language = c++
# cython: language_level=3

# Import C++ standard library components
from libcpp.vector cimport vector

# Declare the C++ functions you want to use
cdef extern from "greedy_mi_max_variable_order.hpp":
    cdef vector[int] greedyMiMaxVariableOrderSimd(const vector[vector[int]]& states)
    cdef vector[int] greedyEntropyMinVariableOrderSimd(const vector[vector[int]]& states)
    cdef vector[int] greedyEntropySingleMinVariableOrderSimd(const vector[vector[int]]& states)


def greedy_mi_max_variable_order(py_states):
    cdef vector[vector[int]] states
    cdef vector[int] vec
    for py_vector in py_states:
        vec.clear()
        for val in py_vector:
            vec.push_back(val)
        states.push_back(vec)
    
    
    cdef vector[int] result = greedyMiMaxVariableOrderSimd(states)
    return list(result)



def greedy_ent_min_variable_order(py_states):
    cdef vector[vector[int]] states
    cdef vector[int] vec
    for py_vector in py_states:
        vec.clear()
        for val in py_vector:
            vec.push_back(val)
        states.push_back(vec)
        
    cdef vector[int] result = greedyEntropyMinVariableOrderSimd(states)
    return list(result)



def greedy_ent_single_min_variable_order(py_states):
    cdef vector[vector[int]] states
    cdef vector[int] vec
    for py_vector in py_states:
        vec.clear()
        for val in py_vector:
            vec.push_back(val)
        states.push_back(vec)
    
    
    cdef vector[int] result = greedyEntropySingleMinVariableOrderSimd(states)
    return list(result)
