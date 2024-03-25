import os

from .FuSolver import FuSolver



def create_solver(opt):
    if opt['mode'] == 'fu':
        solver = FuSolver(opt)
    else:
        raise NotImplementedError

    return solver


# def create_solver(opt):
#     if opt['mode'] == 'fu':
#         solver = FuSolver(opt)
#     else:
#         raise NotImplementedError
#
#     return solver
