import time

from lib import Environment

env = Environment(gui=True)
ret = env.solve_ik(env.co_grasp_pre, False, random_sampling=True)
print(f"finish first ik: {ret}")

traj = env.default_relative_trajectory()
env.reproduce_relative_trajectory(traj)

print(f"finish second ik: {ret}")
env.grasp(False)
print("finish third grasp")
env.translate([0, 0, 0.05], True)
time.sleep(1000)
