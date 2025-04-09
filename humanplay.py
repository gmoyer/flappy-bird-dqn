from game import Environment

env = Environment(render=True, mode='human')
env.play()
print(env.steps)
env.quit()