from game import Environment

env = Environment(renderGame=True, mode='human')
env.play()
print(env.steps)
env.quit() 