import multiprocessing
from game import Environment
from dqn import DQN
import torch
import pygame
import random


def human_game(env):
    steps = env.play()
    return steps
    

def bot_game(env, n_observations, n_actions, state, conn):
    model = DQN(n_observations, n_actions)
    model.load_state_dict(torch.load("model2.pth"))

    steps = 0
    other_steps = 0

    done = False
    while not done:
        steps += 1
        action = model.action(state)
        next_state, reward, done = env.step(action)
        state = next_state

        if not done:
            env.render()

        if steps > 5000:
            break

        if conn.poll():
            msg = conn.recv()
            if msg == 'halt':
                done = True
                break
            other_steps = msg

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return -1
            elif event.type == pygame.MOUSEBUTTONDOWN:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True

    return steps, other_steps


def run_window(conn, mode):
    env = Environment(renderGame=True, mode=mode)

    done = False
    while not done:
        # Wait for a seed
        seed = conn.recv()
        if seed == -1:
            done = True
            break
        # Reset the environment with the received seed
        state = env.reset(seed)
        n_observations = len(state)
        n_actions = 2

        steps = 0

        env.render()
        pygame.time.wait(2000)

        other_steps = -1
        if mode == 'human':
            steps = human_game(env)
        else:
            steps, other_steps = bot_game(env, n_observations, n_actions, state, conn)

        conn.send(steps)

        if steps == -1:
            done = True
            break

        if conn.poll() or (mode == 'test' and other_steps == -1):
            other_steps = conn.recv()
        elif mode == 'human':
            # Wait for the other app to send its steps
            text = env.font.render("Hit enter to stop AI", True, (255, 255, 255))
            env.screen.blit(text, (env.width // 2 - text.get_width() // 2, env.height // 2 - text.get_height() // 2))
            pygame.display.flip()
            waiting = True
            while waiting:
                if conn.poll():
                    other_steps = conn.recv()
                    waiting = False
                    break
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting = False
                        done = True
                        conn.send(-1)
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                            conn.send('halt')
            env.render()
        if other_steps == -1:
            done = True
            break

        if other_steps > steps:
            text = env.font.render("You lose", True, (255, 0, 0))
        else:
            text = env.font.render("You win", True, (0, 255, 0))
        env.screen.blit(text, (env.width // 2 - text.get_width() // 2, env.height // 2 - text.get_height() // 2))

        if mode == 'human':
            text2 = env.font.render("Hit enter to try again", True, (255, 255, 255))
            env.screen.blit(text2, (env.width // 2 - text2.get_width() // 2, env.height // 2 + text.get_height() // 2))

        pygame.display.flip()

        # Wait for a keypress or the other app to try again
        waiting = True
        while waiting:
            if conn.poll():
                msg = conn.recv()
                if msg == -1:
                    waiting = False
                    done = True
                    break
                if msg == 'next':
                    waiting = False
                    break
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                    done = True
                    conn.send(-1)
                    break
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    waiting = False
                    conn.send('next')
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        waiting = False
                        conn.send('next')
                    elif event.key == pygame.K_ESCAPE:
                        waiting = False
                        done = True
                        break

    conn.send('closing')
    env.quit()

if __name__ == '__main__':
    parent_conn1, child_conn1 = multiprocessing.Pipe()
    parent_conn2, child_conn2 = multiprocessing.Pipe()

    # Create two processes
    p1 = multiprocessing.Process(target=run_window, args=(child_conn1, 'human'))
    p2 = multiprocessing.Process(target=run_window, args=(child_conn2, 'test'))

    p1.start()
    p2.start()

    seed = random.randint(0, 10000)
    parent_conn1.send(seed)  # send initial seed to the first window
    parent_conn2.send(seed)  # send initial seed to the second window

    try:
        closed1 = False
        closed2 = False
        while not closed1 or not closed2:
            if parent_conn1.poll():
                msg = parent_conn1.recv()
                if msg == 'closing':
                    closed1 = True
                else:
                    parent_conn2.send(msg)  # forward to other window

                    if msg == 'next':
                        seed = random.randint(0, 10000)
                        parent_conn1.send(seed)
                        parent_conn2.send(seed)

            if parent_conn2.poll():
                msg = parent_conn2.recv()
                if msg == 'closing':
                    closed2 = True
                else:
                    parent_conn1.send(msg)  # forward to other window

                    if msg == 'next':
                        seed = random.randint(0, 10000)
                        parent_conn1.send(seed)
                        parent_conn2.send(seed)
    except KeyboardInterrupt:
        pass

    p1.join()
    p2.join()
