import pygame
import random
import torch
from dqn import DQN, ReplayMemory
import os

class Bird:
    def __init__(self, y, draw_offset=100):
        self.y = y
        self.width = 70
        self.height = 70
        self.x = self.width / 2
        self.gravity = 0.6
        self.jump_strength = 12
        self.velocity = 0
        self.draw_offset = draw_offset

    def jump(self):
        self.velocity = -self.jump_strength

    def update(self):
        self.velocity += self.gravity
        self.velocity = round(self.velocity, 1)
        self.y += self.velocity
        self.y = round(self.y, 1)
    
    def getRect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 0, 0), self.getRect().move(self.draw_offset, 0))
    
    def collide(self, pipe):
        return self.getRect().colliderect(pipe.getTopRect()) or self.getRect().colliderect(pipe.getBottomRect())
    
    def getState(self):
        return [self.x, self.width, self.y, self.y + self.height, self.velocity]

class Pipe:
    def __init__(self, x, height, draw_offset=100):
        self.draw_offset = draw_offset
        self.x = x
        self.width = 100
        self.gap = 300
        self.height = height
        self.y = random.randint(50 + self.gap // 2, height - self.gap // 2 - 50)
        self.velocity = 6

    def update(self):
        self.x -= self.velocity

    def getTopRect(self):
        return pygame.Rect(self.x, 0, self.width, self.y - self.gap // 2)
    def getBottomRect(self):
        return pygame.Rect(self.x, self.y + self.gap // 2, self.width, self.height - self.y - self.gap // 2)

    def draw(self, screen):
        pygame.draw.rect(screen, (0, 255, 0), self.getTopRect().move(self.draw_offset, 0))
        pygame.draw.rect(screen, (0, 255, 0), self.getBottomRect().move(self.draw_offset, 0))

    def getState(self):
        return [self.x, self.x + self.width, self.y, self.getTopRect().height, self.getBottomRect().y]


class Environment:
    def __init__(self, renderGame=False, mode='train'):
        self.renderGame = renderGame
        self.mode = mode
        self.width = 1300
        self.height = 800
        self.pipeSpace = 400
        self.pipeVelocity = 6
        self.draw_offset = 200
        self.seed = 0
        self.fps = 60

        if self.renderGame:
            if self.mode == 'human':
                os.environ['SDL_VIDEO_WINDOW_POS'] = '100,200'
            else:
                os.environ['SDL_VIDEO_WINDOW_POS'] = '700,200'
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Flappy Bird DQN")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 30)
        
        self.reset()

    def reset(self, seed=-1):
        if seed != -1:
            random.seed(self.seed)
        self.bird = Bird(self.height/2 + random.randint(-100, 100), draw_offset=self.draw_offset)
        self.pipes = []
        self.score = 0
        self.steps = 0
        self.done = False
        self.seed = seed
        self.fps = 60

        self.pipes.append(self.newPipe(self.width - self.pipeSpace*2))
        self.pipes.append(self.newPipe(self.width - self.pipeSpace))
        self.pipes.append(self.newPipe(self.width))

        return self.getState()
    
    def quit(self):
        if self.renderGame:
            pygame.quit()
        self.done = True

    def newPipe(self, x):
        return Pipe(x, self.height, draw_offset=self.draw_offset)
    
    def getState(self):
        bird_state = self.bird.getState()
        upcoming_pipes = [pipe for pipe in self.pipes if pipe.x + pipe.width > 0]
        pipe = upcoming_pipes[0]
        next_pipe = upcoming_pipes[1]
        pipe_state = pipe.getState()
        next_pipe_state = next_pipe.getState()
        return torch.tensor(bird_state + pipe_state + next_pipe_state + [self.pipeVelocity], dtype=torch.float32)
    
    def step(self, action):
        if self.done:
            raise ValueError("Environment is done. Please reset before stepping.")
        self.steps += 1

        if action == 1:
            self.bird.jump()
        
        self.bird.update()
        if self.bird.y < 0 or self.bird.y > self.height:
            self.done = True

        for pipe in self.pipes:
            pipe.velocity = self.pipeVelocity
            pipe.update()
            if pipe.x + pipe.width < -self.draw_offset:
                self.pipes.remove(pipe)
                self.score += 1
            if self.bird.collide(pipe):
                self.done = True
        # self.pipeVelocity += 0.0005

        if len(self.pipes) < 2 or self.pipes[-1].x < self.width - self.pipeSpace:
            self.pipes.append(self.newPipe(self.width))

        # upcoming_pipes = [pipe for pipe in self.pipes if pipe.x + pipe.width > 0]
        # pipe = upcoming_pipes[0]
        reward = 1 if not self.done else -100

        return self.getState(), reward, self.done
    
    def render(self):
        if not self.renderGame:
            raise ValueError("Render is not enabled.")

        self.screen.fill((0, 0, 255))
        for pipe in self.pipes:
            pipe.draw(self.screen)
        self.bird.draw(self.screen)

        score_text = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.fps += 0.1
        self.clock.tick(self.fps)
    
    def play(self):
        if self.mode != 'human':
            raise ValueError("Mode must be 'human' for play method.")
        if not self.renderGame:
            raise ValueError("Render must be True for play method.")
        
        steps = 0

        while not self.done:
            steps += 1
            self.render()
            action = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
                    steps = -1
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        action = 1
            if not self.done:
                state, _, _ = self.step(action)

        return steps