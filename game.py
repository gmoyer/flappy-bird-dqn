import pygame
import random
import torch

class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 50
        self.height = 50
        self.gravity = 0.4
        self.jump_strength = 6
        self.velocity = 0

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
        pygame.draw.rect(screen, (255, 0, 0), self.getRect())
    
    def collide(self, pipe):
        return self.getRect().colliderect(pipe.getTopRect()) or self.getRect().colliderect(pipe.getBottomRect())
    
    def getState(self, offset):
        return [self.x - offset, self.y, self.velocity]

class Pipe:
    def __init__(self, x, height):
        self.x = x
        self.width = 80
        self.gap = 150
        self.height = height
        self.y = random.randint(50 + self.gap // 2, height - self.gap // 2 - 50)
        self.velocity = 3

    def update(self):
        self.x -= self.velocity

    def getTopRect(self):
        return pygame.Rect(self.x, 0, self.width, self.y - self.gap // 2)
    def getBottomRect(self):
        return pygame.Rect(self.x, self.y + self.gap // 2, self.width, self.height - self.y - self.gap // 2)

    def draw(self, screen):
        pygame.draw.rect(screen, (0, 255, 0), self.getTopRect())
        pygame.draw.rect(screen, (0, 255, 0), self.getBottomRect())

    def getState(self, offset):
        return [self.x - offset, self.x - offset + self.width, self.y, self.getTopRect().height, self.getBottomRect().y]


class Environment:
    def __init__(self, renderGame=False, mode='train'):
        self.renderGame = renderGame
        self.mode = mode
        self.width = 600
        self.height = 400
        self.pipeSpace = 200
        self.offset = 100 # Offset for the bird's x position

        if self.renderGame:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Flappy Bird DQN")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 30)
        
        self.reset()

    def reset(self):
        self.bird = Bird(self.offset, 200)
        self.pipes = []
        self.score = 0
        self.steps = 0
        self.done = False

        self.pipes.append(self.newPipe(self.width - self.pipeSpace))
        self.pipes.append(self.newPipe(self.width))

        return self.getState()
    
    def quit(self):
        if self.renderGame:
            pygame.quit()
        self.done = True

    def newPipe(self, x):
        return Pipe(x, self.height)
    
    def getState(self):
        bird_state = self.bird.getState(self.offset)
        upcoming_pipes = [pipe for pipe in self.pipes if pipe.x + pipe.width > self.offset]
        pipe = upcoming_pipes[0]
        next_pipe = upcoming_pipes[1]
        pipe_state = pipe.getState(self.offset)
        next_pipe_state = next_pipe.getState(self.offset)
        return torch.tensor(bird_state + pipe_state + next_pipe_state, dtype=torch.float32)
    
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
            pipe.update()
            if pipe.x + pipe.width < 0:
                self.pipes.remove(pipe)
                self.score += 1
            if self.bird.collide(pipe):
                self.done = True

        if len(self.pipes) < 2 or self.pipes[-1].x < self.width - self.pipeSpace:
            self.pipes.append(self.newPipe(self.width))

        reward = 1 if not self.done else -10

        return self.getState(), reward, self.done
    
    def render(self):
        if not self.renderGame:
            raise ValueError("Render is not enabled.")
        if self.done:
            raise ValueError("Environment is done. Please reset before rendering.")

        self.screen.fill((0, 0, 255))
        for pipe in self.pipes:
            pipe.draw(self.screen)
        self.bird.draw(self.screen)

        score_text = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(60)
    
    def play(self):
        if self.mode != 'human':
            raise ValueError("Mode must be 'human' for play method.")
        if not self.renderGame:
            raise ValueError("Render must be True for play method.")

        while not self.done:
            action = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        action = 1

            self.step(action)
            self.render()