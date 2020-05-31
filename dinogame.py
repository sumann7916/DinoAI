import pygame
import random
import math
import os
import neat
import time

# Initializing the game and the size of the window
pygame.init()
WIN_WIDTH = 800
WIN_HEIGHT = 600
score_value = 0
font = pygame.font.Font("freesansbold.ttf", 32)
textX = 10
textY = 10

pygame.display.set_caption("Dino game")

# Uploading Sprites
playerimg = []
for i in range(3):
    playerimg.append(pygame.image.load("img/run0.png"))

for i in range(3):
    playerimg.append(pygame.image.load('img/run1.png'))

OBS_IMG = [pygame.image.load("img/cactus1.png"), pygame.image.load("img/cactus2.png"),
           pygame.image.load("img/cactus3.png"), pygame.image.load("img/berd.png")]
jump_img = pygame.image.load("img/dinojump0000.png")


# for keeping score
def show_score(screen, score):
    score = font.render("Score: " + str(score), True, (0, 0, 0))
    screen.blit(score, (textX, textY))


class Player:
    IMGS = playerimg
    tick = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vel = 0
        self.img = self.IMGS[0]

    def jump(self):
        if self.y > 399:
            self.vel = -15

    def move(self):
        self.y += self.vel
        if self.y < 300:
            self.vel += 3
        if self.y > 399:
            self.y = 400

    # for checking pixel perfect collision
    def get_mask(self):
        return pygame.mask.from_surface(self.img)

    def draw(self, screen):
        self.tick += 1
        self.img = self.IMGS[(self.tick % 6)]
        screen.blit(self.img, (self.x, self.y))


class Obstacle:
    # movment velocity of the obstacles
    VEL = 30

    def __init__(self):
        self.x = 850
        self.y = 0
        self.passed = False
        # randomly generating Images from list of images
        self.img = random.choice(OBS_IMG)

    def move(self):
        self.x -= self.VEL

    def draw(self, screen):
        # Changing the Y position of the obstacles with relation to object
        if self.img == OBS_IMG[0]:
            self.y = 380
        if self.img == OBS_IMG[1]:
            self.y = 420
        if self.img == OBS_IMG[2]:
            self.y = 420
        if self.img == OBS_IMG[3]:
            self.y = 400
        screen.blit(self.img, (self.x, self.y))

    def collide(self, player):
        # Checking for collision using get mask function
        player_mask = player.get_mask()
        obj_mask = pygame.mask.from_surface(self.img)
        obj_offset = (self.x - player.x, self.y - round(player.y))

        collision_point = player_mask.overlap(obj_mask, obj_offset)

        if collision_point:
            return True
        return False


def draw_window(screen, players, obstacles, score):
    # Drawing and updating the screen
    screen.fill((255, 255, 255))
    pygame.draw.line(screen, (0, 0, 0), (0, 500), (800, 500))
    for obs in obstacles:
        obs.draw(screen)
    show_score(screen, score)
    for player in players:
        player.draw(screen)
    pygame.display.update()


def main(genomes, config):
    players = []
    nets = []
    ge = []

    # List of Genomes
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        players.append(Player(100, 400))
        ge.append(genome)

    # Creating list of obstacle class objects
    obstacles = [Obstacle()]
    screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    score = 0

    run = True

    while run:

        # for slowing the frame
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
        obj_ind = 0
        if len(players) > 0:
            if len(obstacles) > 1 and players[0].x > obstacles[0].x + obstacles[0].img.get_width():
                obj_ind = 1
        else:
            break

        for x, player in enumerate(players):  # give each bird a fitness of 0.1 for each frame it stays alive
            player.move()
            output = nets[x].activate((player.x, abs(player.x-obstacles[obj_ind].x)))

            if output[0] > 0.5:
                player.jump()
        add_obs = False
        rem = []
        for obs in obstacles:
            for x, player in enumerate(players):
                if obs.collide(player):
                    ge[x].fitness -= 1
                    players.pop(x)
                    nets.pop(x)
                    ge.pop(x)


            # to see if obs has crossed the player
                if not obs.passed and obs.x < player.x:
                    obs.passed = True
                    add_obs = True

                    # for removing the obs
            if obs.x < -50:
                rem.append(obs)
                score += 1

                for g in ge:
                    g.fitness += 10
            obs.move()

        if add_obs:
            obstacles.append(Obstacle())

        # removing obstacles which has crossed
        for r in rem:
            obstacles.remove(r)

        draw_window(screen, players, obstacles, score)


def run(config_file):
    # Config function for NEAT
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 50)

    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
