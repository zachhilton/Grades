

from __future__ import print_function
import os
import neat
import visualize
import math

from sklearn.datasets import load_boston

#[Subject 1 (O,1),Subject ME (0,1), Gen Ed (0,1), Year level (1,2,3,4), how many credits that semester, which semester I was in (1-8), credits per class]

inputs = [
         #SUBJECT 0
         (0,1,1,1,16,1,3,),  #Essentials of Christian Thought
         (0,1,1,2,16,1,3,),  #New Testament Survey I
         (0,1,0,1,16,1,3,),  #Intro to Computer Programming
         (0,1,0,1,16,1,1,),  #Lab for ^^
         (0,1,1,1,16,1,3,),  #English Comp
         (0,1,0,1,16,1,3,),  #Music Appreciation

         (0,1,1, 2, 15, 2, 3,),  # New Testament Survey II
         (0,1,1, 3, 15, 2, 3,),  # Christian Theo II
         (0,1,1, 1, 15, 2, 3,),  # Spoken Communications
         (0,1,0, 1, 15, 2, 3,),  # Web Design
         (0,1,0, 1, 15, 2, 3,),  # Computer Hardware

         (0,1,1, 3, 18, 3, 3,),  # Christian Theo I
         (0,1,0, 4, 18, 3, 3,),  # Gothic
         (0,1,1, 1, 18, 3, 3,),  # Geology
         (0,1,1, 1, 18, 3, 1,),  # Lab for ^^

         (0,1,0, 3, 15, 4, 6,),  # Land and the Bible
         (0,1,0, 3, 15, 4, 3,),  # Modern hebrew
         (0,1,0, 3, 15, 4, 3,),  # Jewish thought and culture
         (0,1,1, 3, 15, 4, 3,),  # History of Ancient Israel

         #SUBJECT 1
         (1,0,0,2,15,1,3,),  #Accounting Fundamentals I
         (1,0,1,1,15,1,3,),  #Old Testament Survey I
         (1,0,1,1,15,1,3,),  #Essentials of christian Thought
         (1,0,0,2,15,1,3,),  #Microeconomics
         (1,0,1,1,15,1,3,),  #World History I

         (1,0,0, 2, 17, 2, 3,),  # Accounting Fund II
         (1,0,1, 1, 17, 2, 3,),  # World History II
         (1,0,1, 1, 17, 2, 3,),  # Biology
         (1,0,1, 1, 17, 2, 1,), # Lab for Biology
         (1,0,0, 3, 17, 2, 3,),  # Business comunications
         (1,0,1, 1, 17, 2, 3,),  # Music and Art

         (1,0,0, 3, 15, 3, 3,),  # Accounting Professional Responsibilities
         (1,0,1, 3, 15, 3, 3,),  # Christian Theo I
         (1,0,0, 3, 15, 3, 3,),  # Business Law
         (1,0, 1, 2, 15, 3, 3,),  # English Lit I
         (1,0,0, 3, 15, 3, 3,),  # Management Theory

         (1,0,0, 3, 15, 4, 6,),  # Land and the Bible
         (1,0,0, 3, 15, 4, 3,),  # Modern hebrew
         (1,0,0, 3, 15, 4, 3,),  # Jewish thought and culture
         (1,0,1, 3, 15, 4, 3,),  # History of Ancient Israel
         ]

outputs = [(4,),(4,),(4,),(4,),(3.3,),(4,),
          (3.7,),(4,),(4,),(4,),(3.7,),
          (4,),(3.3,),(4,),(3.7,),
          (3.3,),(4,),(4,),(4,),

          (3.7,),(4,),(4,),(3.7,),(4,),
          (3.7,),(3.7,),(4,),(4,),(4,),(4,),
          (4,),(4,),(3.7,),(4,),(3,),
          (3.7,),(3.3,),(4,),(4,),
          ]


# FEATURE SCALING FOR INPUTS
input_dummy =[]
i = 0
while i < 39:
    input_dummy.append([inputs[i][0], inputs[i][1], inputs[i][2], inputs[i][3]
                        , inputs[i][4], inputs[i][5], inputs[i][6]])
    i = i + 1



highests = []
varCounter = 0
while varCounter < 7:
    i = 0
    highest = 0
    while i < 39:
        if input_dummy[i][varCounter] > highest:
            highest = inputs[i][varCounter]
        i = i + 1
    highests.append(highest)
    varCounter = varCounter + 1


varCounter = 0
while varCounter < 7:
    i = 0
    while i < 39:
        input_dummy[i][varCounter] = input_dummy[i][varCounter]/highests[varCounter]
        i = i + 1
    varCounter = varCounter + 1


x = 0
inputs = []
while x < 39:
    dumbledore = input_dummy[x]
    inputs.append(tuple(dumbledore))
    x+=1



def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(inputs, outputs):
            output = net.activate(xi)
            genome.fitness -= ((10*output[0] - 10*xo[0]) ** 2)/(39)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 100 generations.
    winner = p.run(eval_genomes, 100)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(inputs, outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)