

from __future__ import print_function
import os
import neat
import visualize
import math

from sklearn.datasets import load_boston
boston = load_boston()

input_dummy_dumb = boston.data
input_dummy = []

#LIMITING IT TO ONLY THE IMPORTANT VARIABLES SO MAYBE I HAVE A CHANCE
i = 0
while i < 506:
    input_dummy.append([input_dummy_dumb[i][5], input_dummy_dumb[i][10], input_dummy_dumb[i][12]])
    i = i + 1



# FEATURE SCALING FOR INPUTS
highests = []
varCounter = 0
while varCounter < 3:
    i = 0
    highest = 0
    while i < 506:
        if input_dummy[i][varCounter] > highest:
            highest = input_dummy[i][varCounter]
        i = i + 1
    highests.append(highest)
    varCounter = varCounter + 1

i = 0
while i < 506:
    varCounter = 0
    while varCounter < 3:
        i = 0
        while i < 506:
            input_dummy[i][varCounter] = input_dummy[i][varCounter]/highests[varCounter]
            i = i + 1
        varCounter = varCounter + 1


x = 0
boston_inputs = []
while x < 506:
    dumbledore = input_dummy[x]
    boston_inputs.append(tuple(dumbledore))
    x+=1


x = 0
boston_outputs = []
while x < 506:
    dumbledore = boston.target[x]
    bob = (dumbledore,)
    boston_outputs.append(bob)
    x+=1



def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(boston_inputs, boston_outputs):
            output = net.activate(xi)
            genome.fitness -= ((output[0] - xo[0]) ** 2)/(506)


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
    for xi, xo in zip(boston_inputs, boston_outputs):
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