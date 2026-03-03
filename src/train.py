import argparse
import os
import neat
import pickle
import sys
import multiprocessing
from tournament import run_tournament

def create_population(config_path):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(5, filename_prefix="models/neat-checkpoint-"))
    return pop, config, stats

def eval_genomes(genomes, config):
    # This acts as the fitness function for the NEAT population
    # the genomes are a list of (genome_id, genome)
    run_tournament(genomes, config, group_size=4, num_workers=4)

def main():
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=50)
    args = parser.parse_args()
    
    config_path = "neat_config.ini"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        sys.exit(1)
        
    os.makedirs("models", exist_ok=True)
    
    pop, config, stats = create_population(config_path)
    
    print(f"Starting training for {args.generations} generations...")
    try:
        winner = pop.run(eval_genomes, n=args.generations)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving best genome so far...")
        winner = pop.best_genome
        
    print(f"\nTraining complete. Best genome fitness: {winner.fitness}")
    
    with open("models/best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
        
    print("Saved best genome to models/best_genome.pkl")

if __name__ == "__main__":
    main()
