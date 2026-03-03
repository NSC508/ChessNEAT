import numpy as np
import neat
from chess_agent import ChessAgent
from chess_game import play_match
import multiprocessing

def evaluate_group(args):
    """
    Evaluates a group of genomes.
    Arguments:
        group_genomes: list of (genome_id, genome)
        config: NEAT config
    """
    group_genomes, config = args
    agents = [ChessAgent(g, config) for gid, g in group_genomes]
    scores = [0.0] * len(agents)
    
    # Round robin: every agent plays against every other agent
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            # 2 games (each player plays White once)
            s1, s2 = play_match(agents[i], agents[j], games=2)
            scores[i] += s1
            scores[j] += s2
            
    return scores

def run_tournament(genomes, config, group_size=4, num_workers=4):
    """
    Runs a knockout tournament where genomes compete in groups.
    Updates the fitness of each genome in-place.
    """
    # initialize all fitness to 0
    for genome_id, genome in genomes:
        genome.fitness = 0.0

    remaining = [g for g in genomes]
    np.random.shuffle(remaining)
    
    round_number = 1
    
    pool = multiprocessing.Pool(num_workers)
    
    while len(remaining) > 1:
        groups = []
        for i in range(0, len(remaining), group_size):
            groups.append(remaining[i:i+group_size])
            
        group_args = [(g, config) for g in groups]
        # Evaluate groups in parallel
        # Note: PyTorch in multiprocessing might need 'spawn' start method
        # which is the default in PyTorch, but we'll see.
        group_scores = pool.map(evaluate_group, group_args)
        
        next_round = []
        for g, scores in zip(groups, group_scores):
            # assign fitness based on round performance
            for i, (gid, genome) in enumerate(g):
                genome.fitness += (round_number * 10) + scores[i]
                
            # Standard knockout: top 2 advance per group
            indices = np.argsort(scores)[::-1]
            # Advance up to top 2, ensure at least one if group size > 0
            if len(g) >= 2:
                top_n = 2
            else:
                top_n = 1
                
            for k in range(min(top_n, len(g))):
                next_round.append(g[indices[k]])
                
        remaining = next_round
        round_number += 1
        
    pool.close()
    pool.join()
