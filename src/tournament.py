import numpy as np
import neat
from chess_agent import ChessAgent
from chess_game import play_match
from concurrent.futures import ThreadPoolExecutor


def evaluate_group(group_genomes, config):
    """
    Evaluates a group of genomes via round-robin matches.

    Args:
        group_genomes: list of (genome_id, genome)
        config: NEAT config
    Returns:
        list of scores for each genome in the group
    """
    # Force CPU for tournament play — fast enough with policy network
    agents = [ChessAgent(g, config, device='cpu') for gid, g in group_genomes]
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

    Uses ThreadPoolExecutor for parallelism (avoids multiprocessing
    serialization issues with PyTorch/CUDA).
    """
    # Initialize all fitness to 0
    for genome_id, genome in genomes:
        genome.fitness = 0.0

    remaining = [g for g in genomes]
    np.random.shuffle(remaining)

    round_number = 1

    while len(remaining) > 1:
        groups = []
        for i in range(0, len(remaining), group_size):
            groups.append(remaining[i:i + group_size])

        # Evaluate groups in parallel using threads
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(evaluate_group, g, config)
                for g in groups
            ]
            group_scores = [f.result() for f in futures]

        next_round = []
        for g, scores in zip(groups, group_scores):
            # Assign fitness based on round performance
            for i, (gid, genome) in enumerate(g):
                genome.fitness += (round_number * 10) + scores[i]

            # Standard knockout: top 2 advance per group
            indices = np.argsort(scores)[::-1]
            top_n = 2 if len(g) >= 2 else 1

            for k in range(min(top_n, len(g))):
                next_round.append(g[indices[k]])

        remaining = next_round
        round_number += 1
