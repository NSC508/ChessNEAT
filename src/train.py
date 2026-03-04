import argparse
import glob
import os
import neat
import pickle
import re
import sys
import time
import threading
import multiprocessing
import numpy as np
from tournament import run_tournament
from chess_game import play_game
from chess_agent import ChessAgent
from board_encoder import encode_board
from move_logger import MoveLogger
from elo_evaluator import run_quick_elo_check, STOCKFISH_PATH

# Global references so the eval callback can access them
_config = None
_generation_counter = 0
_pop = None
_viz_enabled = False


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


def find_latest_checkpoint(checkpoint_dir="models", prefix="neat-checkpoint-"):
    """
    Find the most recent NEAT checkpoint file.
    Returns (path, generation_number) or (None, None) if no checkpoints exist.
    """
    pattern = os.path.join(checkpoint_dir, f"{prefix}*")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None, None

    # Extract generation numbers and find the latest
    best_path, best_gen = None, -1
    for cp in checkpoints:
        basename = os.path.basename(cp)
        # Extract the number after the prefix
        match = re.search(r'(\d+)$', basename)
        if match:
            gen = int(match.group(1))
            if gen > best_gen:
                best_gen = gen
                best_path = cp

    return best_path, best_gen


def restore_population(checkpoint_path, config_path):
    """
    Restore a NEAT population from a checkpoint file.
    Re-attaches reporters since they are not saved with the checkpoint.
    """
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    pop = neat.Checkpointer.restore_checkpoint(checkpoint_path)
    # Re-attach reporters (they are not serialized)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(5, filename_prefix="models/neat-checkpoint-"))
    return pop, config, stats


class VisualizerReporter(neat.reporting.BaseReporter):
    """
    NEAT reporter that hooks into the evolution lifecycle to emit
    visualization events to the training dashboard.
    """

    def __init__(self, config, showcase_every=1, move_delay=0.25,
                 move_logger=None, elo_every=10, stockfish_path=None,
                 elo_games=6):
        self.config = config
        self.showcase_every = showcase_every
        self.move_delay = move_delay
        self.move_logger = move_logger
        self.elo_every = elo_every
        self.stockfish_path = stockfish_path or STOCKFISH_PATH
        self.elo_games = elo_games
        self.generation = 0
        self.elo_history = []
        self._gen_start_time = None

    def start_generation(self, generation):
        self.generation = generation
        self._gen_start_time = time.time()
        try:
            from visualizer import emit_generation_start
            emit_generation_start(generation, pop_size=0)
        except ImportError:
            pass

    def end_generation(self, config, population, species_set):
        """Called at the end of each generation — logs comprehensive stats."""
        gen_elapsed = time.time() - self._gen_start_time if self._gen_start_time else 0.0

        genomes = list(population.values())
        fitnesses = [g.fitness for g in genomes if g.fitness is not None]

        if not fitnesses:
            return

        best_fitness = max(fitnesses)
        avg_fitness = float(np.mean(fitnesses))
        min_fitness = min(fitnesses)
        median_fitness = float(np.median(fitnesses))
        std_fitness = float(np.std(fitnesses))
        num_species = len(species_set.species)

        # Species sizes
        species_sizes = [len(s.members) for s in species_set.species.values()]

        # Find the best genome
        best_genome = max(genomes, key=lambda g: g.fitness if g.fitness is not None else float('-inf'))

        # Network topology stats
        def _genome_complexity(genome):
            nodes = len([n for n in genome.nodes])
            conns = len([c for c in genome.connections.values() if c.enabled])
            return nodes, conns

        best_nodes, best_conns = _genome_complexity(best_genome)
        all_complexity = [_genome_complexity(g) for g in genomes]
        avg_nodes = float(np.mean([c[0] for c in all_complexity]))
        avg_conns = float(np.mean([c[1] for c in all_complexity]))

        # ---- Log stats to JSONL ----
        if self.move_logger is not None:
            self.move_logger.log_generation_stats(
                generation=self.generation,
                best_fitness=best_fitness,
                avg_fitness=avg_fitness,
                min_fitness=min_fitness,
                median_fitness=median_fitness,
                std_fitness=std_fitness,
                num_species=num_species,
                species_sizes=species_sizes,
                best_genome_nodes=best_nodes,
                best_genome_connections=best_conns,
                avg_genome_nodes=avg_nodes,
                avg_genome_connections=avg_conns,
                gen_elapsed_time=round(gen_elapsed, 2),
                population_size=len(genomes),
                best_genome_id=best_genome.key if hasattr(best_genome, 'key') else None,
            )

        # Emit visualization
        try:
            from visualizer import emit_generation_end
            emit_generation_end(
                generation=self.generation,
                best_fitness=best_fitness,
                avg_fitness=avg_fitness,
                num_species=num_species,
                best_genome=best_genome,
                config=config,
            )
        except ImportError:
            pass

        # ---- Showcase game between top 2 genomes ----
        if self.generation % self.showcase_every == 0:
            self._play_showcase_game(genomes, config)

        # ---- Periodic Elo evaluation ----
        if self.elo_every > 0 and self.generation > 0 and self.generation % self.elo_every == 0:
            self._run_elo_evaluation(best_genome, config)

    def _play_showcase_game(self, genomes, config):
        """Play a visualized game between the top 2 genomes, logged to move logger."""
        scored = [(g, g.fitness) for g in genomes if g.fitness is not None]
        scored.sort(key=lambda x: x[1], reverse=True)

        if len(scored) < 2:
            return

        genome1, fit1 = scored[0]
        genome2, fit2 = scored[1]

        agent1 = ChessAgent(genome1, config)
        agent2 = ChessAgent(genome2, config)

        white_name = f"Best (#{self.generation})"
        black_name = f"2nd Best (#{self.generation})"

        # Play one showcase game (genome1 as white)
        board, _ = play_game(
            agent1, agent2,
            max_moves=80,
            visualize=True,
            white_name=white_name,
            black_name=black_name,
            white_fitness=fit1,
            black_fitness=fit2,
            move_delay=self.move_delay,
            return_board=True,
        )

        # Log the game
        if self.move_logger is not None and board is not None:
            # Determine result from board
            if board.is_checkmate():
                result = "0-1" if board.turn == chess.WHITE else "1-0"
            else:
                result = "1/2-1/2"
            self.move_logger.log_game(
                board=board,
                generation=self.generation,
                white_name=white_name,
                black_name=black_name,
                result=result,
                white_fitness=fit1,
                black_fitness=fit2,
                game_type="showcase",
            )

        # Emit network activations
        try:
            import chess
            import torch
            board_state = chess.Board()
            input_vec = encode_board(board_state, chess.WHITE)
            inp = torch.tensor(input_vec, device=agent1.device)

            all_activations = agent1.net.activate_full(inp)
            str_activations = {str(k): v for k, v in all_activations.items()}

            from visualizer import _extract_network, socketio
            net_data = _extract_network(genome1, config)
            net_data["activations"] = str_activations
            socketio.emit("network_activations", net_data)
        except Exception:
            pass

    def _run_elo_evaluation(self, best_genome, config):
        """Run an Elo evaluation against calibrated Stockfish levels."""
        try:
            elo_result = run_quick_elo_check(
                genome=best_genome,
                config=config,
                stockfish_path=self.stockfish_path,
                move_logger=self.move_logger,
                generation=self.generation,
                games_per_level=self.elo_games,
            )
            self.elo_history.append({
                "generation": self.generation,
                "estimated_elo": elo_result["estimated_elo"],
                "results": elo_result["results"],
            })
        except Exception as e:
            print(f"  ⚠️ Elo evaluation failed: {e}")


def eval_genomes(genomes, config):
    """Fitness function for the NEAT population."""
    run_tournament(genomes, config, group_size=4, num_workers=4)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=50,
                        help="Number of generations to train (additional if resuming)")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="Resume from checkpoint. Pass a path or omit for auto-detect")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable training visualization dashboard")
    parser.add_argument("--viz-port", type=int, default=5555,
                        help="Port for the visualization dashboard")
    parser.add_argument("--showcase-every", type=int, default=1,
                        help="Play a showcase game every N generations")
    parser.add_argument("--move-delay", type=float, default=0.25,
                        help="Delay between moves in showcase games (seconds)")
    parser.add_argument("--elo-every", type=int, default=10,
                        help="Run Elo evaluation every N generations (0 to disable)")
    parser.add_argument("--elo-games", type=int, default=6,
                        help="Number of games per Elo level during evaluation")
    parser.add_argument("--stockfish-path", type=str, default=None,
                        help="Path to Stockfish binary")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory for move logs and Elo evaluations")
    args = parser.parse_args()

    config_path = "neat_config.ini"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        sys.exit(1)

    os.makedirs("models", exist_ok=True)

    # ---- Initialize move logger ----
    move_logger = MoveLogger(log_dir=args.log_dir)
    print(f"📝 Move logging to: {os.path.abspath(args.log_dir)}")

    # ---- Create or restore population ----
    resumed_gen = 0
    if args.resume is not None:
        if args.resume == "auto":
            cp_path, cp_gen = find_latest_checkpoint()
            if cp_path is None:
                print("⚠️  No checkpoint found — starting fresh.")
                pop, config, stats = create_population(config_path)
            else:
                print(f"🔄 Resuming from checkpoint: {cp_path} (generation {cp_gen})")
                pop, config, stats = restore_population(cp_path, config_path)
                resumed_gen = cp_gen
        else:
            # Explicit checkpoint path
            if not os.path.exists(args.resume):
                print(f"Error: checkpoint '{args.resume}' not found.")
                sys.exit(1)
            match = re.search(r'(\d+)$', args.resume)
            resumed_gen = int(match.group(1)) if match else 0
            print(f"🔄 Resuming from checkpoint: {args.resume} (generation {resumed_gen})")
            pop, config, stats = restore_population(args.resume, config_path)
    else:
        pop, config, stats = create_population(config_path)

    # ---- Start visualization dashboard ----
    if not args.no_viz:
        try:
            from visualizer import start_server
            start_server(port=args.viz_port)
        except ImportError:
            print("⚠️ Visualizer not available, continuing without dashboard.")

    viz_reporter = VisualizerReporter(
        config,
        showcase_every=args.showcase_every,
        move_delay=args.move_delay,
        move_logger=move_logger,
        elo_every=args.elo_every,
        stockfish_path=args.stockfish_path,
        elo_games=args.elo_games,
    )
    pop.add_reporter(viz_reporter)

    if resumed_gen > 0:
        total_target = resumed_gen + args.generations
        print(f"▶️  Continuing training for {args.generations} more generations "
              f"(gen {resumed_gen} → {total_target})...")
    else:
        print(f"Starting training for {args.generations} generations...")
    if args.elo_every > 0:
        print(f"🏆 Elo evaluation every {args.elo_every} generations "
              f"({args.elo_games} games/level)")

    try:
        winner = pop.run(eval_genomes, n=args.generations)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving best genome so far...")
        winner = pop.best_genome
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        winner = pop.best_genome

    if winner is None:
        print("⚠️  No completed generations — nothing to save.")
        sys.exit(1)

    final_gen = resumed_gen + args.generations
    print(f"\nTraining complete. Best genome fitness: {winner.fitness}")

    with open("models/best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Saved best genome to models/best_genome.pkl")

    # ---- Final Elo evaluation ----
    print("\n🏆 Running final Elo evaluation...")
    final_elo = run_quick_elo_check(
        genome=winner,
        config=config,
        stockfish_path=args.stockfish_path,
        move_logger=move_logger,
        generation=final_gen,
        games_per_level=max(args.elo_games, 10),  # more games for final eval
    )

    # ---- Print summary ----
    print(f"\n{'='*60}")
    print(f"  📊 TRAINING SUMMARY")
    print(f"{'='*60}")
    if resumed_gen > 0:
        print(f"  Resumed from gen:      {resumed_gen}")
    print(f"  Total generations:     {final_gen}")
    print(f"  Best fitness:          {winner.fitness}")
    print(f"  Total games logged:    {move_logger.total_games}")
    print(f"  Final estimated Elo:   {final_elo['estimated_elo']}")
    print(f"  Elo evaluations:       {len(viz_reporter.elo_history) + 1}")

    if viz_reporter.elo_history:
        print(f"\n  Elo Progression:")
        for entry in viz_reporter.elo_history:
            print(f"    Gen {entry['generation']:4d}: {entry['estimated_elo']} Elo")
        print(f"    Gen {final_gen:4d}: {final_elo['estimated_elo']} Elo (final)")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
