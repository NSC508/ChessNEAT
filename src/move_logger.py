"""
Move Logger for ChessNEAT.

Records all games played by the best agents in PGN format,
along with structured JSON metadata for analysis.
"""

import os
import json
import chess
import chess.pgn
import io
from datetime import datetime


class MoveLogger:
    """
    Logs chess games during training in both PGN and JSON formats.
    
    Directory structure:
        logs/
          games/
            gen_000/
              game_001.pgn
              ...
            gen_001/
              ...
          game_log.jsonl      # One JSON line per game (structured)
          elo_log.jsonl        # Elo evaluation results
          summary.json         # Running summary statistics
    """

    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.games_dir = os.path.join(log_dir, "games")
        self.game_log_path = os.path.join(log_dir, "game_log.jsonl")
        self.elo_log_path = os.path.join(log_dir, "elo_log.jsonl")
        self.stats_log_path = os.path.join(log_dir, "training_stats.jsonl")
        self.summary_path = os.path.join(log_dir, "summary.json")

        os.makedirs(self.games_dir, exist_ok=True)

        # Running stats
        self.total_games = 0
        self.games_by_generation = {}
        self._load_summary()

    def _load_summary(self):
        """Load existing summary if resuming training."""
        if os.path.exists(self.summary_path):
            try:
                with open(self.summary_path, "r") as f:
                    data = json.load(f)
                self.total_games = data.get("total_games", 0)
                self.games_by_generation = data.get("games_by_generation", {})
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_summary(self):
        """Save running summary."""
        with open(self.summary_path, "w") as f:
            json.dump({
                "total_games": self.total_games,
                "games_by_generation": self.games_by_generation,
                "last_updated": datetime.now().isoformat(),
            }, f, indent=2)

    def log_game(self, board, generation, white_name, black_name,
                 result, white_fitness=None, black_fitness=None,
                 game_type="showcase", move_evaluations=None):
        """
        Log a completed game.

        Args:
            board: chess.Board with the final game state (with move stack)
            generation: current generation number
            white_name: name/id of white player
            black_name: name/id of black player
            result: game result string ("1-0", "0-1", "1/2-1/2")
            white_fitness: optional fitness of white player
            black_fitness: optional fitness of black player
            game_type: "showcase", "elo_eval", or "tournament"
            move_evaluations: optional list of eval scores per move
        """
        gen_dir = os.path.join(self.games_dir, f"gen_{generation:04d}")
        os.makedirs(gen_dir, exist_ok=True)

        gen_key = str(generation)
        game_num = self.games_by_generation.get(gen_key, 0) + 1
        self.games_by_generation[gen_key] = game_num
        self.total_games += 1

        # ---- Save PGN ----
        pgn_game = chess.pgn.Game()
        pgn_game.headers["Event"] = f"ChessNEAT Training Gen {generation}"
        pgn_game.headers["Site"] = "ChessNEAT"
        pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        pgn_game.headers["Round"] = str(game_num)
        pgn_game.headers["White"] = white_name
        pgn_game.headers["Black"] = black_name
        pgn_game.headers["Result"] = result

        # Reconstruct move sequence from board
        moves = list(board.move_stack)
        node = pgn_game
        for move in moves:
            node = node.add_variation(move)

        pgn_path = os.path.join(gen_dir, f"game_{game_num:04d}.pgn")
        with open(pgn_path, "w") as f:
            print(pgn_game, file=f)

        # ---- Append to JSONL game log ----
        move_sans = []
        replay = chess.Board()
        for move in moves:
            move_sans.append(replay.san(move))
            replay.push(move)

        game_record = {
            "timestamp": datetime.now().isoformat(),
            "generation": generation,
            "game_number": game_num,
            "game_type": game_type,
            "white": white_name,
            "black": black_name,
            "result": result,
            "total_moves": len(moves),
            "moves": move_sans,
            "white_fitness": white_fitness,
            "black_fitness": black_fitness,
            "final_fen": board.fen(),
            "pgn_path": pgn_path,
        }

        if move_evaluations:
            game_record["evaluations"] = move_evaluations

        with open(self.game_log_path, "a") as f:
            f.write(json.dumps(game_record) + "\n")

        self._save_summary()
        return pgn_path

    def log_generation_stats(self, generation, best_fitness, avg_fitness,
                             min_fitness, median_fitness, std_fitness,
                             num_species, species_sizes,
                             best_genome_nodes, best_genome_connections,
                             avg_genome_nodes, avg_genome_connections,
                             gen_elapsed_time, population_size,
                             best_genome_id=None, extra=None):
        """
        Log per-generation training statistics for research paper data.

        This captures everything needed to build fitness curves, species
        dynamics plots, network complexity charts, and timing analysis.
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "generation": generation,
            "population_size": population_size,
            "fitness": {
                "best": best_fitness,
                "avg": avg_fitness,
                "min": min_fitness,
                "median": median_fitness,
                "std": std_fitness,
            },
            "species": {
                "count": num_species,
                "sizes": species_sizes,
            },
            "network_topology": {
                "best_nodes": best_genome_nodes,
                "best_connections": best_genome_connections,
                "avg_nodes": avg_genome_nodes,
                "avg_connections": avg_genome_connections,
            },
            "timing": {
                "generation_seconds": gen_elapsed_time,
            },
            "best_genome_id": best_genome_id,
        }
        if extra:
            record.update(extra)

        with open(self.stats_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        return record

    def log_elo_evaluation(self, generation, elo_results):
        """
        Log Elo evaluation results.

        Args:
            generation: generation number
            elo_results: dict with Elo evaluation data, e.g.:
                {
                    "estimated_elo": 1200,
                    "games": [
                        {"opponent_elo": 800, "wins": 3, "losses": 1, "draws": 1},
                        ...
                    ]
                }
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "generation": generation,
            **elo_results,
        }

        with open(self.elo_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        return record

    def get_elo_history(self):
        """Load and return all Elo evaluations as a list."""
        if not os.path.exists(self.elo_log_path):
            return []
        results = []
        with open(self.elo_log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results

    def get_game_history(self, generation=None, game_type=None):
        """Load and return game records, optionally filtered."""
        if not os.path.exists(self.game_log_path):
            return []
        results = []
        with open(self.game_log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if generation is not None and record["generation"] != generation:
                    continue
                if game_type is not None and record["game_type"] != game_type:
                    continue
                results.append(record)
        return results
