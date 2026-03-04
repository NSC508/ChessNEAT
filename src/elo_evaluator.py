"""
Elo Evaluator for ChessNEAT.

Uses Stockfish at various UCI_Elo levels to evaluate the strength
of NEAT agents. The agent plays multiple games at each level,
and we estimate its Elo using a binary search / ladder approach.

Stockfish UCI_LimitStrength supports Elo range: 1320 - 3190.
For sub-1320 Elo, we use a random-move bot as the floor.

Elo Ladder (benchmarking levels):
    Level 0:  Random moves          (~400 Elo)
    Level 1:  Stockfish Elo 1320    (Beginner)
    Level 2:  Stockfish Elo 1400    (Novice)
    Level 3:  Stockfish Elo 1500    (Intermediate)
    Level 4:  Stockfish Elo 1600    (Club player)
    Level 5:  Stockfish Elo 1700    (Intermediate club)
    Level 6:  Stockfish Elo 1800    (Strong club)
    Level 7:  Stockfish Elo 1900    (Expert)
    Level 8:  Stockfish Elo 2000    (Candidate Master)
    Level 9:  Stockfish Elo 2200    (National Master)
"""

import os
import chess
import chess.engine
import random
import numpy as np
from chess_agent import ChessAgent
from board_encoder import encode_board

# Path to stockfish binary - relative to project root
STOCKFISH_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "engines", "stockfish_binary"
)

# Elo ladder definition
ELO_LADDER = [
    {"level": 0, "name": "Random",        "elo": 400,  "engine": "random"},
    {"level": 1, "name": "Beginner",       "elo": 1320, "engine": "stockfish"},
    {"level": 2, "name": "Novice",         "elo": 1400, "engine": "stockfish"},
    {"level": 3, "name": "Intermediate",   "elo": 1500, "engine": "stockfish"},
    {"level": 4, "name": "Club Player",    "elo": 1600, "engine": "stockfish"},
    {"level": 5, "name": "Strong Club",    "elo": 1700, "engine": "stockfish"},
    {"level": 6, "name": "Advanced",       "elo": 1800, "engine": "stockfish"},
    {"level": 7, "name": "Expert",         "elo": 1900, "engine": "stockfish"},
    {"level": 8, "name": "Candidate Master", "elo": 2000, "engine": "stockfish"},
    {"level": 9, "name": "National Master",  "elo": 2200, "engine": "stockfish"},
]


class RandomEngine:
    """A random-move 'engine' that acts as the Elo floor (~400)."""

    def select_move(self, board):
        legal = list(board.legal_moves)
        return random.choice(legal) if legal else None


class StockfishOpponent:
    """
    Wraps Stockfish UCI engine at a specific Elo level.
    Uses python-chess's engine interface.
    """

    def __init__(self, elo, stockfish_path=None, move_time=0.05):
        self.elo = elo
        self.move_time = move_time
        path = stockfish_path or STOCKFISH_PATH

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Stockfish binary not found at {path}. "
                f"Build it: cd engines/stockfish_src/src && make build ARCH=x86-64-avx2"
            )

        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure({
            "UCI_LimitStrength": True,
            "UCI_Elo": elo,
            "Threads": 1,
            "Hash": 16,
        })

    def select_move(self, board):
        result = self.engine.play(board, chess.engine.Limit(time=self.move_time))
        return result.move

    def close(self):
        self.engine.quit()

    def __del__(self):
        try:
            self.engine.quit()
        except Exception:
            pass


def play_evaluation_game(neat_agent, opponent, neat_color=chess.WHITE, max_moves=200):
    """
    Play one evaluation game between the NEAT agent and an opponent.

    Args:
        neat_agent: ChessAgent instance
        opponent: object with select_move(board) method
        neat_color: chess.WHITE or chess.BLACK
        max_moves: maximum moves before declaring draw

    Returns:
        board: final board state (with move stack for PGN)
        result: "1-0", "0-1", or "1/2-1/2"
        neat_score: 1.0 for win, 0.5 for draw, 0.0 for loss
    """
    board = chess.Board()
    moves_played = 0

    while not board.is_game_over() and moves_played < max_moves:
        if board.turn == neat_color:
            move = neat_agent.select_move(board)
        else:
            move = opponent.select_move(board)

        if move is None or move not in board.legal_moves:
            # Illegal move = loss for that side
            if board.turn == neat_color:
                result = "0-1" if neat_color == chess.WHITE else "1-0"
                return board, result, 0.0
            else:
                result = "1-0" if neat_color == chess.WHITE else "0-1"
                return board, result, 1.0

        board.push(move)
        moves_played += 1

    # Determine result
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            result = "0-1"
        else:
            result = "1-0"
    else:
        result = "1/2-1/2"

    # Compute NEAT agent's score
    if result == "1/2-1/2":
        neat_score = 0.5
    elif (result == "1-0" and neat_color == chess.WHITE) or \
         (result == "0-1" and neat_color == chess.BLACK):
        neat_score = 1.0
    else:
        neat_score = 0.0

    return board, result, neat_score


def evaluate_against_level(neat_agent, level_info, games_per_match=10,
                           stockfish_path=None, move_logger=None, generation=0):
    """
    Play multiple games against a specific Elo level.

    Returns:
        dict with wins, losses, draws, score, and game boards
    """
    is_random = level_info["engine"] == "random"

    if is_random:
        opponent = RandomEngine()
    else:
        opponent = StockfishOpponent(
            elo=level_info["elo"],
            stockfish_path=stockfish_path,
        )

    wins, losses, draws = 0, 0, 0
    game_boards = []

    try:
        for i in range(games_per_match):
            # Alternate colors
            neat_color = chess.WHITE if i % 2 == 0 else chess.BLACK

            board, result, score = play_evaluation_game(
                neat_agent, opponent, neat_color=neat_color
            )

            if score == 1.0:
                wins += 1
            elif score == 0.0:
                losses += 1
            else:
                draws += 1

            game_boards.append({
                "board": board,
                "result": result,
                "neat_color": "white" if neat_color == chess.WHITE else "black",
                "moves": len(board.move_stack),
            })

            # Log the game
            if move_logger is not None:
                white_name = f"NEAT Gen-{generation}" if neat_color == chess.WHITE else \
                    f"{level_info['name']} ({level_info['elo']})"
                black_name = f"{level_info['name']} ({level_info['elo']})" if neat_color == chess.WHITE else \
                    f"NEAT Gen-{generation}"

                move_logger.log_game(
                    board=board,
                    generation=generation,
                    white_name=white_name,
                    black_name=black_name,
                    result=result,
                    game_type="elo_eval",
                )

    finally:
        if not is_random and hasattr(opponent, 'close'):
            opponent.close()

    total = wins + losses + draws
    score_pct = (wins + 0.5 * draws) / total if total > 0 else 0.0

    return {
        "level": level_info["level"],
        "level_name": level_info["name"],
        "opponent_elo": level_info["elo"],
        "games_played": total,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": score_pct,
        "game_boards": game_boards,
    }


def estimate_elo(neat_agent, games_per_level=10, max_level=None,
                 stockfish_path=None, move_logger=None, generation=0):
    """
    Run the NEAT agent through the Elo ladder, starting from the bottom.
    Stops when the agent scores < 30% against a level.

    Uses the performance rating formula:
        If score > 0 and score < 1:
            Elo_estimated = opponent_elo + 400 * log10(score / (1 - score))

    Returns:
        dict with estimated_elo and per-level results
    """
    results = []
    estimated_elo = 400  # default floor

    ladder = ELO_LADDER
    if max_level is not None:
        ladder = [l for l in ladder if l["level"] <= max_level]

    for level_info in ladder:
        print(f"  ⚔️  Testing vs {level_info['name']} (Elo {level_info['elo']})...")

        level_result = evaluate_against_level(
            neat_agent, level_info,
            games_per_match=games_per_level,
            stockfish_path=stockfish_path,
            move_logger=move_logger,
            generation=generation,
        )

        # Remove board objects for serialization
        level_result_clean = {k: v for k, v in level_result.items() if k != "game_boards"}
        results.append(level_result_clean)

        score = level_result["score"]
        print(f"    Result: +{level_result['wins']} ={level_result['draws']} "
              f"-{level_result['losses']} (score: {score:.1%})")

        # Estimate Elo from this matchup using performance rating
        if 0 < score < 1:
            performance = level_info["elo"] + 400 * np.log10(score / (1 - score))
            estimated_elo = max(estimated_elo, performance)
        elif score >= 1.0:
            # Won all games - Elo is at least 400 above opponent
            estimated_elo = max(estimated_elo, level_info["elo"] + 400)
        else:
            # Lost all games - can't compute, but we know it's below
            pass

        # Stop if agent is getting crushed (< 30% score)
        if score < 0.30 and level_info["level"] > 0:
            print(f"    ⛔ Agent scored below 30% — stopping ladder climb.")
            break

    estimated_elo = round(estimated_elo)

    elo_data = {
        "estimated_elo": estimated_elo,
        "levels_tested": len(results),
        "results": results,
    }

    # Log the Elo evaluation
    if move_logger is not None:
        move_logger.log_elo_evaluation(generation, elo_data)

    print(f"  📊 Estimated Elo: {estimated_elo}")
    return elo_data


def run_quick_elo_check(genome, config, stockfish_path=None,
                        move_logger=None, generation=0, games_per_level=6):
    """
    Convenience function to run an Elo evaluation on a single genome.
    Creates the ChessAgent internally.
    """
    agent = ChessAgent(genome, config)

    print(f"\n{'='*50}")
    print(f"  🏆 Elo Evaluation — Generation {generation}")
    print(f"{'='*50}")

    result = estimate_elo(
        agent,
        games_per_level=games_per_level,
        stockfish_path=stockfish_path,
        move_logger=move_logger,
        generation=generation,
    )

    print(f"{'='*50}\n")
    return result
