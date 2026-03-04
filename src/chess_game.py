import chess
import time
from chess_agent import ChessAgent

def play_game(agent_white: ChessAgent, agent_black: ChessAgent, max_moves=80, 
              visualize=False, white_name="White", black_name="Black",
              white_fitness=None, black_fitness=None, move_delay=0.3,
              return_board=False):
    """
    Simulates a game of chess between two agents.
    Returns:
        (white_score, black_score)
        where 1.0 is win, 0.5 is draw, 0.0 is loss
        
    If visualize=True, emits move-by-move updates to the training dashboard.
    """
    board = chess.Board()
    moves_played = 0
    
    # Lazy import to avoid circular deps if visualizer isn't used
    emit_fn = None
    emit_result_fn = None
    emit_activations_fn = None
    if visualize:
        try:
            from visualizer import emit_game_state, emit_game_result, emit_network_activations
            emit_fn = emit_game_state
            emit_result_fn = emit_game_result
            emit_activations_fn = emit_network_activations
        except ImportError:
            visualize = False
    
    while not board.is_game_over() and moves_played < max_moves:
        if board.turn == chess.WHITE:
            move = agent_white.select_move(board)
        else:
            move = agent_black.select_move(board)
            
        if move is None or move not in board.legal_moves:
            break
        
        # Get SAN before pushing
        san = board.san(move)
        board.push(move)
        moves_played += 1
        
        if visualize and emit_fn is not None:
            emit_fn(
                fen=board.fen(),
                move_san=san,
                move_number=moves_played,
                white_name=white_name,
                black_name=black_name,
                white_fitness=white_fitness,
                black_fitness=black_fitness,
            )
            # Small delay so the frontend can animate the move
            time.sleep(move_delay)
        
    # Determine the result
    if board.is_checkmate():
        if board.turn == chess.WHITE: # White is mated
            result = "0-1"
            scores = (0.0, 1.0)
        else: # Black is mated
            result = "1-0"
            scores = (1.0, 0.0)
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        result = "1/2-1/2"
        scores = (0.5, 0.5)
    else:
        result = "1/2-1/2"
        scores = (0.5, 0.5)
    
    if visualize and emit_result_fn is not None:
        emit_result_fn(
            result=result,
            white_name=white_name,
            black_name=black_name,
            total_moves=moves_played,
        )
    
    if return_board:
        return board, scores
    return scores

def play_match(agent1: ChessAgent, agent2: ChessAgent, games=2):
    """
    Plays an even number of games, alternating colors.
    Returns: total_score1, total_score2
    """
    score1, score2 = 0.0, 0.0
    for i in range(games):
        if i % 2 == 0:
            # agent1 plays White
            s1, s2 = play_game(agent1, agent2)
            score1 += s1
            score2 += s2
        else:
            # agent2 plays White
            s2, s1 = play_game(agent2, agent1)
            score1 += s1
            score2 += s2
    return score1, score2
