import chess
from chess_agent import ChessAgent

def play_game(agent_white: ChessAgent, agent_black: ChessAgent, max_moves=200):
    """
    Simulates a game of chess between two agents.
    Returns:
        (white_score, black_score)
        where 1.0 is win, 0.5 is draw, 0.0 is loss
    """
    board = chess.Board()
    moves_played = 0
    
    while not board.is_game_over() and moves_played < max_moves:
        if board.turn == chess.WHITE:
            move = agent_white.select_move(board)
        else:
            move = agent_black.select_move(board)
            
        if move is None or move not in board.legal_moves:
            break
            
        board.push(move)
        moves_played += 1
        
    # Determine the result
    if board.is_checkmate():
        if board.turn == chess.WHITE: # White is mated
            return 0.0, 1.0
        else: # Black is mated
            return 1.0, 0.0
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0.5, 0.5
    else:
        # If the game reaches max moves, treat as a draw
        # or evaluate by material
        return 0.5, 0.5

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
