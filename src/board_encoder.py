import chess
import numpy as np

def encode_board(board: chess.Board, perspective: chess.Color = chess.WHITE) -> np.ndarray:
    """
    Encodes a chess.Board into a 768-dimensional binary vector (64 squares x 12 piece types).
    The encoding is always from the given perspective.
    
    If perspective == chess.BLACK:
        The board is flipped vertically, and white/black pieces are swapped,
        so the neural network always thinks it is playing as White.
    """
    vector = np.zeros(768, dtype=np.float32)
    
    # 6 piece types for White, 6 for Black
    # order: P, N, B, R, Q, K (White), then p, n, b, r, q, k (Black)
    # The piece types in python-chess are 1:P, 2:N, 3:B, 4:R, 5:Q, 6:K
    
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            c = piece.color
            pt = piece.piece_type
            
            # if we are evaluating from black's perspective, flip the board vertically and mirror colors
            if perspective == chess.BLACK:
                sq = chess.square_mirror(sq) # rank flip (a1 -> a8)
                c = not c # swap color
                
            color_offset = 0 if c == chess.WHITE else 6
            piece_idx = color_offset + (pt - 1)
            
            # index = square_index * 12 + piece_idx
            idx = sq * 12 + piece_idx
            vector[idx] = 1.0
            
    return vector

def encode_boards_batch(boards, perspective: chess.Color = chess.WHITE) -> np.ndarray:
    """
    Encodes multiple boards into a shape (N, 768) array.
    """
    n = len(boards)
    vectors = np.zeros((n, 768), dtype=np.float32)
    for i, b in enumerate(boards):
        vectors[i] = encode_board(b, perspective)
    return vectors
