import torch
import numpy as np
import chess
from neat.graphs import feed_forward_layers
from board_encoder import encode_board, encode_boards_batch

class TorchNEATNetwork:
    """
    A PyTorch implementation of a NEAT feed-forward network to allow fast
    batched evaluation on GPUs.
    """
    def __init__(self, genome, config, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # input nodes are negative: -1, -2, ..., -num_inputs
        input_nodes = [-i - 1 for i in range(config.genome_config.num_inputs)]
        output_nodes = [i for i in range(config.genome_config.num_outputs)]
        
        connections = [cg for cg in genome.connections.values() if cg.enabled]
        conn_tuples = [cg.key for cg in connections]
        layers = feed_forward_layers(input_nodes, output_nodes, conn_tuples)
        
        self.node_to_idx = {n: i for i, n in enumerate(input_nodes)}
        
        req_nodes = set(n for l in layers for n in l)
        for n in req_nodes:
            if n not in self.node_to_idx:
                self.node_to_idx[n] = len(self.node_to_idx)
                
        self.total_nodes = len(self.node_to_idx)
        self.input_size = len(input_nodes)
        self.output_idx = [self.node_to_idx[n] for n in output_nodes]
        
        self.layers = []
        for layer in layers:
            incoming = []
            for node in layer:
                incoming.append([(c.key[0], c.weight) for c in connections if c.key[1] == node])
                
            src_nodes = set([src for node_incoming in incoming for src, w in node_incoming])
            src_indices = [self.node_to_idx[n] for n in src_nodes]
            dst_indices = [self.node_to_idx[n] for n in layer]
            
            W = torch.zeros((len(layer), len(src_indices)), device=self.device)
            b = torch.zeros((len(layer), ), device=self.device)
            
            for i, node in enumerate(layer):
                b[i] = genome.nodes[node].bias if node in genome.nodes else 0.0
                node_incoming = incoming[i]
                for src, weight in node_incoming:
                    j = src_indices.index(self.node_to_idx[src])
                    W[i, j] = weight
                    
            self.layers.append((src_indices, dst_indices, W, b))
            
    @torch.no_grad()
    def activate(self, inputs):
        """
        inputs: (batch_size, num_inputs) tensor
        returns: (batch_size, num_outputs) tensor
        """
        batch_size = inputs.shape[0]
        values = torch.zeros((batch_size, self.total_nodes), device=self.device)
        values[:, :self.input_size] = inputs
        
        for src_indices, dst_indices, W, b in self.layers:
            src_vals = values[:, src_indices]
            out = torch.matmul(src_vals, W.t()) + b
            values[:, dst_indices] = torch.tanh(out) # Since config uses tanh
            
        return values[:, self.output_idx]


class ChessAgent:
    def __init__(self, genome, config, device='cuda'):
        self.net = TorchNEATNetwork(genome, config, device=device)
        self.device = self.net.device
        
    def evaluate_positions(self, boards, perspective=chess.WHITE):
        """
        Batch evaluate multiple chess.Board objects.
        """
        if not boards:
            return np.array([])
            
        vectors = encode_boards_batch(boards, perspective)
        inputs = torch.tensor(vectors, device=self.device)
        outputs = self.net.activate(inputs).cpu().numpy()
        return outputs.flatten()
        
    def select_move(self, board):
        """
        Simulates all legal moves, evaluates the resulting positions,
        and returns the one with the highest score.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
            
        next_boards = []
        for move in legal_moves:
            b = board.copy()
            b.push(move)
            next_boards.append(b)
            
        # The agent evaluates from its own perspective.
        # If the agent is playing as White, it wants to maximize the score
        # on the *resulting* board.
        # However, the resulting board is evaluated from the perspective
        # of the *current* player of that board (which would be the opponent).
        # Actually, let's always evaluate the board from the agent's absolute color.
        
        color = board.turn # The agent is playing its turn right now
        
        scores = self.evaluate_positions(next_boards, perspective=color)
        
        # We want the max score.
        best_idx = np.argmax(scores)
        return legal_moves[best_idx]
