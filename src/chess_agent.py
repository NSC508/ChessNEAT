import torch
import numpy as np
import chess
from neat.graphs import feed_forward_layers
from board_encoder import encode_board

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
        result = feed_forward_layers(input_nodes, output_nodes, conn_tuples)
        # feed_forward_layers may return (layers, required) or just layers
        if isinstance(result, tuple):
            layers, _ = result
        else:
            layers = result
        
        self.node_to_idx = {n: i for i, n in enumerate(input_nodes)}
        
        # Each layer is a set of node ids — flatten them
        req_nodes = set()
        for l in layers:
            for n in l:
                req_nodes.add(n)
        for n in req_nodes:
            if n not in self.node_to_idx:
                self.node_to_idx[n] = len(self.node_to_idx)
                
        self.total_nodes = len(self.node_to_idx)
        self.input_size = len(input_nodes)
        self.output_idx = [self.node_to_idx[n] for n in output_nodes if n in self.node_to_idx]
        
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
            
    def activate(self, inputs):
        """
        inputs: (batch_size, num_inputs) tensor or (num_inputs,) tensor
        returns: (batch_size, num_outputs) tensor
        """
        with torch.no_grad():
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
            batch_size = inputs.shape[0]
            values = torch.zeros((batch_size, self.total_nodes), device=self.device)
            values[:, :self.input_size] = inputs
            
            for src_indices, dst_indices, W, b in self.layers:
                src_vals = values[:, src_indices]
                out = torch.matmul(src_vals, W.t()) + b
                values[:, dst_indices] = torch.tanh(out)
                
            return values[:, self.output_idx]

    def activate_full(self, inputs):
        """
        Like activate(), but returns activations for ALL nodes (not just outputs).
        Returns a dict mapping NEAT node ID -> activation value (for a single input).
        Useful for visualization.
        """
        with torch.no_grad():
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
            values = torch.zeros((1, self.total_nodes), device=self.device)
            values[:, :self.input_size] = inputs
            
            for src_indices, dst_indices, W, b in self.layers:
                src_vals = values[:, src_indices]
                out = torch.matmul(src_vals, W.t()) + b
                values[:, dst_indices] = torch.tanh(out)
            
            # Build reverse mapping: idx -> neat node id
            idx_to_node = {v: k for k, v in self.node_to_idx.items()}
            activations = {}
            vals_cpu = values[0].cpu().numpy()
            for idx, node_id in idx_to_node.items():
                activations[node_id] = float(vals_cpu[idx])
            
            return activations


class ChessAgent:
    """
    Policy-network chess agent.
    
    The network has 128 outputs:
      - outputs[0:64]  = "from square" scores  (one per square a1..h8)
      - outputs[64:128] = "to square" scores    (one per square a1..h8)
    
    Move selection: ONE forward pass on the current board, then score
    each legal move as:  from_score[move.from_sq] + to_score[move.to_sq]
    Pick the legal move with the highest combined score.
    
    This is ~30x faster than the old value-network approach which required
    a separate forward pass for every legal move.
    """
    
    def __init__(self, genome, config, device='cuda'):
        self.net = TorchNEATNetwork(genome, config, device=device)
        self.device = self.net.device
        self.num_outputs = config.genome_config.num_outputs
        
    def select_move(self, board):
        """
        Single forward pass → pick the best legal move.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        # Encode current board from the perspective of the side to move
        color = board.turn
        board_vec = encode_board(board, perspective=color)
        inputs = torch.tensor(board_vec, device=self.device)
        
        # ONE forward pass
        outputs = self.net.activate(inputs).cpu().numpy().flatten()
        
        # Split into from-square and to-square scores
        # outputs[0:64] = from-square scores, outputs[64:128] = to-square scores
        from_scores = outputs[:64]
        to_scores = outputs[64:128] if len(outputs) >= 128 else np.zeros(64)
        
        # Score each legal move
        best_move = None
        best_score = float('-inf')
        
        for move in legal_moves:
            from_sq = move.from_square
            to_sq = move.to_square
            
            # If perspective is black, we need to mirror squares since
            # the board encoding flips the board for black
            if color == chess.BLACK:
                from_sq = chess.square_mirror(from_sq)
                to_sq = chess.square_mirror(to_sq)
            
            score = from_scores[from_sq] + to_scores[to_sq]
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
