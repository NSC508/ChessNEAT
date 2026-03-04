"""
Training Visualizer for ChessNEAT.

Runs a Flask-SocketIO server that serves a live dashboard showing:
  1. The chess board with the current showcase game
  2. The NEAT neural network topology with active node highlighting
  3. Training statistics (generation, fitness, species, etc.)
"""

import threading
import json
import os
import chess
import numpy as np
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO
from neat.graphs import feed_forward_layers

# ---------------------------------------------------------------------------
# Flask / SocketIO setup
# ---------------------------------------------------------------------------
template_dir = os.path.join(os.path.dirname(__file__), "viz_templates")
static_dir = os.path.join(os.path.dirname(__file__), "viz_static")

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.config["SECRET_KEY"] = "chessneat-viz"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


@app.route("/")
def index():
    return render_template("dashboard.html")


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_server_thread = None
_lock = threading.Lock()


def start_server(port=5555):
    """Start the visualizer web server in a background daemon thread."""
    global _server_thread
    if _server_thread is not None:
        return  # already running

    def _run():
        socketio.run(app, host="0.0.0.0", port=port, allow_unsafe_werkzeug=True)

    _server_thread = threading.Thread(target=_run, daemon=True)
    _server_thread.start()
    print(f"\n{'='*60}")
    print(f"  🎯 Training Dashboard:  http://localhost:{port}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Emit helpers (called from training code)
# ---------------------------------------------------------------------------

def emit_generation_start(generation, pop_size):
    """Called at the start of each generation."""
    socketio.emit("generation_start", {
        "generation": generation,
        "pop_size": pop_size,
    })


def emit_generation_end(generation, best_fitness, avg_fitness, num_species, best_genome=None, config=None):
    """Called at the end of each generation with stats."""
    payload = {
        "generation": generation,
        "best_fitness": float(best_fitness),
        "avg_fitness": float(avg_fitness),
        "num_species": num_species,
    }

    # If we have the best genome, extract its network topology
    if best_genome is not None and config is not None:
        payload["network"] = _extract_network(best_genome, config)

    socketio.emit("generation_end", payload)


def emit_game_state(fen, move_san, move_number, white_name, black_name, white_fitness=None, black_fitness=None):
    """Called for each move in a showcase game."""
    socketio.emit("game_state", {
        "fen": fen,
        "move_san": move_san,
        "move_number": move_number,
        "white_name": white_name,
        "black_name": black_name,
        "white_fitness": float(white_fitness) if white_fitness is not None else None,
        "black_fitness": float(black_fitness) if black_fitness is not None else None,
    })


def emit_game_result(result, white_name, black_name, total_moves):
    """Called when a showcase game ends."""
    socketio.emit("game_result", {
        "result": result,
        "white_name": white_name,
        "black_name": black_name,
        "total_moves": total_moves,
    })


def emit_network_activations(genome, config, input_values, output_values):
    """
    Emit the network structure *with* activation values for each node,
    so the frontend can colour nodes by intensity.
    """
    net_data = _extract_network(genome, config)

    # Map input node ids to their activation values
    input_nodes = net_data["input_nodes"]
    activations = {}
    for i, nid in enumerate(input_nodes):
        if i < len(input_values):
            activations[str(nid)] = float(input_values[i])

    # Output
    output_nodes = net_data["output_nodes"]
    for i, nid in enumerate(output_nodes):
        if i < len(output_values):
            activations[str(nid)] = float(output_values[i])

    # Hidden nodes — we can compute them here by forward-passing
    # Actually for simplicity, we mark hidden nodes using the TorchNEATNetwork
    # The frontend will re-use the last provided activations

    net_data["activations"] = activations
    socketio.emit("network_activations", net_data)


# ---------------------------------------------------------------------------
# Network topology extraction
# ---------------------------------------------------------------------------

def _extract_network(genome, config):
    """
    Extract the NEAT genome's network topology into a JSON-friendly dict
    that the frontend D3.js visualisation can render.
    """
    input_nodes = [-i - 1 for i in range(config.genome_config.num_inputs)]
    output_nodes = [i for i in range(config.genome_config.num_outputs)]

    connections = [cg for cg in genome.connections.values() if cg.enabled]
    conn_tuples = [cg.key for cg in connections]

    try:
        result = feed_forward_layers(input_nodes, output_nodes, conn_tuples)
        layers = result[0] if isinstance(result, tuple) else result
    except Exception:
        layers = []

    # Collect all hidden nodes
    hidden_nodes = set()
    for layer in layers:
        for n in layer:
            if n not in input_nodes and n not in output_nodes:
                hidden_nodes.add(n)

    # Build nodes list.  For the 768-input network, we'll summarize
    # inputs into "groups" so the visualisation isn't overwhelmed.
    # Group inputs by piece type: 64 squares x 12 piece channels = 768
    piece_labels = ["♙", "♘", "♗", "♖", "♕", "♔", "♟", "♞", "♝", "♜", "♛", "♚"]
    input_groups = []
    for p in range(12):
        input_groups.append({
            "id": f"input_group_{p}",
            "label": piece_labels[p],
            "type": "input_group",
            "piece_index": p,
            "node_ids": [input_nodes[sq * 12 + p] for sq in range(64)],
        })

    nodes = []
    for grp in input_groups:
        nodes.append(grp)

    for n in hidden_nodes:
        bias = genome.nodes[n].bias if n in genome.nodes else 0.0
        nodes.append({
            "id": n,
            "label": f"H{n}",
            "type": "hidden",
            "bias": float(bias),
        })

    # Group outputs into "from-square" and "to-square" blocks
    # 128 outputs: [0..63] = from-square, [64..127] = to-square
    output_groups = []
    if len(output_nodes) == 128:
        output_groups.append({
            "id": "output_from",
            "label": "From □",
            "type": "output_group",
            "node_ids": output_nodes[:64],
        })
        output_groups.append({
            "id": "output_to",
            "label": "To □",
            "type": "output_group",
            "node_ids": output_nodes[64:],
        })
        for grp in output_groups:
            nodes.append(grp)
    else:
        # Fallback for other output sizes
        for n in output_nodes:
            bias = genome.nodes[n].bias if n in genome.nodes else 0.0
            nodes.append({
                "id": n,
                "label": f"Out{n}",
                "type": "output",
                "bias": float(bias),
            })

    # Build edges
    edges = []
    # Map input node id -> group id for visualization
    node_to_group = {}
    for grp in input_groups:
        for nid in grp["node_ids"]:
            node_to_group[nid] = grp["id"]
    # Also map output nodes to groups
    for grp in output_groups:
        for nid in grp["node_ids"]:
            node_to_group[nid] = grp["id"]

    seen_edges = set()
    for c in connections:
        src, dst = c.key
        # Map src/dst to their group if applicable
        viz_src = node_to_group.get(src, src)
        viz_dst = node_to_group.get(dst, dst)
        edge_key = (viz_src, viz_dst)
        if edge_key not in seen_edges:
            edges.append({
                "source": viz_src,
                "target": viz_dst,
                "weight": float(c.weight),
                "original_src": src,
            })
            seen_edges.add(edge_key)

    return {
        "input_nodes": input_nodes,
        "output_nodes": output_nodes,
        "hidden_nodes": list(hidden_nodes),
        "nodes": nodes,
        "edges": edges,
        "num_connections": len(connections),
        "num_hidden": len(hidden_nodes),
    }
