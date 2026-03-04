import pickle
import json
import os
import neat
import torch
import sys

# We need the feed_forward_layers from neat to sort topologically
from neat.graphs import feed_forward_layers

def export_model(genome_path="models/best_genome.pkl", config_path="neat_config.ini", out_path="docs/js/trained_model.json"):
    if not os.path.exists(genome_path):
        print(f"Error: genome file '{genome_path}' not found.")
        # If user is running tests before full training, save a mock network
        return create_mock_model(config_path, out_path)
        
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)
        
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
        
    input_nodes = [-i - 1 for i in range(config.genome_config.num_inputs)]
    output_nodes = [i for i in range(config.genome_config.num_outputs)]
    
    connections = [cg for cg in genome.connections.values() if cg.enabled]
    conn_tuples = [cg.key for cg in connections]
    
    print("DEBUG inputs:", input_nodes[:5])
    print("DEBUG outputs:", output_nodes)
    print("DEBUG conn_tuples:", conn_tuples[:5])
    
    result = feed_forward_layers(input_nodes, output_nodes, conn_tuples)
    layers = result[0] if isinstance(result, tuple) else result
    
    # Export format:
    # {
    #   "num_inputs": 768,
    #   "layers": [
    #       { "nodes": [2, 3], "weights": [{"src": -1, "dst": 2, "w": 0.5}, ...], "biases": {"2": 0.1, "3": -0.2} }
    #   ],
    #   "outputs": [0]
    # }
    
    model_json = {
        "num_inputs": len(input_nodes),
        "num_outputs": len(output_nodes),
        "network_type": "policy",  # 64 from-square + 64 to-square scores
        "output_semantics": {
            "from_squares": list(range(0, 64)),
            "to_squares": list(range(64, 128)),
        },
        "outputs": output_nodes,
        "layers": []
    }
    
    for layer in layers:
        layer_nodes = list(layer)
        layer_weights = []
        layer_biases = {}
        
        for orig_node in layer_nodes:
            node = list(orig_node)[0] if isinstance(orig_node, set) else orig_node
            
            layer_biases[node] = genome.nodes[node].bias if node in genome.nodes else 0.0
            
            for c in connections:
                if c.key[1] == node:
                    layer_weights.append({
                        "src": c.key[0],
                        "dst": node,
                        "w": c.weight
                    })
                    
        model_json["layers"].append({
            "nodes": [list(n)[0] if isinstance(n, set) else n for n in layer_nodes],
            "weights": layer_weights,
            "biases": layer_biases
        })
        
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model_json, f, indent=2)
        
    print(f"Exported model to {out_path}")

def create_mock_model(config_path, out_path):
    print("Creating mock model for testing frontend...")
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path
    )
    pop = neat.Population(config)
    genome = list(pop.population.values())[0]
    
    os.makedirs("models", exist_ok=True)
    with open("models/best_genome.pkl", "wb") as f:
        pickle.dump(genome, f)
        
    export_model("models/best_genome.pkl", config_path, out_path)

if __name__ == "__main__":
    export_model()
