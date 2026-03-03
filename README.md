# ChessNEAT ♟️🧠

Train a chess-playing neural network using the NEAT algorithm (NeuroEvolution of Augmenting Topologies) with a knockout tournament fitness function, and play against it in your browser!

## Overview
ChessNEAT uses `neat-python` and `python-chess` to build an evolutionary training pipeline. To support the complexity of full chess, it evaluates networks on the GPU using PyTorch, batching multiple board positions concurrently.

- **Inputs**: 768 binary features (64 squares × 12 piece types).
- **Network**: Evolving feed-forward topology.
- **Fitness Function**: Knockout tournament bracket. Genomes play games against each other and earn fitness points by advancing rounds. Win = 3 points, Draw = 1, Loss = 0.
- **Frontend**: A minimal JavaScript implementation of the evolved NEAT network runs client-side, embedded in a beautiful `chessboard.js` UI.

## Local Setup & Training

### Environment
Ensure you have an NVIDIA GPU, then create the conda environment:
```bash
conda env create -f environment.yml
conda activate ChessNEAT
```

### Running the Tournament Training
```bash
python src/train.py --generations 50
```
*Note: Due to the 768 input size and round-robin tournament structure across 150 genomes, training will take several hours.*

### Exporting the Winning Agent
Once training completes, the best genome is saved to `models/best_genome.pkl`.
Convert it to a JSON format for the front-end to use:
```bash
python src/export_model.py
```
This writes the active network topology to `docs/js/trained_model.json`.

## Web Client
The `docs/` folder contains a static web app that can be hosted on GitHub Pages. It includes a custom JavaScript class that mirrors PyTorch's `tanh` feedforward evaluation, allowing the AI to run strictly locally without a backend server.

Simply open `docs/index.html` in your browser to play against your trained model!
