class NeatNetwork {
    constructor(modelData) {
        this.numInputs = modelData.num_inputs;
        this.outputs = modelData.outputs;
        this.layers = modelData.layers;
        // In python, input nodes are -1, -2, ..., -num_inputs
        // Let's create an array of nodes or just a map of node_id -> value
        // Using a Map is easy since node IDs might be arbitrary integers.
    }

    activate(inputs) {
        if (inputs.length !== this.numInputs) {
            throw new Error(`Expected ${this.numInputs} inputs, got ${inputs.length}`);
        }

        let nodeValues = new Map();

        // 1. Set input nodes (-1 to -numInputs)
        for (let i = 0; i < this.numInputs; i++) {
            nodeValues.set(-(i + 1), inputs[i]);
        }

        // 2. Compute layers in topological order
        for (const layer of this.layers) {
            for (const node of layer.nodes) {
                let sum = layer.biases[node] || 0.0;

                // Get all weights targeting this node
                const incoming = layer.weights.filter(w => w.dst === node);

                for (const w of incoming) {
                    const srcValue = nodeValues.get(w.src) || 0.0;
                    sum += w.w * srcValue;
                }

                // Activation function is tanh
                const activated = Math.tanh(sum);
                nodeValues.set(node, activated);
            }
        }

        // 3. Collect outputs
        return this.outputs.map(outNode => nodeValues.get(outNode) || 0.0);
    }
}
