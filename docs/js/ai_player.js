class AIPlayer {
    constructor(modelJson) {
        this.network = new NeatNetwork(modelJson);
    }

    encodeBoard(game, perspective = 'w') {
        const vector = new Array(768).fill(0.0);
        const board = game.board(); // 8x8 array from a8 to h1

        const ptMap = { 'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5 };

        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const piece = board[row][col];
                if (piece) {
                    // Python python-chess: A1 is 0, H1 is 7, A8 is 56.
                    let sq = (7 - row) * 8 + col;
                    let c = piece.color;

                    if (perspective === 'b') {
                        // python: chess.square_mirror flips the rank.
                        // sq ^ 56 inverts the rank
                        sq = sq ^ 56;
                        c = (c === 'w') ? 'b' : 'w';
                    }

                    const colorOffset = (c === 'w') ? 0 : 6;
                    const pieceIdx = colorOffset + ptMap[piece.type];

                    const idx = sq * 12 + pieceIdx;
                    vector[idx] = 1.0;
                }
            }
        }
        return vector;
    }

    getBestMove(game) {
        const moves = game.moves(); // all legal moves in SAN format
        if (moves.length === 0) return null;

        let bestMove = null;
        let bestScore = -Infinity;

        const turn = game.turn();

        for (let move of moves) {
            // Simulate the move
            game.move(move);

            // Evaluate the resulting position from the agent's perspective
            const vector = this.encodeBoard(game, turn);
            const score = this.network.activate(vector)[0];

            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }

            // Undo the simulation move
            game.undo();
        }

        return bestMove;
    }
}
