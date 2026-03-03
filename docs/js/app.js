$(document).ready(function () {
    var board = null;
    var game = new Chess();
    var aiPlayer = null;
    var playerColor = 'w';

    var $status = $('#status');
    var $pgn = $('#pgn');
    var $newGameBtn = $('#newGameBtn');
    var $undoBtn = $('#undoBtn');
    var $flipBtn = $('#flipBtn');

    // Load AI Model
    $.getJSON('js/trained_model.json', function (data) {
        aiPlayer = new AIPlayer(data);
        $status.text('Model loaded! You are White.');
        enableControls();
    }).fail(function () {
        $status.text('Failed to load trained model. Play anyway against random moves.');
        enableControls();
    });

    function enableControls() {
        $newGameBtn.prop('disabled', false);
        $undoBtn.prop('disabled', false);
        $flipBtn.prop('disabled', false);
    }

    function onDragStart(source, piece, position, orientation) {
        // Do not pick up pieces if the game is over
        if (game.game_over()) return false;

        // Only pick up pieces for the side to move
        if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
            (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
            return false;
        }

        // Only allow human pieces to be dragged
        if (game.turn() !== playerColor) return false;
    }

    function onDrop(source, target) {
        // see if the move is legal
        var move = game.move({
            from: source,
            to: target,
            promotion: 'q' // NOTE: always promote to a queen for example simplicity
        });

        // illegal move
        if (move === null) return 'snapback';

        updateStatus();

        // Make AI move
        window.setTimeout(makeAIMove, 250);
    }

    function makeAIMove() {
        if (game.game_over() || game.turn() === playerColor) return;

        var move = aiPlayer ? aiPlayer.getBestMove(game) : getRandomMove();
        if (move) {
            game.move(move);
        }

        board.position(game.fen());
        updateStatus();
    }

    function getRandomMove() {
        var moves = game.moves();
        if (moves.length === 0) return null;
        var rIdx = Math.floor(Math.random() * moves.length);
        return moves[rIdx];
    }

    function onSnapEnd() {
        board.position(game.fen());
    }

    function updateStatus() {
        var statusHtml = '';
        var moveColor = (game.turn() === 'w') ? 'White' : 'Black';

        if (game.in_checkmate()) {
            statusHtml = 'Game over, ' + moveColor + ' is in checkmate.';
        } else if (game.in_draw()) {
            statusHtml = 'Game over, drawn position';
        } else {
            statusHtml = moveColor + ' to move';
            if (game.in_check()) {
                statusHtml += ', ' + moveColor + ' is in check';
            }
        }

        $status.text(statusHtml);
        $pgn.text(game.pgn());

        // auto-scroll
        $pgn.scrollTop($pgn[0].scrollHeight);
    }

    var config = {
        pieceTheme: 'img/chesspieces/wikipedia/{piece}.png',
        draggable: true,
        position: 'start',
        onDragStart: onDragStart,
        onDrop: onDrop,
        onSnapEnd: onSnapEnd
    };

    board = Chessboard('board', config);

    $newGameBtn.on('click', function () {
        game.reset();
        board.start();
        updateStatus();
        if (playerColor === 'b') {
            window.setTimeout(makeAIMove, 250);
        }
    });

    $undoBtn.on('click', function () {
        game.undo();
        if (game.turn() !== playerColor) {
            game.undo();
        }
        board.position(game.fen());
        updateStatus();
    });

    $flipBtn.on('click', function () {
        board.flip();
        playerColor = playerColor === 'w' ? 'b' : 'w';
        if (game.turn() !== playerColor) {
            window.setTimeout(makeAIMove, 250);
        }
    });
});
