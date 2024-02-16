# Reversi / Othello Adversarial Search (with alpha-beta pruning)
# ---------------------------------------------------------------------------------------------------------------------------------------
import os
import time

start_time = time.time()

global ASSIGNED_PLAYER, UTILITY_TYPE
INPUT_FILE = "input.txt"
OUTPUT_FILE = "output.txt"
FILE_WRITE_FORMAT = "w"

BOARD = []
X_GRID_SIZE, Y_GRID_SIZE = 12, 12
result = ""
UTILITY_TYPE = "staticWeightsEvaluation"


# Preprocessing
# ---------------------------------------------------------------------------------------------------------------------------------------
output_file_exists = os.path.isfile(OUTPUT_FILE)

if output_file_exists:
    os.remove(OUTPUT_FILE)
    print(f"Removed '{OUTPUT_FILE}' file")

with open(INPUT_FILE, "r") as input_file:
    ASSIGNED_PLAYER = input_file.readline().strip() # Color X: Black, Color O: White
    GAME_PLAYTIME = input_file.readline().strip()
    STARTING_GAME_STATE = input_file.readlines()

    p1_time, p2_time = GAME_PLAYTIME.split()
    p1_time, p2_time = float(p1_time), float(p2_time)

    for row_string in STARTING_GAME_STATE:
        row = [cell for cell in row_string.strip()]
        BOARD.append(row)

if X_GRID_SIZE != len(BOARD[0]):
    raise Exception("X_GRID_SIZE does not match the number of columns in the game state")
elif Y_GRID_SIZE != len(BOARD):
    raise Exception("Y_GRID_SIZE does not match the number of rows in the game state")


# Helper Functions
# ---------------------------------------------------------------------------------------------------------------------------------------
def location_mapper(location):
    x, y = location
    return chr(x + 97) + str(y + 1)

def transform_board_to_binary(board):
    global BOARD_TRANSFORMED
    BOARD_TRANSFORMED = [list(map(lambda x: 1 if x == "X" else -1 if x == "O" else 0, row)) for row in board]


# Helper Classes
# ---------------------------------------------------------------------------------------------------------------------------------------
class GameState:
    def __init__(self, board, prev_board, player):
        self.board = board
        # self.prev_board = prev_board

        self.player = player
        self.is_maximizer = True if player == "X" else False
        self.is_minimizer = True if player == "O" else False

        self.alpha = -float('inf')
        self.beta = float('inf')
        self.depth = 0

        self.get_possible_moves = self.calculate_possible_moves()

    def calculate_possible_moves(self):
        possible_moves = []
        for index_i in range(len(self.board)):
            for index_j in range(len(self.board[0])):
                if self.is_valid_move(index_i, index_j):
                    possible_moves.append((index_i, index_j))
        return possible_moves

    def is_valid_move(self, x, y):
        # Check if the cell is empty
        if self.board[x][y] != '.':
            return False
        # Check if placing a disc at this position will flip any opponent's discs
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if self.check_direction(x, y, dx, dy):
                    return True
        return False

    def check_direction(self, x, y, dx, dy):
        # Check if placing a disc at this position will flip any opponent's discs *** in this direction ***
        opponent = 'X' if self.player == 'O' else 'O'
        x += dx
        y += dy
        if x < 0 or x >= len(self.board) or y < 0 or y >= len(self.board[0]) or self.board[x][y] != opponent:
            return False
        x += dx
        y += dy
        while x >= 0 and x < len(self.board) and y >= 0 and y < len(self.board[0]):
            if self.board[x][y] == '.':
                return False
            if self.board[x][y] == self.player:
                return True
            x += dx
            y += dy
        return False

        # self.tiles = []
        # self.all_tiles = []
        # self.all_moves = []
        # self.best_move = None

        # self.get_all_tiles()
        # self.get_all_moves()
        # self.get_best_move()
    def get_evaluation():
        return

class MinimaxAlphaBeta:
    def __init__(self, state):
        self.state = state

    def utility(self, state, type):
        if type == "discDifference":
            player_discs = sum(row.count(state.player) for row in state.board)
            opponent_discs = sum(row.count('X' if state.player == 'O' else 'O') for row in state.board)
            return player_discs - opponent_discs

        elif type == "staticWeightsEvaluation":
            weights = [[100, -20, 10, 5, 5, 10, -20, 100],
                    [-20, -50, -2, -2, -2, -2, -50, -20],
                    [10, -2, -1, -1, -1, -1, -2, 10],
                    [5, -2, -1, -1, -1, -1, -2, 5],
                    [5, -2, -1, -1, -1, -1, -2, 5],
                    [10, -2, -1, -1, -1, -1, -2, 10],
                    [-20, -50, -2, -2, -2, -2, -50, -20],
                    [100, -20, 10, 5, 5, 10, -20, 100]]

            player_score = 0
            opponent_score = 0

            for i in range(len(state.board)):
                for j in range(len(state.board[0])):
                    if state.board[i][j] == state.player:
                        player_score += weights[i][j]
                    elif state.board[i][j] != '.':
                        opponent_score += weights[i][j]

            return player_score - opponent_score

        elif type == "cornerCapture":
            return 

    def terminal_test(self, state):
        return False if state.get_possible_moves else True

    def solve(self):
        # TODO: reevaluate this function
        value = self.maximizer(self.state, float('-inf'), float('inf'), 0)
        self.state.evaluation = value

        answer = self.state.best_move, self.state.evaluation, self.state.depth, self.state.alpha, self.state.beta, self.state.evaluation - self.state.alpha, self.state.evaluation - self.state.beta, self.state.evaluation - self.state.beta / (self.state.depth + 1) if self.state.depth else 0, self.state.evaluation

        return answer

    def maximizer(self, state, alpha, beta, depth):
        if self.terminal_test(state):
            return self.utility(state, UTILITY_TYPE)

        value = float('-inf')
        for move in state.get_possible_moves:
            value = max(value, self.minimizer(move, alpha, beta, depth + 1))
            alpha = max(alpha, value)
            if alpha >= beta:
                break

        return value

    def minimizer(self, state, alpha, beta, depth):
        if self.terminal_test(state):
            return self.utility(state, UTILITY_TYPE)

        value = float('inf')
        for move in state.get_possible_moves:
            value = min(value, self.maximizer(move, alpha, beta, depth + 1))
            beta = min(beta, value)
            if alpha >= beta:
                break

        return value


# Game Playing
# ---------------------------------------------------------------------------------------------------------------------------------------
transform_board_to_binary(BOARD)

state = GameState(BOARD, None, ASSIGNED_PLAYER)
algorithm = MinimaxAlphaBeta(state)

result = algorithm.solve()

elapsed_time = time.time() - start_time
print(f"\nElapsed Time = {'%.2f' % round(elapsed_time, 2)} seconds \n\n{result}")

with open(OUTPUT_FILE, FILE_WRITE_FORMAT) as output_file:
    output_file.write(result + "\n")