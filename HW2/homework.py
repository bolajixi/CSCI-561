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

def make_move(board, move, player):
    """
    Makes a move on the board for the given player.
    """
    x, y, flip_directions = move
    new_board = [row[:] for row in board]  # Create a copy of the board
    new_board[y][x] = player

    def flip_discs(board, x, y, dx, dy, player):
        """
        Flips discs in the given direction for the given player.
        """
        x += dx
        y += dy

        while 0 <= y < len(board) and 0 <= x < len(board[0]) and board[y][x] != '.' and board[y][x] != player:
            board[y][x] = player
            x += dx
            y += dy

    # Flipping discs in given direction
    for dx, dy in flip_directions:
        if dx == 0 and dy == 0:
            continue
        flip_discs(new_board, x, y, dx, dy, player)

    return new_board


# Helper Classes
# ---------------------------------------------------------------------------------------------------------------------------------------
class GameState:
    def __init__(self, board, player):
        self.board = board

        self.player = player
        self.opponent = 'X' if player == 'O' else 'O'

        self.is_maximizer = True if player == 'X' else False
        self.is_minimizer = True if player == 'O' else False

        self.alpha = float('-inf')
        self.beta = float('inf')
        self.depth = 0

        self.get_possible_moves = self.calculate_possible_moves()

    def calculate_possible_moves(self):
        possible_moves = []
        for index_i in range(len(self.board)):              # (y) Row index
            for index_j in range(len(self.board[0])):       # (x) Column index
                move_is_valid, directions = self.is_valid_move(index_j, index_i)

                if move_is_valid:
                    possible_moves.append((index_j, index_i, directions))

        return possible_moves

    def is_valid_move(self, x, y):
        directions = []

        # Check if the cell is empty
        if self.board[y][x] != '.':
            return False, directions

        # Check if placing a disc at this position will flip any opponent's discs
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                direction_is_valid = self.check_direction(x, y, dx, dy)
                if direction_is_valid:
                    directions.append((dx, dy))

        if directions:
            return True, directions
        else:
            return False, directions

    def check_direction(self, x, y, dx, dy):
        # Check if placing a disc at this position will flip any opponent's discs *** in this direction ***
        x += dx
        y += dy
        if y < 0 or y >= len(self.board) or x < 0 or x >= len(self.board[0]) or self.board[y][x] != self.opponent:
            return False
        x += dx
        y += dy
        while y >= 0 and y < len(self.board) and x >= 0 and x < len(self.board[0]):
            if self.board[y][x] == '.':
                return False
            if self.board[y][x] == self.player:
                return True
            x += dx
            y += dy
        return False


class MinimaxAlphaBeta:
    def __init__(self, state):
        self.state = state

    def utility(self, state, type):
        if type == "discDifference":
            player_discs = sum(row.count(state.player) for row in state.board)
            opponent_discs = sum(row.count(state.opponent) for row in state.board)
            return player_discs - opponent_discs

        elif type == "staticWeightsEvaluation":
            weights = [
                [100, -20, 10, 5, 5, 10, -20, 100],
                [-20, -50, -2, -2, -2, -2, -50, -20],
                [10, -2, -1, -1, -1, -1, -2, 10],
                [5, -2, -1, -1, -1, -1, -2, 5],
                [5, -2, -1, -1, -1, -1, -2, 5],
                [10, -2, -1, -1, -1, -1, -2, 10],
                [-20, -50, -2, -2, -2, -2, -50, -20],
                [100, -20, 10, 5, 5, 10, -20, 100]
            ]

            player_score = 0
            opponent_score = 0

            for i in range(len(state.board)):
                for j in range(len(state.board[0])):
                    if state.board[i][j] == state.player:
                        player_score += weights[i][j]
                    elif state.board[i][j] == state.opponent:
                        opponent_score += weights[i][j]

            return player_score - opponent_score

        elif type == "cornerCapture":
            return 

    def terminal_test(self, state):
        # Check if the board is full
        if all(cell != '.' for row in state.board for cell in row):
            return True
        # Check if both players have no valid moves left
        return False if state.get_possible_moves else True

    def maximizer(self, state):
        if self.terminal_test(state):
            return self.utility(state, UTILITY_TYPE), None

        best_value, best_move = float('-inf'), None

        for move in state.get_possible_moves:
            new_board_after_move = make_move(state.board, move, state.player)
            new_state = GameState(board=new_board_after_move, player=state.opponent)

            value, _ = self.minimizer(new_state)
            if value > best_value:
                best_value, best_move = value, move

            state.alpha = max(state.alpha, value)
            if state.alpha >= state.beta:
                break

        return value

    def minimizer(self, state):
        if self.terminal_test(state):
            return self.utility(state, UTILITY_TYPE), None

        best_value, best_move = float('inf'), None
        
        for move in state.get_possible_moves:
            new_board_after_move = make_move(state.board, move, state.player)
            new_state = GameState(board=new_board_after_move, player=state.opponent)

            value, _ = self.minimizer(new_state)
            if value < best_value:
                best_value, best_move = value, move

            state.beta = min(state.beta, value)
            if state.alpha >= state.beta:
                break

        return value

    def solve(self):
        if self.state.is_maximizer:
            value, best_move = self.maximizer(self.state)
        else:
            value, best_move = self.minimizer(self.state)
            
        return location_mapper((best_move[0], best_move[1]))


# Game Playing
# ---------------------------------------------------------------------------------------------------------------------------------------
transform_board_to_binary(BOARD)

# (Test Make Move): Recall x = column, y = row
# print(location_mapper((2, 4)))
# new_move = make_move(BOARD, (2, 4, [(0,1), (0, -1)]), "X")
# print('\n'.join([' '.join(map(str, row)) for row in new_move]))

state = GameState(BOARD, ASSIGNED_PLAYER)
algorithm = MinimaxAlphaBeta(state)

result = algorithm.solve()

elapsed_time = time.time() - start_time
print(f"\nElapsed Time = {'%.2f' % round(elapsed_time, 2)} seconds \n\n{result}")

with open(OUTPUT_FILE, FILE_WRITE_FORMAT) as output_file:
    output_file.write(result + "\n")