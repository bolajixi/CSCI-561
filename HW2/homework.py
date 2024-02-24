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
MAX_DEPTH = 4      # Optimize for better depth limited search
result = ""


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

        self.phase = self.detect_phase()

    def get_possible_moves(self, player):
        return self.calculate_possible_moves(player)

    def calculate_possible_moves(self, player):
        possible_moves = []
        for index_i in range(len(self.board)):              # (y) Row index
            for index_j in range(len(self.board[0])):       # (x) Column index
                move_is_valid, directions = self.is_valid_move(index_j, index_i, player)

                if move_is_valid:
                    possible_moves.append((index_j, index_i, directions))

        return possible_moves

    def is_valid_move(self, x, y, player):
        directions = []

        # Check if the cell is empty
        if self.board[y][x] != '.':
            return False, directions

        # Check if placing a disc at this position will flip any opponent's discs
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                direction_is_valid = self.check_direction(x, y, dx, dy, player)
                if direction_is_valid:
                    directions.append((dx, dy))

        if directions:
            return True, directions
        else:
            return False, directions

    def check_direction(self, x, y, dx, dy, player):
        opponent = 'X' if player == 'O' else 'O'
        # Check if placing a disc at this position will flip any opponent's discs *** in this direction ***
        x += dx
        y += dy
        if y < 0 or y >= len(self.board) or x < 0 or x >= len(self.board[0]) or self.board[y][x] != opponent:
            return False
        x += dx
        y += dy
        while y >= 0 and y < len(self.board) and x >= 0 and x < len(self.board[0]):
            if self.board[y][x] == '.':
                return False
            if self.board[y][x] == player:
                return True
            x += dx
            y += dy
        return False

    def detect_phase(self):
        total_pieces = sum(row.count(self.player) + row.count(self.opponent) for row in self.board)

        if total_pieces < 30:       # Adjusted threshold (20)
            return "early"
        elif total_pieces < 80:     # Adjusted threshold (50)
            return "mid"
        else:
            return "late"

class UtilityEvaluator:
    def __init__(self, state):
        self.state = state

    def discDifference(self):
        '''
        Returns the difference between the number of discs owned by the 
        player and the opponent on the game board.
        '''
        player_discs = sum(row.count(self.state.player) for row in self.state.board)
        opponent_discs = sum(row.count(self.state.opponent) for row in self.state.board)
        # if player_discs > opponent_discs:
        #     return player_discs - opponent_discs
        return player_discs - opponent_discs

    def stability(self):
        '''
        Returns the number of stable disks owned by the player about to move
	    minus the number of stable disks owned by the other player
        '''
        def stable_discs(board, player):
            stable_discs = 0
            for i in range(len(self.state.board)):
                for j in range(len(self.state.board[0])):
                    if board[i][j] == player:
                        if i == 0 or i == 11 or j == 0 or j == 11:
                            stable_discs += 1
            return stable_discs

        return stable_discs(self.state.board, self.state.player) - stable_discs(self.state.board, self.state.opponent)

    def mobility(self):
        '''
        Returns the number of legal moves available to the player about to move
        minus the number of legal moves available to the other player
        '''
        return len(self.state.get_possible_moves(self.state.player)) - len(self.state.get_possible_moves(self.state.opponent))

    def frontier(self):
        '''
        Returns the number of spaces adjacent to opponent pieces minus the
        the number of spaces adjacent to the current player's pieces.
        '''
        def calc_frontier(player):
            frontier = 0
            for i in range(len(self.state.board)):
                for j in range(len(self.state.board[0])):
                    if self.state.board[i][j] == player and (i == 0 or i == 11 or j == 0 or j == 11):
                        frontier += 1
            return frontier

        return calc_frontier(self.state.opponent) - calc_frontier(self.state.player)

    def cornerCapture(self):
        '''
        Returns the number of spaces adjacent to opponent pieces minus the
        the number of spaces adjacent to the current player's pieces.
        '''
        corners = [(0, 0), (0, 11), (11, 0), (11, 11)]

        for move in self.state.get_possible_moves(self.state.player):
            if move[:2] in corners:
                return 1

        return 0

class MinimaxAlphaBeta:
    def __init__(self, state):
        self.state = state

    def utility(self, state):
        evaluate = UtilityEvaluator(state)

        if state.phase == "early":
            return 1000*evaluate.cornerCapture() + 50*evaluate.mobility()
        elif state.phase == "mid":
            return 1000*evaluate.cornerCapture() + 20*evaluate.mobility() + 10*evaluate.discDifference() + 10*evaluate.stability()
        elif state.phase == "late":
            return 1000*evaluate.cornerCapture() + 100*evaluate.mobility() + 500*evaluate.discDifference() + 50*evaluate.stability()

    def terminal_test(self, state):
        # Check if the board is full
        if all(cell != '.' for row in state.board for cell in row):
            return True
        # Check if both players have no valid moves left
        return False if state.get_possible_moves(state.player) else True

    def maximizer(self, state, depth):
        if depth == 0 or self.terminal_test(state):
            return self.utility(state), None

        best_value, best_move = float('-inf'), None

        for move in state.get_possible_moves(state.player):
            new_board_after_move = make_move(state.board, move, state.player)
            new_state = GameState(board=new_board_after_move, player=state.opponent)

            value, _ = self.minimizer(new_state, depth - 1)
            if value > best_value:
                best_value, best_move = value, move

            state.alpha = max(state.alpha, value)
            if state.alpha >= state.beta:
                break

        return best_value, best_move

    def minimizer(self, state, depth):
        if depth == 0 or self.terminal_test(state):
            return self.utility(state), None

        best_value, best_move = float('inf'), None
        
        for move in state.get_possible_moves(state.player):
            new_board_after_move = make_move(state.board, move, state.player)
            new_state = GameState(board=new_board_after_move, player=state.opponent)

            value, _ = self.minimizer(new_state, depth - 1)
            if value < best_value:
                best_value, best_move = value, move

            state.beta = min(state.beta, value)
            if state.alpha >= state.beta:
                break

        return best_value, best_move

    def solve(self):
        if self.state.is_maximizer:
            value, best_move = self.maximizer(self.state, MAX_DEPTH)
        else:
            value, best_move = self.minimizer(self.state, MAX_DEPTH)
            
        return location_mapper((best_move[0], best_move[1]))


# Game Playing
# ---------------------------------------------------------------------------------------------------------------------------------------
state = GameState(BOARD, ASSIGNED_PLAYER)
algorithm = MinimaxAlphaBeta(state)

result = algorithm.solve()

elapsed_time = time.time() - start_time
print(f"\n{result} \n\nElapsed Time = {'%.2f' % round(elapsed_time, 2)} seconds")

with open(OUTPUT_FILE, FILE_WRITE_FORMAT) as output_file:
    output_file.write(result + "\n")