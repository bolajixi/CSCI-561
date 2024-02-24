# Othello Heuristic evaluator

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
