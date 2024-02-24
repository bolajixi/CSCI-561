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
        corner_grab = 0
        for i in range(len(self.state.board)):
            for j in range(len(self.state.board[0])):
                if self.state.board[i][j] == self.state.player and (i == 0 or i == 11 or j == 0 or j == 11):
                    corner_grab += 1
        return corner_grab


        # elif type == "staticWeightsEvaluation":
        #     weights = [
        #         [100, -20, 10, 5, 5, 10, -20, 100],
        #         [-20, -50, -2, -2, -2, -2, -50, -20],
        #         [10, -2, -1, -1, -1, -1, -2, 10],
        #         [5, -2, -1, -1, -1, -1, -2, 5],
        #         [5, -2, -1, -1, -1, -1, -2, 5],
        #         [10, -2, -1, -1, -1, -1, -2, 10],
        #         [-20, -50, -2, -2, -2, -2, -50, -20],
        #         [100, -20, 10, 5, 5, 10, -20, 100]
        #     ]

        #     player_score = 0
        #     opponent_score = 0

        #     for i in range(len(state.board)):
        #         for j in range(len(state.board[0])):
        #             if state.board[i][j] == state.player:
        #                 player_score += weights[i][j]
        #             elif state.board[i][j] == state.opponent:
        #                 opponent_score += weights[i][j]

        #     return player_score - opponent_score
