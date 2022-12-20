import os
import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras.layers import Dense
import tensorflow_probability as tfp
from tictactoe import *

from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete, Text

from random import choice

class TicTacToe:
    def __init__(self) -> None:
        self.rows = [[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']]
        self.number_rows = [0] * 9

    def reset_board(self):
        self.rows = [[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']]

    def print_board(self):
        print(f" {self.rows[0][0]} | {self.rows[0][1]} | {self.rows[0][2]}")
        print("-----------")
        print(f" {self.rows[1][0]} | {self.rows[1][1]} | {self.rows[1][2]}")
        print("-----------")
        print(f" {self.rows[2][0]} | {self.rows[2][1]} | {self.rows[2][2]}\n")

    def update_number_rows(self):
        # ' ' = 0 | 'X' = 1 | 'O' = 2
        num = []
        for r in self.rows:
            for s in r:
                if s == ' ':
                    num.append(0)
                elif s == 'X':
                    num.append(1)
                else:
                    num.append(2)
        self.number_rows = num

    def number_to_string(self, number_board):
        board = [[],[],[]]
        for i in range(len(number_board)):
            if number_board[i] == 0:
                board[i//3].append(' ')
            elif number_board[i] == 1:
                board[i//3].append('X')
            else:
                board[i//3].append('O')
        return board

    def board_to_string(self):
        num = ""
        for r in self.rows:
            for s in r:
                num += s
        return num

    def number_of_empty_twos(self, board, token):
        # X = 1 | O = 2
        if token == 'X':
            key = 1
        else:
            key = 2

        num_twos = 0
        blocks = []

        # Rows
        for i in range(3):
            row = [board[(i*3)], board[(i*3)+1], board[(i*3)+2]]

            # For X's so key = 1
            if (row.count(0) == 1) and ((sum(row) == 2 and row.count(key) == 2) or (sum(row) == 4 and row.count(key) == 2)):
                #print(row, row.count(0))
                blocks.append(row.index(0)+(i*3))
                num_twos += 1

        # Cols
        for i in range(3):
            col = [board[i], board[i+3], board[i+6]]

            if (col.count(0) == 1) and ((sum(col) == 2 and col.count(key) == 2) or (sum(col) == 4 and col.count(key) == 2)):
                blocks.append((col.index(0)*3)+i)
                num_twos += 1

        # Diagnols
        # TL -> BR
        dia = [board[0], board[4], board[8]]
        if (dia.count(0) == 1) and ((sum(dia) == 2 and dia.count(key) == 2) or (sum(dia) == 4 and dia.count(key) == 2)):
            blocks.append(dia.index(0)*4)
            num_twos += 1

        # BL -> TR
        dia = [board[2], board[4], board[6]]
        if (dia.count(0) == 1) and ((sum(dia) == 2 and dia.count(key) == 2) or (sum(dia) == 4 and dia.count(key) == 2)):
            blocks.append((dia.index(0)+1)*2)
            num_twos += 1

        return (num_twos, blocks)

    def observe(self):
        self.update_number_rows()
        return tuple(self.number_rows)

    def evaluate(self, before, move, token='O'):
        score = 0.0

        self.update_number_rows()

        # Does this move win the game
        wins, winning_spots = self.number_of_empty_twos(before, 'O')
        if wins > 0:
            if move in winning_spots:
                score += (125 + (len(self.possible_moves(self.number_to_string(self.number_rows))) * 10))
            else:
                score -= 125

        # Does this move block stop a loss
        num_enemy_twos, enemy_blocks = self.number_of_empty_twos(before, 'X')
        if num_enemy_twos > 0:
            #print(enemy_blocks)
            if move not in enemy_blocks:
                score -= 75
            else:
                score += 75

        # Does this move give any open 2 in a rows
        more_twos, two_slots = self.number_of_empty_twos(self.number_rows, 'O')
        if more_twos > wins and wins == 0:
            score += 25

        #print(score+1)
        return score+1

    def game_over(self, board) -> bool:
        #Rows
        board = self.number_to_string(board)
        for i in range(3):            
            if board[i].count('X') == 3 or board[i].count('O') == 3:
                return True

        #Columns
        cols = [[],[],[]]
        for i in range(3):
            for x in range(3):
                cols[i].append(board[x][i])

        for i in range(3):            
            if cols[i].count('X') == 3 or cols[i].count('O') == 3:
                return True 

        # Diagonals
        if (board[0][0] == board[1][1] and board[1][1] == board[2][2] and board[1][1] != ' ') \
           or (board[0][2] == board[1][1] and board[1][1] == board[2][0] and board[1][1] != ' '):
            return True

        # Board Full
        count = 0
        for i in range(3):
            count += board[i].count(' ')
        if count == 0:
            return True

        return False

    def player_place(self, token='X'):
        num = int(input("Where? "))
        self.place_piece(token, num)

    def comp_place(self, slot, token='O'):
        self.place_piece(token, slot)

    def possible_moves(self, board):
        empty = []
        counter = 0
        for r in board:
            for s in r:
                if s == ' ':
                    empty.append(counter)
                counter += 1
        return empty

    def random_move(self, token='X'):
        self.place_piece(token, choice(self.possible_moves(self.rows)))

    def random_but_wins(self, token='X'):
        pass

    def place_piece(self, token, slot):
        self.rows[slot//3][slot%3] = token

class TTT_Actor(keras.Model):
    def __init__(self, n_actions, fc1_dim=1024, fc2_dim=512, name='TicTacToe_Engine'):
        super(TTT_Actor, self).__init__()
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.n_action = n_actions
        self.model_name = name
        self.checkpoint_file = os.path.join(name+'_ac')

        self.fc1 = Dense(self.fc1_dim, activation='relu')
        self.fc2 = Dense(self.fc2_dim, activation='relu')
        self.v = Dense(1, activation=None)
        self.pi = Dense(n_actions, activation='softmax')

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)
        v = self.v(value)
        pi = self.pi(value)

        return v, pi

class TTT_Agent():
    def __init__(self, alpha=0.0003, gamma=0.99, n_actions=2) -> None:
        self.gamma = gamma
        self.alpha = alpha
        self.n_actions = n_actions
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]

        self.actor_critic = TTT_Actor(n_actions)
        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))

    def chose_action(self, observation, possible_moves):
        state = tf.convert_to_tensor([observation])
        _, prob = self.actor_critic(state)

        action_probabilities = tfp.distributions.Categorical(probs=prob)

        action = action_probabilities.sample()

        while action not in possible_moves:
            action = action_probabilities.sample()

        self.action = action
        return action.numpy()[0]

    def save_model(self):
        print('SAVING MODEL!!!')
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_model(self):
        print('LOADING MODEL!!!')
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)

    def learn(self, state, reward_recieved, state_, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward_recieved = tf.convert_to_tensor([reward_recieved], dtype=tf.float32)

        with tf.GradientTape(persistent=False) as tape:
            state_value, prob = self.actor_critic(state)
            state_value_, _ = self.actor_critic(state_)

            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            action_probs = tfp.distributions.Categorical(probs=prob)
            log_prob = action_probs.log_prob(self.action)
            delta = reward_recieved + self.gamma * state_value_ * (1-int(done)) - state_value
            actor_loss = -log_prob * delta
            critic_loss = delta**2

            total_loss = actor_loss + critic_loss

        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(zip(gradient, self.actor_critic.trainable_variables))


class TTT_env(Env):
    def __init__(self) -> None:
        self.game = TicTacToe()
        self.action_space = Discrete(9)
        self.observation_space = Discrete(9)
        self.state = [0] * 9
        self.length = 5

    def step(self, action):
        #Action
        before_action = self.game.observe()
        self.game.comp_place(action, 'O')
        obs = self.game.observe()
        reward = self.game.evaluate(before_action, action)
        done = self.game.game_over(obs)
        return obs, reward, done, {}

    def render(self):
        pass

    def reset(self):
        self.game.reset_board()
        return self.game.observe()