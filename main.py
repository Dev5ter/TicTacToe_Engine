from tictactoe import *

if __name__ == "__main__":
    env = TTT_env()
    agent = TTT_Agent(alpha=1e-5, n_actions=9)
    n_games = 10005

    best_score = env.reward_range[0]
    score_history = []

    load_checkpoint = False

    if load_checkpoint:
        pass

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            moves = env.game.possible_moves(env.game.rows)
            action = agent.chose_action(observation, moves)
            observation_, reward, done, info = env.step(action)
            #print("------------\n", score)
            score += reward
            #print(score, "\n----------------")

            if not load_checkpoint:
                agent.learn(observation, reward, observation_, done)

            observation = observation_

            #env.game.update_number_rows()
            #print("------------Engine Turn------------")
            #print(env.game.number_rows)
            #env.game.print_board()
            #done = env.game.game_over(env.game.number_rows)
            #print(done)
            #print("-------------------------------------")
            if not done:
                #env.game.print_board()
                if i < n_games-3:
                    env.game.random_move('X')
                else:
                    env.game.print_board()
                    env.game.player_place('X')
                #env.game.player_place('X')
                env.game.update_number_rows()
                done = env.game.game_over(env.game.number_rows)

                #print("---------RANDO---------")
                #env.game.print_board()
                #print("-----------------------")

        score_history.append(score)
        if i % 50 == 0 and i > 0:
            avg = sum(score_history)/len(score_history)
            print(avg, best_score, f"({min(score_history)} | {max(score_history)})")
            if avg > best_score:
                best_score = avg
                if not load_checkpoint:
                    agent.save_model()
            score_history = []

        if i % 500 == 0 and i > 0:
            print(f"{i} games completed")
        if not i < n_games-3:
            env.game.print_board()
        #print("-------------NEW GAME-------------")
        score_history.append(score)
        avg_score = []
