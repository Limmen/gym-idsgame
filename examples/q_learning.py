import gym

# Program entrypoint
if __name__ == '__main__':
    pass
    #env = gym.make("gym_yagw:yagw-v1", width=5, height=5)
    # q_agent = QAgent(env, gamma=0.99, alpha=0.2, epsilon=1, render=False, eval_sleep=0.3,
    #                  min_epsilon=0.1, eval_epochs=0, log_frequency=100, epsilon_decay=0.999, video=True,
    #                  video_fps=5, video_dir="./videos")
    # episode_rewards, episode_steps, epsilon_values = q_agent.run(1000)
    # q_agent.print_state_values(height=env.height, width=env.width)
    # q_agent.eval()
    # plot_results(episode_rewards, episode_steps)