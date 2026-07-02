from reinforcement_learning import ExampleGridWorldEnv
from pynput import keyboard


env = ExampleGridWorldEnv()
keys_to_action = {"d": 0, "w": 1, "a": 2, "s": 3}
print(env.reset())

def on_press(key):
    if key.char not in keys_to_action:
        return
    action = keys_to_action[key.char]
    state, reward, done, info = env.step([action])
    env.render()
    print("state:{}, action: {}, reward: {}, done: {}".format(state, action, reward, done))
    if done:
        print("Episode finished")
        env.reset()

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()