from pyothello.learn import Net1, NeuralEvaluator, Trainer
import time

SAMPLE_INTERVAL = 15  # seconds

trainer = Trainer()
start = time.time()
while True:
    train_games = 0
    train_start = time.time()
    cur = train_start
    while cur < train_start + SAMPLE_INTERVAL:
        trainer.play_training_game()
        trainer.learn_from_game()
        train_games += 1
        cur = time.time()
    break
    # trainer.play_sample_game()
    # print(f"Trained {train_games} games in {cur - train_start:0.03f} seconds = {train_games / (cur - train_start):0.03f} games per second")