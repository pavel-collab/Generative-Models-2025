import json
import matplotlib.pyplot as plt
import os

plt.style.use("ggplot")

JSON_DUMP_PATH = './train_info'

with open(f'{JSON_DUMP_PATH}/gan_train.json', 'r') as f:
    json_str = f.read()
    gan_train = json.loads(json_str)
    
d_losses = gan_train['d_losses']
g_losses = gan_train['g_losses']
real_scores = gan_train['real_scores']
fake_scores = gan_train['fake_scores']

fig = plt.figure()
fig.add_subplot(1, 2, 1)

plt.plot(d_losses, '-')
plt.plot(g_losses, '-')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Discriminator', 'Generator'])
plt.title('Losses')

IMAGE_SAVE_PATH = './images'
if not os.path.exists(IMAGE_SAVE_PATH):
    os.mkdir(IMAGE_SAVE_PATH)

fig.add_subplot(1, 2, 2)

plt.plot(real_scores, '-')
plt.plot(fake_scores, '-')
plt.xlabel('epoch')
plt.ylabel('score')
plt.legend(['Real Score', 'Fake score'])
plt.title('Scores')

fig.savefig(f'{IMAGE_SAVE_PATH}/gan_train.png')
