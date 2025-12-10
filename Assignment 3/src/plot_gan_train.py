import json
import matplotlib.pyplot as plt
import os

plt.style.use("ggplot")

JSON_DUMP_PATH = './train_info'

with open(f'{JSON_DUMP_PATH}/gan_train.json', 'w') as f:
    gan_train = json.load(f)
    
d_losses = gan_train['d_losses']
g_losses = gan_train['g_losses']
real_scores = gan_train['real_scores']
fake_scores = gan_train['fake_scores']

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)

ax1.plot(d_losses, '-')
ax1.plot(g_losses, '-')
ax1.xlabel('epoch')
ax1.ylabel('loss')
plt.legend(['Discriminator', 'Generator'])
ax1.title('Losses')

IMAGE_SAVE_PATH = './images'
if not os.path.exists(IMAGE_SAVE_PATH):
    os.mkdir(IMAGE_SAVE_PATH)

ax2 = fig.add_subplot(1, 2, 2)

ax2.plot(real_scores, '-')
ax2.plot(fake_scores, '-')
ax2.xlabel('epoch')
ax2.ylabel('score')
plt.legend(['Real Score', 'Fake score'])
ax2.title('Scores')

fig.savefig(f'{IMAGE_SAVE_PATH}/gan_train.png')