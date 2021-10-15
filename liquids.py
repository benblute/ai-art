import os
import shutil
import numpy as np
import PIL.Image
import torch
import pickle
import glob
import math
import moviepy.video.io.ImageSequenceClip
from flask import Flask, request, send_file


app = Flask(__name__)


def ellipse(frames):
    a = np.random.randn(512)
    b = np.random.randn(512)
    zs = []
    for frame in range(frames):
        c1 = a * math.sin(math.tau * frame / frames)
        c2 = b * math.cos(math.tau * frame / frames)
        zs.append(c1 + c2)
    return zs


def hexagon(a, b, c, frames, speed):
    w = np.random.randn(512) * speed
    ab = 2*a/3 + 2*b/3 - c/3
    bc = 2*b/3 + 2*c/3 - a/3
    ca = 2*c/3 + 2*a/3 - b/3
    points = [a, ab, b, bc, c, ca]
    pairs = list(zip(points, points[1:] + [points[0]]))
    segment_frames = frames // 6
    zs = []
    for frame in range(frames):
        p1, p2 = pairs[frame // segment_frames]
        distance = (frame % segment_frames) / segment_frames
        c1 = p1 * (1 - distance)
        c2 = p2 * distance
        c3 = w * math.sin(3 * math.tau * frame / frames)
        zs.append(c1 + c2 + c3)
    i = np.random.randint(frames)
    return zs[i:] + zs[:i]


def get_zs(name):
    if len(name) == 7:
        np.random.seed(int(name[1:]))
        return ellipse(240)
    a = get_zs(name + '1')[0]
    b = get_zs(name + '2')[0]
    c = get_zs(name + '3')[0]
    np.random.seed(int('0' + name[1:]))
    level = 7 - len(name)
    return hexagon(a, b, c, 24 * (10 + level * 5), level * 2 / 3)


def image(z, network):
    device = torch.device('cuda')
    with open(network, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device).float()
    label = torch.zeros([1, G.c_dim], device=device)
    z = torch.from_numpy(z.reshape(1, 512)).to(device)
    img = G(z, label, truncation_psi=1, noise_mode='const', force_fp32=True)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')


@app.route('/liquid')
def liquid():
    name = request.args['name']
    network = 'minimalist.pkl' if 'A' in name else 'landscapes.pkl'
    zs = get_zs(name)
    shutil.rmtree('frames', ignore_errors=True)
    os.makedirs('frames')
    for i, z in enumerate(zs):
        image(z, network).save(f'frames/{i:04}.png')
    filenames = sorted(glob.glob(f'frames/*.png'))
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(filenames, fps=24)
    clip.write_videofile('liquid.mp4')
    return send_file('liquid.mp4', attachment_filename=name + '.mp4')


app.run(host='0.0.0.0')

