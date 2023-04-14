"""Network architecture of the ablated (without onset and chroma streams)
discriminator
"""

NET_D = {
    'pitch_time_private': [
        ('conv3d', (32, (1, 1, 12), (1, 1, 12)), None, 'lrelu'),
        ('conv3d', (64, (1, 6, 1), (1, 6, 1)), None, 'lrelu'),
    ],
    'time_pitch_private': [
        ('conv3d', (32, (1, 6, 1), (1, 6, 1)), None, 'lrelu'),
        ('conv3d', (64, (1, 1, 12), (1, 1, 12)), None, 'lrelu'),
    ],
    'merged_private': [('conv3d', (64, (1, 1, 1), (1, 1, 1)), None, 'lrelu')],
    'shared': [
        ('conv3d', (128, (1, 4, 3), (1, 4, 2)), None, 'lrelu'),
        ('conv3d', (256, (1, 4, 3), (1, 4, 3)), None, 'lrelu'),
    ],
    'onset': None,
    'chroma': None,
    'merged': [
        ('conv3d', (512, (2, 1, 1), (1, 1, 1)), None, 'lrelu'),
        ('reshape', (3 * 512)),
        ('dense', 1),
    ],
}
