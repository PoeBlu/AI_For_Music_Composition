"""Network architecture of the baseline discriminator
"""

NET_D = {
    'pitch_time_private': None,
    'time_pitch_private': None,
    'merged_private': None,
    'shared': None,
    'onset': None,
    'chroma': None,
    'merged': [
        ('conv3d', (128, (1, 1, 12), (1, 1, 12)), None, 'lrelu'),
        ('conv3d', (128, (1, 1, 3), (1, 1, 2)), None, 'lrelu'),
        ('conv3d', (256, (1, 6, 1), (1, 6, 1)), None, 'lrelu'),
        ('conv3d', (256, (1, 4, 1), (1, 4, 1)), None, 'lrelu'),
        ('conv3d', (512, (1, 1, 3), (1, 1, 3)), None, 'lrelu'),
        ('conv3d', (512, (1, 4, 1), (1, 4, 1)), None, 'lrelu'),
        ('conv3d', (1024, (2, 1, 1), (1, 1, 1)), None, 'lrelu'),
        ('reshape', (3 * 1024)),
        ('dense', 1),
    ],
}
