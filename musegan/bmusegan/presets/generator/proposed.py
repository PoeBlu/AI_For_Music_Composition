"""Network architecture of the proposed generator.
"""

NET_G = {
    'z_dim': 128,
    'shared': [
        ('dense', (3 * 512), 'bn', 'relu'),
        ('reshape', (3, 1, 1, 512)),
        ('transconv3d', (256, (2, 1, 1), (1, 1, 1)), 'bn', 'relu'),
        ('transconv3d', (128, (1, 4, 1), (1, 4, 1)), 'bn', 'relu'),
        ('transconv3d', (128, (1, 1, 3), (1, 1, 3)), 'bn', 'relu'),
        ('transconv3d', (64, (1, 4, 1), (1, 4, 1)), 'bn', 'relu'),
        ('transconv3d', (64, (1, 1, 3), (1, 1, 2)), 'bn', 'relu'),
    ],
    'pitch_time_private': [
        ('transconv3d', (64, (1, 1, 12), (1, 1, 12)), 'bn', 'relu'),
        ('transconv3d', (32, (1, 6, 1), (1, 6, 1)), 'bn', 'relu'),
    ],
    'time_pitch_private': [
        ('transconv3d', (64, (1, 6, 1), (1, 6, 1)), 'bn', 'relu'),
        ('transconv3d', (32, (1, 1, 12), (1, 1, 12)), 'bn', 'relu'),
    ],
    'merged_private': [
        ('transconv3d', (1, (1, 1, 1), (1, 1, 1)), 'bn', 'sigmoid')
    ],
}
