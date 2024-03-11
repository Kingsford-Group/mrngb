config = {}
config['omniglot'] = {
    'c': 1,
    'h': 28,
    'w': 28,
    'ways': 5,
    'shots': 1,
    'queries': 15,
    'conv_hidden': 64,
    'proto_hidden': 64,
    'max_pool': False,
    'save_dir': '../artifact/omniglot-1shot15queries'
}
config['jigsaw-omniglot'] = {
    'c': 1,
    'h': 28,
    'w': 28,
    'ways': 5,
    'shots': 1,
    'queries': 15,
    'conv_hidden': 64,
    'proto_hidden': 64,
    'max_pool': False,
    'save_dir': '../artifact/jigsaw-omniglot-1shot15queries'
}
config['mini-imagenet'] = {
    'c': 3,
    'h': 84,
    'w': 84,
    'ways': 5,
    'shots': 5,
    'queries': 5,
    'conv_hidden': 32,
    'proto_hidden': 32,
    'max_pool': True,
    'n_layers': 3,
    'save_dir': '../artifact/mini-imagenet-5shot5queries'
}
config['mini-imagenet-hard'] = {
    'c': 3,
    'h': 84,
    'w': 84,
    'ways': 5,
    'shots': 1,
    'queries': 31,
    'conv_hidden': 32,
    'proto_hidden': 32,
    'max_pool': True,
    'n_layers': 3,
    'save_dir': '../artifact/mini-imagenet-1shot5queries'
}
config['jigsaw-mini-imagenet-hard'] = {
    'c': 3,
    'h': 84,
    'w': 84,
    'ways': 5,
    'shots': 1,
    'queries': 15,
    'conv_hidden': 32,
    'proto_hidden': 32,
    'max_pool': True,
    'n_layers': 3,
    'save_dir': '../artifact/jigsaw-mini-imagenet-1shot5queries'
}
config['jigsaw-mini-imagenet'] = {
    'c': 3,
    'h': 84,
    'w': 84,
    'ways': 5,
    'shots': 5,
    'queries': 15,
    'conv_hidden': 32,
    'proto_hidden': 32,
    'max_pool': True,
    'n_layers': 3,
    'save_dir': '../artifact/jigsaw-mini-imagenet-5shot5queries'
}
config['jigsaw-44-mini-imagenet'] = {
    'c': 3,
    'h': 84,
    'w': 84,
    'ways': 5,
    'shots': 5,
    'queries': 5,
    'conv_hidden': 32,
    'proto_hidden': 32,
    'max_pool': True,
    'n_layers': 3,
    'save_dir': '../artifact/jigsaw-44-mini-imagenet-5shot5queries'
}
config['faf'] = {
    'c': 3,
    'h': 112,
    'w': 112,
    'ways': 4,
    'shots': 1,
    'queries': 5,
    'conv_hidden': 64,
    'proto_hidden': 64,
    'max_pool': True,
    'n_layers': 3,
    'save_dir': '../artifact/faf-4way-1shot-5queries'
}
config['faf-same-start'] = {
    'c': 3,
    'h': 112,
    'w': 112,
    'ways': 4,
    'shots': 1,
    'queries': 5,
    'conv_hidden': 32,
    'proto_hidden': 32,
    'max_pool': False,
    'n_layers': 3,
    'save_dir': '../artifact/faf-same-start-4ways-1shot-5queries'
}