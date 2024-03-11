from src.utils import *
from torchvision.models import resnet18
ROUTING_TYPES = {
    'gb': lambda c: GumbelBenes(),
    'gbp': lambda c: ParameterizedGumbelBenes(c),
    'gs': lambda c: GumbelSinkhorn(c),
    'id': lambda c: l2l.nn.Lambda(lambda x, w: x)
}

class GumbelSinkhorn(nn.Module):
    def __init__(self, n_channels, tau=1., n_iter=20):
        super(GumbelSinkhorn, self).__init__()
        self.n_channels = n_channels
        self.params = nn.ParameterDict({
            'states': nn.Parameter(torch.randn((self.n_channels, self.n_channels)), requires_grad=True)
        })
        self.tau, self.n_iter= tau, n_iter

    def gumbel_sinkhorn(self, states):
        uniform_noise = torch.rand(states.shape).to(states.device)
        gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-20) + 1e-20)
        log_alpha = (states + gumbel_noise) / self.tau
        for _ in range(self.n_iter):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
        return log_alpha.exp()

    def forward(self, x, w=None):
        """
        :param x: (N, C, H, W)
        """
        if w is None:
            states = self.params['states']
        else:
            states = self.params['states'] * w.reshape(self.params['states'].shape)
        return torch.einsum('nahw,ab->nbhw', x, self.gumbel_sinkhorn(states))

# Implement the GumbelBenes channel shuffling layer
class GumbelBenes(nn.Module):
    def __init__(self):
        super(GumbelBenes, self).__init__()
        self.standard_states = torch.stack([torch.eye(2), 1.0 - torch.eye(2)]).to(device)

    def shuffle(self, x, w):
        """
        :param x: (N, C//2, 2, H*W)
        :param w: (N, C//2, 2, 2)
        :return: apply C//2 permute switches on C//2 channel pairs
        """
        return w @ x

    # routing step
    def exchange(self, x):
        """
        :param x: (N, C//2, 2, H*W)
        :return: route left channel to top-half, right channel to bottom-half
        """
        xl = x[:, :, 0, :].unflatten(1, (x.shape[1]//2, 2))
        xr = x[:, :, 1, :].unflatten(1, (x.shape[1]//2, 2))
        return torch.cat([xl, xr], dim=1)

    def forward(self, x, w):
        """
        :param x: (N, C, H, W)
        :param w: (n_layers, C//2, 2)
        :return:
        """
        w = w.reshape(-1, x.shape[1] // 2, 2)
        w = torch.einsum(
            'abc,cde->abde',
            F.gumbel_softmax(w, hard=False, dim=-1),
            self.standard_states.to(x.device)
        )
        xh, xw = x.shape[-2], x.shape[-1]
        x = x.unflatten(1, (x.shape[1]//2, 2)).flatten(-2, -1)
        for i in range(w.shape[0]):
            x = self.shuffle(x, w[i])
            x = self.exchange(x)
        x = x.flatten(1, 2).unflatten(-1, (xh, xw))
        return x

# Implement GumbelBenes layer with parameters
class ParameterizedGumbelBenes(GumbelBenes):
    def __init__(self, n_channels, n_layers=None):
        super(ParameterizedGumbelBenes, self).__init__()
        self.n_layers = int(2 * np.log2(n_channels)) if n_layers is None else n_layers
        self.n_channels = n_channels
        self.params = nn.ParameterDict({
            'states': nn.Parameter(torch.randn((self.n_layers, n_channels // 2, 2)), requires_grad=True)
        })
        self.dim = self.n_layers * self.n_channels

    def forward(self, x, w=None):
        """
        :param x: (N, C, H, W)
        :param w: (n_layers, C//2, 2)
        :return:
        """
        states = self.params['states'] if w is None else self.params['states'] * w.reshape(self.params['states'].shape)
        w = torch.einsum(
            'abc,cde->abde',
            F.gumbel_softmax(states, hard=False, dim=-1),
            self.standard_states
        )
        xh, xw = x.shape[-2], x.shape[-1]
        x = x.unflatten(1, (x.shape[1]//2, 2)).flatten(-2, -1)
        for i in range(w.shape[0]):
            x = self.shuffle(x, w[i])
            x = self.exchange(x)
        x = x.flatten(1, 2).unflatten(-1, (xh, xw))
        return x

class ConvBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 max_pool=True,
                 max_pool_factor=1.0):
        super(ConvBlock, self).__init__()
        stride = (int(2 * max_pool_factor), int(2 * max_pool_factor))
        if max_pool:
            self.max_pool = torch.nn.MaxPool2d(
                kernel_size=stride,
                stride=stride,
                ceil_mode=False,
            )
            stride = (1, 1)
        else:
            self.max_pool = lambda x: x
        self.normalize = torch.nn.BatchNorm2d(out_channels, affine=True)
        torch.nn.init.uniform_(self.normalize.weight)
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=1,
            bias=True,
        )
        init_weights(self.conv)

    def forward(self, x, w=None):
        w = self.conv.weight * w.reshape(self.conv.weight.shape) if w is not None else self.conv.weight
        x = F.conv2d(x, w, stride=self.conv.stride, bias=self.conv.bias, padding=1)
        x = self.normalize(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x

class ConvBase(nn.Module):
    def __init__(self, c, h, w, output_dim, hidden=64,
                 max_pool=False, max_pool_factor=1., n_layers=3):
        super(ConvBase, self).__init__()
        self.n_layers = n_layers
        # init embedding conv layer
        self.embedding = nn.Sequential(
            l2l.nn.Lambda(lambda x: x.view(-1, c, h, w)),
            l2l.vision.models.cnn4.ConvBlock(
                c, hidden, (3, 3),
                max_pool=max_pool,
                max_pool_factor=max_pool_factor
            ))

        # init core conv layers
        self.core = nn.ModuleList([
            ConvBlock(
                hidden, hidden, (3, 3),
                max_pool=max_pool,
                max_pool_factor=max_pool_factor
            ) for _ in range(n_layers)
        ])

        # init classifier layer
        self.classifier = nn.Sequential(
            l2l.nn.Lambda(lambda x: x.mean(dim=[2, 3])),
            l2l.nn.Flatten(),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        x = self.embedding(x)
        for i in range(self.n_layers):
            x = self.core[i](x)
        x = self.classifier(x)
        return x

class ConvBaseMMAML(nn.Module):
    def __init__(self, c, h, w, output_dim, hidden=64,
                 max_pool=False, max_pool_factor=1., n_layers=3):
        super(ConvBaseMMAML, self).__init__()
        self.n_layers = n_layers
        # init embedding conv layer
        self.embedding = nn.Sequential(
            l2l.nn.Lambda(lambda x: x.view(-1, c, h, w)),
            l2l.vision.models.cnn4.ConvBlock(
                c, hidden, (3, 3),
                max_pool=max_pool,
                max_pool_factor=max_pool_factor
            ))

        # init core conv layers
        self.core = nn.ModuleList([
            ConvBlock(
                hidden, hidden, (3, 3),
                max_pool=max_pool,
                max_pool_factor=max_pool_factor
            ) for _ in range(n_layers)
        ])

        # init classifier layer
        self.classifier = nn.Sequential(
            l2l.nn.Lambda(lambda x: x.mean(dim=[2, 3])),
            l2l.nn.Flatten(),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x, w):
        x = self.embedding(x)
        for i in range(self.n_layers):
            x = self.core[i](x, w[i])
        x = self.classifier(x)
        return x
    def mmaml_forward(self, x, w):
        x = self.embedding(x)
        for i in range(self.n_layers):
            x = self.core[i](x, w[i])
        x = self.classifier(x)
        return x

class ResNetRouting(nn.Module):
    def __init__(self, output_dim, routers=None):
        super(ResNetRouting, self).__init__()

        self.layers = nn.ModuleList(
            list(torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).children())
        )
        #
        # for p in self.layers.parameters():
        #     p.requires_grad = False
        self.resize = transforms.Resize(224)

        # init routing layers
        if routers is None:
            routers = ['gbp' for _ in range(4)]
        router_dim = [64, 128, 256, 512]
        self.router = nn.ModuleList([ROUTING_TYPES[routers[i]](router_dim[i]) for i in range(4)])
        # init classifier layer
        self.classifier = nn.Sequential(
            # nn.Linear(1000, 256),
            # nn.LeakyReLU(),
            # nn.Linear(256, 64),
            # nn.LeakyReLU(),
            nn.LeakyReLU(),
            nn.Linear(1000,output_dim)
            # nn.Tanh()
        )

        # self.classifier = nn.Linear(1000, output_dim)

    def forward(self, x, routing_weights):
        for i, l in enumerate(self.layers):
            x = self.layers[i](x)
            if i in [4,5,6,7]:
                x = self.router[i - 4](x, routing_weights[i - 4])
            if i == 8:
                x = torch.flatten(x, 1)
        return self.classifier(x)

class ResNetEmbedding(nn.Module):
    def __init__(self):
        super(ResNetEmbedding, self).__init__()

        self.layers = nn.ModuleList(
            list(torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).children())
        )

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = self.layers[i](x)
            if i == 8:
                return torch.flatten(x, 1)

class ResNet(nn.Module):
    def __init__(self, output_dim):
        super(ResNet, self).__init__()

        self.layers = nn.ModuleList(
            list(torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).children())
        )

        # init classifier layer
        self.classifier = nn.Sequential(
            # nn.Linear(1000, 256),
            # nn.LeakyReLU(),
            # nn.Linear(256, 64),
            # nn.LeakyReLU(),
            nn.LeakyReLU(),
            nn.Linear(1000,output_dim)
            # nn.Tanh()
        )

        # self.classifier = nn.Linear(1000, output_dim)

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = self.layers[i](x)
            if i == 8:
                x = torch.flatten(x, 1)
        return self.classifier(x)

class ResNetMMAML(nn.Module):
    def __init__(self, output_dim):
        super(ResNetMMAML, self).__init__()

        self.layers = nn.ModuleList(
            list(torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).children())
        )

        # init classifier layer
        self.classifier = nn.Sequential(
            # nn.Linear(1000, 256),
            # nn.LeakyReLU(),
            # nn.Linear(256, 64),
            # nn.LeakyReLU(),
            nn.LeakyReLU(),
            nn.Linear(1000,output_dim)
            # nn.Tanh()
        )

    def forward(self, x, w):
        for i, l in enumerate(self.layers):
            if i in [4, 5, 6, 7]:
                unrolled_li = nn.ModuleList(list(self.layers[i].children()))
                unrolled_bb1 = nn.ModuleList(list(unrolled_li[0].children()))
                identity = x
                if w[i] is None:
                    x = unrolled_bb1[0](x)
                else:
                    x = F.conv2d(x, weight=w[i],
                                 stride=unrolled_bb1[0].stride,
                                 bias=unrolled_bb1[0].bias,
                                 padding=unrolled_bb1[0].bias)
                for j in range(1, len(unrolled_bb1)):
                    x = unrolled_bb1[j](x)
                x = x + identity
                x = x + unrolled_li[1](x)
            else:
                x = self.layers[i](x)
            if i == 8:
                x = torch.flatten(x, 1)
        return self.classifier(x)

class ConvBaseRouting(nn.Module):
    def __init__(self, c, h, w, output_dim, hidden=64,
                 max_pool=False, max_pool_factor=1.,
                 n_layers=3, routers=None):
        super(ConvBaseRouting, self).__init__()
        self.n_layers = n_layers

        # init embedding conv layer
        self.embedding = nn.Sequential(
            l2l.nn.Lambda(lambda x: x.view(-1, c, h, w)),
            l2l.vision.models.cnn4.ConvBlock(
            c, hidden, (3, 3),
            max_pool=max_pool,
            max_pool_factor=max_pool_factor
        ))

        # init core conv layers
        self.core = nn.ModuleList([
            l2l.vision.models.cnn4.ConvBlock(
                hidden, hidden, (3, 3),
                max_pool=max_pool,
                max_pool_factor=max_pool_factor
            ) for _ in range(n_layers)
        ])

        # init routing layers
        if routers is None:
            routers = ['gb' for _ in range(n_layers)]

        self.router = nn.ModuleList([
            ROUTING_TYPES[routers[i]](hidden)
            for i in range(n_layers)
        ])

        # init classifier layer
        self.classifier = nn.Sequential(
            l2l.nn.Lambda(lambda x: x.mean(dim=[2, 3])),
            l2l.nn.Flatten(),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x, routing_weights):
        x = self.embedding(x)
        for i in range(self.n_layers):
            x = self.router[i](x, routing_weights[i])
            x = self.core[i](x)
        x = self.classifier(x)
        return x

# NOTE:
#     Omniglot: hidden=64, channels=1, no max_pool
#     MiniImagenet: hidden=32, channels=3, max_pool
class ProtoNetEmbedding(nn.Module):
    def __init__(self, c, h, w, hidden=64, n_layers=4, max_pool=False):
        super(ProtoNetEmbedding, self).__init__()
        self.hidden_size = hidden
        self.base = l2l.vision.models.cnn4.ConvBase(
            hidden=hidden,
            channels=c,
            max_pool=max_pool,
            layers=n_layers,
        )
        self.features = torch.nn.Sequential(
            l2l.nn.Lambda(lambda x: x.view(-1, c, h, w)),
            self.base,
            l2l.nn.Lambda(lambda x: x.mean(dim=[2, 3])),
            l2l.nn.Flatten(),
        )

    def forward(self, x):
        return self.features(x)


class ModulatorNet(nn.Module):
    def __init__(self, routers, task_embedding_dim, switch_dim):
        super(ModulatorNet, self).__init__()
        self.task_embedding_dim = task_embedding_dim
        self.switch_dim = switch_dim
        self.modulators = nn.ModuleList([
            nn.Sequential(
                FFW([task_embedding_dim, switch_dim]),
                nn.Tanh() if routers[i] in ['gbp', 'gs'] else nn.Identity()
            ) for i in range(len(routers))
        ])

    def forward(self, task_embedding):
        return [m(task_embedding) for m in self.modulators]

class FFW(nn.Sequential):
    def __init__(self, hidden_size):
        core = []
        for i in range(len(hidden_size) - 1):
            core.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
            core.append(nn.ReLU())
        super(FFW, self).__init__(*core)