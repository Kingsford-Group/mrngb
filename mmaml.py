import torch.cuda

from layers import *
from config import *
from taskset_wrapper import *

class MMAML(nn.Module):
    def __init__(self, c, h, w, ways, shots, queries,
                 proto_hidden=64, conv_hidden=64,
                 n_layers=4, max_pool=False,
                 resnet=False,
                 save_dir='../artifact'):
        super(MMAML, self).__init__()
        self.c, self.h, self.w = c, h, w
        self.ways, self.shots, self.queries =  ways, shots, queries
        self.proto_hidden = proto_hidden
        self.conv_hidden = conv_hidden
        self.n_layers = n_layers
        # init learning networks
        self.proto_net = ProtoNetEmbedding(
            c, h, w,
            hidden=proto_hidden,
            max_pool=max_pool,
            n_layers = n_layers
        )
        if resnet:
            self.predictor_net = ResNetMMAML(ways)
            self.modulator_net = nn.Sequential(
                FFW([ways * proto_hidden, 36864]),
                nn.Tanh()
            )
        else:
            self.predictor_net = ConvBaseMMAML(
                c, h, w, ways,
                hidden=conv_hidden,
                n_layers= n_layers,
                max_pool=max_pool
            )
            self.modulator_net = nn.Sequential(
                FFW([ways * proto_hidden, 9 * conv_hidden ** 2]),
                nn.Tanh()
            )
        self.results = None
        self.save_dir = save_dir
        self.loss = nn.CrossEntropyLoss()

    def compute_embedding(self, data, labels):
        embedding = []
        for i in range(self.ways):
            xi = data[torch.where(labels==i)]
            if xi.shape[0] == 0:
                embedding.append(torch.zeros(self.task_embedding_dim).to(data.device))
            else:
                embedding.append(torch.mean(self.proto_net(xi), dim=0, keepdim=False))
        return torch.cat(embedding)

    def maml_fast_adapt(self, batch, learner, adaptation_steps=1, predict=True):
        adapt_data, adapt_labels, eval_data, eval_labels = batch
        task_embedding = self.compute_embedding(adapt_data, adapt_labels)
        # modulation = [m(task_embedding) for m in self.modulator_net]
        modulation = [
            self.modulator_net(task_embedding), None, None, None
        ]
        # Adapt the model
        for step in range(adaptation_steps):
            train_error = self.loss(learner(adapt_data, modulation), adapt_labels)
            learner.adapt(train_error)
        logits = learner(eval_data, modulation)
        eval_loss = self.loss(logits, eval_labels)
        if not predict:
            return eval_loss, logits
        else:
            prediction = logits.argmax(dim=1).view(eval_labels.shape)
            eval_acc = (prediction == eval_labels).sum().float() / eval_labels.shape[0]
            return eval_loss, eval_acc

    def protonet_loss(self, batch, logit=False):
        adapt_data, adapt_labels, eval_data, eval_labels = batch
        support = self.compute_embedding(adapt_data, adapt_labels).reshape(self.ways, -1)
        query = self.proto_net(eval_data)
        logits = torch.cdist(query, support)
        if logit:
            return logits
        else:
            return self.loss(logits, eval_labels)

    def meta_train(self, task, tasksets, max_epoch=800, meta_batch_size=32, test_batch_size=5):
        maml = l2l.algorithms.MAML(
            self.predictor_net, lr=5e-1,
            first_order=False,
            allow_unused=True,
            allow_nograd=True
        )
        maml_opt = optim.Adam(maml.parameters(), lr=3e-3)
        routing_opt = optim.Adam([
            {'params': self.proto_net.parameters(), 'lr': 3e-3},
            {'params': self.modulator_net.parameters(), 'lr': 3e-3},
        ])
        results = {
            'mean_loss': [],
            'std_loss': [],
            'mean_acc': [],
            'std_acc': [],
            'time': []
        }
        bar = trange(max_epoch)
        for epoch in bar:
            start_time = time()
            bar.set_description_str(f'Train Epoch {epoch}')
            maml_opt.zero_grad()
            for i in range(meta_batch_size):
                # Compute meta-training loss
                routing_opt.zero_grad()
                learner = maml.clone()
                train_batch = tasksets.split_batch(tasksets.sample_task('train'))
                eval_loss, eval_acc = self.maml_fast_adapt(train_batch, learner)
                eval_loss.backward()
                routing_opt.step()
                bar.set_postfix_str(f'Eval acc {i+1}/{meta_batch_size}={eval_acc}')
            for p in maml.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(1.0 / meta_batch_size)
            maml_opt.step()
            results['time'].append(time() - start_time)

            bar.set_description_str(f'Test Epoch {epoch}')
            test_loss, test_acc = [], []
            for _ in range(test_batch_size):
                # Compute meta-testing loss
                learner = maml.clone()
                test_batch = tasksets.split_batch(tasksets.sample_task('test'))
                eval_loss, logits = self.maml_fast_adapt(test_batch, learner, predict=False)
                logits += self.protonet_loss(test_batch, logit=True)
                adapt_data, adapt_labels, eval_data, eval_labels = test_batch
                prediction = logits.argmax(dim=1).view(eval_labels.shape)
                eval_acc = (prediction == eval_labels).sum().float() / eval_labels.shape[0]
                test_loss.append(eval_loss.item())
                test_acc.append(eval_acc.item())
            results['mean_loss'].append(torch.tensor(test_loss).mean().item())
            results['std_loss'].append(torch.tensor(test_loss).std().item())
            results['mean_acc'].append(torch.tensor(test_acc).mean().item())
            results['std_acc'].append(torch.tensor(test_acc).std().item())

            torch.save(self.state_dict(), os.path.join(self.save_dir, f'mmaml-{task}-model.pt'))
            torch.save(results, os.path.join(self.save_dir, f'mmaml-{task}-results.pt'))
            bar.set_postfix_str(f'Test loss={results["mean_loss"][-1]:.3f} '
                                f'acc={results["mean_acc"][-1]:.3f}')

            print('')
        return results

def run_mmaml(task, max_epoch=10000, meta_batch_size=32, test_batch_size=5):
    seed()
    config['resnet'] = False
    model = MMAML(**config[task])
    model = model.to(device)
    tasksets = TASKS[task](model.ways, model.shots, model.queries)
    results = model.meta_train(task, tasksets, max_epoch, meta_batch_size, test_batch_size)
    return model, results

def run_mmaml_resnet(task, max_epoch=2000, meta_batch_size=32, test_batch_size=5):
    seed()
    config['True'] = False
    model = MMAML(**config[task])
    model = model.to(device)
    tasksets = TASKS[task](model.ways, model.shots, model.queries)
    results = model.meta_train(task, tasksets, max_epoch, meta_batch_size, test_batch_size)
    return model, results

def run_mmaml_measure_time(task, max_epoch=20, meta_batch_size=32, test_batch_size=5):
    seed()
    for c in [16, 32, 64, 128, 256]:
        cfg = deepcopy(config[task])
        cfg['conv_hidden'] = cfg['proto_hidden'] = c
        cfg['save_dir'] = cfg['save_dir'] + f'-{c}c'
        model = MMAML(**cfg)
        model = model.to(device)
        tasksets = TASKS[task](model.ways, model.shots, model.queries)
        results = model.meta_train(task, tasksets, max_epoch, meta_batch_size, test_batch_size)
    return model, results

if __name__ == '__main__':
    torch.cuda.set_device(1)
    if len(sys.argv) < 2:
        # model, results = run_mmaml_measure_time('omniglot')
        # model, results = run_mmaml('jigsaw-44-mini-imagenet')
        # model, results = run_mmaml('faf-same-start')
        # model, results = run_mmaml_resnet('jigsaw-mini-imagenet-hard')
        model, results = run_mmaml_resnet('mini-imagenet-hard')
    else:
        model, results = run_mmaml(sys.argv[1])