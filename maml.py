import torch.cuda

from layers import *
from config import *
from taskset_wrapper import *

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, shots, ways, queries, adaptation_steps=1):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evaluation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shots + queries)
    for offset in range(shots):
        adaptation_indices[selection + offset] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adapt_data, adapt_labels = data[adaptation_indices], labels[adaptation_indices]
    eval_data, eval_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adapt_data), adapt_labels)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(eval_data)
    valid_error = loss(predictions, eval_labels)
    valid_accuracy = accuracy(predictions, eval_labels)
    return valid_error, valid_accuracy


def run_maml(task, max_epoch=10000, meta_batch_size=32, test_batch_size=5, first_order=False, resnet=False):
    cfg = config[task]
    if resnet:
        model = ResNet(cfg['ways'])
    else:
        model = ConvBase(
            cfg['c'], cfg['h'], cfg['w'], cfg['ways'],
            hidden=cfg['conv_hidden'],
            max_pool=cfg['max_pool']
        )
    model.to(device)
    tasksets = TASKS[task](cfg['ways'], cfg['shots'], cfg['queries'])

    maml = l2l.algorithms.MAML(model, lr=5e-3 if resnet else 5e-1, first_order=first_order)
    opt = optim.Adam(maml.parameters(), 3e-3)
    loss = nn.CrossEntropyLoss(reduction='mean')
    bar = trange(max_epoch)
    results = {
        'mean_loss': [],
        'std_loss': [],
        'mean_acc': [],
        'std_acc': []
    }
    for epoch in bar:
        bar.set_description_str(f'Train Epoch {epoch}')
        opt.zero_grad()
        for i in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch = tasksets.sample_task('train')
            eval_error, eval_acc = fast_adapt(
                batch, learner, loss,
                cfg['shots'], cfg['ways'], cfg['queries']
            )
            eval_error.backward()
            bar.set_postfix_str(f'Eval acc {i}/{meta_batch_size}={eval_acc.item()}')

        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

        bar.set_description_str(f'Test Epoch {epoch}')
        test_loss, test_acc = [], []
        for _ in range(test_batch_size):
            # Compute meta-testing loss
            learner = maml.clone()
            batch = tasksets.sample_task('test')
            eval_loss, eval_acc = fast_adapt(
                batch, learner, loss,
                cfg['shots'], cfg['ways'], cfg['queries']
            )
            test_loss.append(eval_loss.item())
            test_acc.append(eval_acc.item())
        results['mean_loss'].append(torch.tensor(test_loss).mean().item())
        results['std_loss'].append(torch.tensor(test_loss).std().item())
        results['mean_acc'].append(torch.tensor(test_acc).mean().item())
        results['std_acc'].append(torch.tensor(test_acc).std().item())
        prefix = f'{"fo" if first_order else ""}maml-{task}'
        torch.save(maml.state_dict(), os.path.join(cfg['save_dir'], f'{prefix}-model.pt'))
        torch.save(results, os.path.join(cfg['save_dir'], f'{prefix}-results.pt'))
        bar.set_postfix_str(f'Test loss={results["mean_loss"][-1]:.3f} '
                            f'acc={results["mean_acc"][-1]:.3f}')
        print('')
    return maml, results

if __name__ == '__main__':
    torch.cuda.set_device(0)
    if len(sys.argv) < 2:
        # model, results = run_maml('omniglot')
        # model, results = run_maml('jigsaw-44-mini-imagenet')
        # model, results = run_maml('faf-same-start')
        # model, results = run_maml('jigsaw-mini-imagenet-hard', resnet=True)
        model, results = run_maml('mini-imagenet-hard', resnet=True)

    else:
        model, results = run_maml(sys.argv[1])