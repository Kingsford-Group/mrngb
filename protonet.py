import os.path
from taskset_wrapper import *
from layers import *
from config import *

def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    return logits


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(model, batch, ways, shot, query_num):
    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
    embeddings = model(data)
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)

    for offset in range(shot):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]
    support = support.reshape(ways, shot, -1).mean(dim=1)
    query = embeddings[query_indices]
    labels = labels[query_indices].long()

    logits = pairwise_distances_logits(query, support)
    loss = F.cross_entropy(logits, labels)
    acc = accuracy(logits, labels)
    return loss, acc

def run_protonet(task, max_epoch=10000, meta_batch_size=32, test_batch_size=5, resnet=False):
    cfg = config[task]
    if resnet:
        model = ResNetEmbedding().to(device)
    else:
        model = ProtoNetEmbedding(
            cfg['c'], cfg['h'], cfg['w'],
            hidden=cfg['proto_hidden'],
            max_pool=cfg['max_pool']
        ).to(device)

    tasksets = TASKS[task](cfg['ways'], cfg['shots'], cfg['queries'])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5)

    results = {
        'mean_loss': [],
        'std_loss': [],
        'mean_acc': [],
        'std_acc': []
    }
    # Train
    bar = trange(1, max_epoch+1)
    for epoch in bar:
        model.train()
        bar.set_description_str(f'Train Epoch {epoch}')
        for i in range(meta_batch_size):
            batch = tasksets.sample_task('train')
            loss, acc = fast_adapt(model, batch, cfg['ways'], cfg['shots'], cfg['queries'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bar.set_postfix_str(f'Eval acc {i+1}/{meta_batch_size}={acc}')
        lr_scheduler.step()

        model.eval()
        bar.set_description_str(f'Test Epoch {epoch}')
        test_loss, test_acc = [], []
        for _ in range(test_batch_size):
            batch = tasksets.sample_task('test')
            loss, acc = fast_adapt(model, batch, cfg['ways'], cfg['shots'], cfg['queries'])
            test_loss.append(loss.item())
            test_acc.append(acc)
        results['mean_loss'].append(torch.tensor(test_loss).mean().item())
        results['std_loss'].append(torch.tensor(test_loss).std().item())
        results['mean_acc'].append(torch.tensor(test_acc).mean().item())
        results['std_acc'].append(torch.tensor(test_acc).std().item())
        torch.save(model.state_dict(), os.path.join(cfg['save_dir'], f'proto-{task}-model.pt'))
        torch.save(results, os.path.join(cfg['save_dir'], f'proto-{task}-results.pt'))
        bar.set_postfix_str(f'Test loss={results["mean_loss"][-1]:.3f} '
                            f'acc={results["mean_acc"][-1]:.3f}')
        print('')
    return model, results

if __name__ == '__main__':
    torch.cuda.set_device(1)
    if len(sys.argv) < 2:
        # model, results = run_protonet('omniglot')
        # model, results = run_protonet('jigsaw-omniglot')
        # model, results = run_protonet('jigsaw-mini-imagenet-hard', resnet=True)
        model, results = run_protonet('mini-imagenet-hard', resnet=True)
        # model, results = run_maml('mini-imagenet')
        # model, results = run_protonet('faf')
    else:
        model, results = run_protonet(sys.argv[1])