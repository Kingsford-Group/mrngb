from preprocess_datasets import *

UM_TASKS = {
    'omniglot': lambda w, s, q, nt=20000: PreprocessedTaskSet('omniglot', w, s, q, nt),
    'mini-imagenet': lambda w, s, q, nt=20000: PreprocessedTaskSet('mini-imagenet', w, s, q, nt),
    'mini-imagenet-hard': lambda w, s, q, nt=20000: PreprocessedTaskSet('mini-imagenet', w, s, q, nt),
    'jigsaw-omniglot': lambda w, s, q, nt=20000: JigsawTaskSet('omniglot', w, s, q, 2, 2, nt),
    'jigsaw-mini-imagenet': lambda w, s, q, nt=20000: JigsawTaskSet('mini-imagenet', w, s, q, 2, 2, nt),
    'jigsaw-mini-imagenet-hard': lambda w, s, q, nt=20000: JigsawTaskSet('mini-imagenet', w, s, q, 2, 2, nt),
    'jigsaw-44-mini-imagenet': lambda w, s, q, nt=20000: JigsawTaskSet('mini-imagenet', w, s, q, 4, 4, nt),
    'flower': lambda w, s, q, nt=20000: UnprocessedTaskSet('flower', w, s, q, nt),
    'aircraft': lambda w, s, q, nt=20000: UnprocessedTaskSet('aircraft', w, s, q, nt),
    'fungi': lambda w, s, q, nt=20000: UnprocessedTaskSet('fungi', w, s, q, nt),
    'birds': lambda w, s, q, nt=20000: UnprocessedTaskSet('birds', w, s, q, nt),
}
MM_TASKS = {
    'faf': [('flower', 0), ('aircraft', 16000), ('fungi', 32000)],
    'faf-same-start': [('flower', 0), ('aircraft', 0), ('fungi', 0)],
}

TASKS = {}
for umt in UM_TASKS.keys():
    TASKS[umt] = UM_TASKS[umt]
for mmt in MM_TASKS.keys():
    TASKS[mmt] = lambda w, s, q, nt=20000: MultiModalTaskSet(mmt, w, s, q, nt)

class TaskSet:
    def __init__(self, ways, shots, queries):
        self.ways, self.shots, self.queries = ways, shots, queries

    def split_batch(self, batch):
        data, labels = batch
        data, labels = data.to(device), labels.to(device)
        # Separate data into adaptation/evaluation sets
        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        selection = np.arange(self.ways) * (self.shots + self.queries)
        for offset in range(self.shots):
            adaptation_indices[selection + offset] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)
        adapt_data, adapt_labels = data[adaptation_indices], labels[adaptation_indices]
        eval_data, eval_labels = data[evaluation_indices], labels[evaluation_indices]
        return adapt_data, adapt_labels, eval_data, eval_labels

    def sample_task(self, mode):
        pass

class PreprocessedTaskSet(TaskSet):
    def __init__(self, task, ways, shots, queries, num_tasks=20000):
        super(PreprocessedTaskSet, self).__init__(ways, shots, queries)
        self.task = task
        self.tasksets = l2l.vision.benchmarks.get_tasksets(
            task, root = '~/data',
            train_ways = ways,
            train_samples= shots + queries,
            test_ways = ways,
            test_samples = shots + queries,
            num_tasks = num_tasks
        )

    def sample_task(self, mode='train'):
        if mode=='train':
            return self.tasksets.train.sample()
        elif mode=='validation':
            return self.tasksets.validation.sample()
        elif mode=='test':
            return self.tasksets.test.sample()

class JigsawTaskSet(PreprocessedTaskSet):
    def __init__(self, task, ways, shots, queries, w_seg, h_seg, num_tasks=20000):
        super(JigsawTaskSet, self).__init__(task, ways, shots, queries, num_tasks)
        self.task = task
        self.w_seg, self.h_seg = w_seg, h_seg

    def permute_image(self, x, perm):
        n, c, w, h = x.shape
        x = x.unflatten(-2, (self.w_seg, w // self.w_seg))
        x = x.unflatten(-1, (self.h_seg, h // self.h_seg))
        x = x.transpose(-2, -3).flatten(2, 3)
        x = x[:, :, perm, :, :]
        x = x.unflatten(2, (self.w_seg, self.h_seg))
        x = x.transpose(-2, -3)
        x = x.flatten(-2, -1)
        x = x.flatten(-3, -2)
        return x

    def sample_task(self, mode='train'):
        perm = torch.randperm(self.w_seg * self.h_seg)
        if mode=='train':
            data, labels = self.tasksets.train.sample()
            return self.permute_image(data, perm), labels
        elif mode=='validation':
            data, labels = self.tasksets.validation.sample()
            return self.permute_image(data, perm), labels
        elif mode=='test':
            data, labels = self.tasksets.test.sample()
            return self.permute_image(data, perm), labels

class UnprocessedTaskSet(TaskSet):
    def __init__(self, task, ways, shots, queries, num_tasks):
        super(UnprocessedTaskSet, self).__init__(ways, shots, queries)
        datasets, transforms = standard_preprocess_tasksets(
            task, root = '~/data',
            train_ways = ways,
            train_samples= shots + queries,
            test_ways = ways,
            test_samples = shots + queries,
        )
        train_dataset, validation_dataset, test_dataset = datasets
        train_transforms, validation_transforms, test_transforms = transforms

        # Instantiate the tasksets
        self.train_tasks = l2l.data.TaskDataset(
            dataset=train_dataset,
            task_transforms=train_transforms,
            num_tasks=num_tasks,
        )
        self.validation_tasks = l2l.data.TaskDataset(
            dataset=validation_dataset,
            task_transforms=validation_transforms,
            num_tasks=num_tasks,
        )
        self.test_tasks = l2l.data.TaskDataset(
            dataset=test_dataset,
            task_transforms=test_transforms,
            num_tasks=num_tasks,
        )

    def sample_task(self, mode='train'):
        if mode=='train':
            return self.train_tasks.sample()
        elif mode=='validation':
            return self.validation_tasks.sample()
        elif mode=='test':
            return self.test_tasks.sample()


class MultiModalTaskSet(TaskSet):
    def __init__(self, task, ways, shots, queries, num_tasks):
        super(MultiModalTaskSet, self).__init__(ways, shots, queries)
        self.task_list, self.inject_points, self.task_name = [], [], []
        for task_name, inject_pt in MM_TASKS[task]:
            self.task_name.append(task_name)
            self.inject_points.append(inject_pt)
            self.task_list.append(UM_TASKS[task_name](ways, shots, queries, num_tasks))
        self.inject_points_np = np.array(self.inject_points)
        self.epoch_counter = 0

    def sample_group(self):
        active_groups = self.inject_points_np <= self.epoch_counter
        return np.random.choice(len(self.task_list), p= active_groups / np.sum(active_groups))

    def sample_task(self, mode='train'):
        if self.epoch_counter in self.inject_points:
            print(f'Injecting {self.task_name[self.inject_points.index(self.epoch_counter)]} dataset')
        task_group = self.sample_group()
        self.epoch_counter += 1

        return self.task_list[task_group].sample_task(mode)