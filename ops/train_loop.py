# train_loop.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn


class hook:

    def __init__(self, start, freq, fn):
        self.start = start
        self.freqency = freq
        self.function = fn

    def __call__(self, cur_step, *args, **kwargs):
        if cur_step > self.start and cur_step % self.freqency == 0:
            self.function(*args, **kwargs)


def train_loop(dataset, step_fn, variables=None, global_level_hooks=None,
               local_level_hooks=None, epoch_level_hooks=None):
    if variables is None:
        variables = {}

    if "cost" not in variables or variables["cost"] is None:
        variables["cost"] = 0.0

    if "step" not in variables or variables["step"] is None:
        variables["step"] = 0

    if "epoch" not in variables or variables["epoch"] is None:
        variables["epoch"] = 1

    if "global_cost" not in variables or variables["global_cost"] is None:
        variables["global_cost"] = 0.0

    if "global_step" not in variables or variables["global_step"] is None:
        variables["global_step"] = 0

    if "local_cost" not in variables or variables["local_cost"] is None:
        variables["local_cost"] = 0.0

    if "local_step" not in variables or variables["local_step"] is None:
        variables["local_step"] = 0

    if not isinstance(global_level_hooks, (list, tuple)):
        if global_level_hooks is None:
            global_level_hooks = []
        else:
            global_level_hooks = [global_level_hooks]

    if not isinstance(local_level_hooks, (list, tuple)):
        if local_level_hooks is None:
            local_level_hooks = []
        else:
            local_level_hooks = [local_level_hooks]

    if not isinstance(epoch_level_hooks, (list, tuple)):
        if epoch_level_hooks is None:
            epoch_level_hooks = []
        else:
            epoch_level_hooks = [epoch_level_hooks]

    while True:
        try:
            for data in dataset:
                variables["step"] += 1
                variables["global_step"] += 1

                cost = step_fn(data, **variables)

                variables["cost"] += cost
                variables["global_cost"] += cost

                for hook in local_level_hooks:
                    hook(variables["step"], data, **variables)

                for hook in global_level_hooks:
                    hook(variables["global_step"], data, **variables)

            variables["epoch"] += 1
            variables["local_cost"] = variables["cost"]
            variables["local_step"] = variables["step"]
            variables["cost"] = 0
            variables["step"] = 0

            for hook in epoch_level_hooks:
                hook(variables["epoch"], data, **variables)



            dataset.reset()
        except StopIteration:
            return
