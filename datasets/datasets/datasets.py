import copy


datasets = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(dataset_spec, args=None):
    if args is not None:
        dataset_args = copy.deepcopy(dataset_spec['args'])
        dataset_args.update(args)
    else:
        dataset_args = dataset_spec['args']
    for k, v in dataset_args.items():
        if isinstance(v, dict) and 'name' in v and v['name'] in datasets:
            # If it is, recursively call 'make' on it.
            dataset_args[k] = make(v)
    dataset = datasets[dataset_spec['name']](**dataset_args)
    return dataset
