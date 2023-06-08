from multiprocessing import Process


def _is_iterable(x):
    try:
        _ = iter(x)
        return True
    except TypeError as te:
        return False


def gen_options_list(options):
    options_keys = list(options.keys())
    options_keys.reverse()
    options_list = []

    def iterate(ptr, values):
        if ptr < len(options_keys):
            key = options_keys[ptr]
            iterable = options[key] if _is_iterable(options[key]) else [options[key]]
            for v in iterable:
                iterate(ptr + 1, {
                    **values,
                    key: v
                })
        else:
            op = {'i': len(options_list), **{k: v() if callable(v) else v for k, v in values.items()}}
            options_list.append(op)

    iterate(0, {})

    return options_list


def run_parallel(fn, options, parallel=1):
    # wtf python, this solves hanging opencv/numpy
    # when running multiple processes.
    import multiprocessing

    multiprocessing.set_start_method('spawn')

    options_list = gen_options_list(options)
    r_processes = []

    for op in options_list:
        p = Process(target=fn, kwargs=op)
        r_processes.append(p)

        print('RUN', op)
        p.start()

        if len(r_processes) >= parallel:
            print('Waiting...')
            [p.join() for p in r_processes]
            r_processes = []

    # r_process['i'] = r_process['i'] + 1


def run_sequential(fn, options):
    options_list = gen_options_list(options)

    for op in options_list:
        print('RUN', op)
        fn(**op)


def run_all(fn, options, parallel=1):
    if parallel == 1:
        run_sequential(fn, options)
    else:
        run_parallel(fn, options, parallel)
