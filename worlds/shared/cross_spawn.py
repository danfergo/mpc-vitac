def cross_spawn(fn, options, parallel=1):
    from multiprocessing import Process

    r_process = {'i': 0, 'processes': []}
    options_keys = list(options.keys())
    options_keys.reverse()

    def iterate(ptr, values):
        if ptr < len(options_keys):
            key = options_keys[ptr]
            for v in range(*options[key]):
                iterate(ptr + 1, {
                    **values,
                    key: v
                })
        else:
            values = {'i': r_process['i'], **values}
            print('RUN', values)
            p = Process(target=fn, kwargs=values)
            p.start()
            r_process['processes'].append(p)

            if len(r_process['processes']) >= parallel:
                print('Waiting...')
                [p.join() for p in r_process['processes']]
                r_process['processes'] = []

            r_process['i'] = r_process['i'] + 1

    iterate(0, {})
