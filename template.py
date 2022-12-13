def set_template(args):
    # Set the templates here
    args.template = args.template.lower()

    if args.template.find('rgb') >= 0:
        args.n_colors = 3

    if args.template.find('fm') >= 0:
        args.model = "metafmrdn"

    if args.template.find('basic') >= 0:
        args.ext = 'sep'
        args.lr_decay = 25
        args.epochs = 250
        args.save_results = True

    if args.template.find('std') >= 0:
        args.model = 'metafrdn'
        args.tail_type = 'multi'
        args.mix_type = 'average'
        args.ca_type = 'none'
        args.head_blocks = 8
        args.body_blocks = 8
        args.tail_blocks = 4

    if args.template.find('ntire') >= 0:
        args.data_train = 'NTIRE'
        args.data_test = 'NTIRE_VAL'

    if args.template.find('icvl') >= 0:
        args.data_train = 'ICVL'
        args.data_test = 'ICVL_VAL'

    if args.template.find('foster') >= 0:
        args.data_train = 'Foster'
        args.data_test = 'Foster_VAL'

    if args.template.find('CAVE') >= 0:
        args.data_train = 'CAVE'
        args.data_test = 'CAVE_VAL'

    if args.template.find('baseline') >= 0:
        args.max_in_channel = 0
        args.min_in_channel = 0
        args.wl_in_type = 'all'
        args.tail_type = 'multi'
        args.wl_out_type = 'max'

    if args.template.find('exter') >= 0:
        args.max_in_channel = 0
        args.min_in_channel = 0
        args.wl_out_type = 'comp'
