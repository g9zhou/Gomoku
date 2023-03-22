class config:
    # env config
    render_train = False
    render_test = False
    overwrite_render = True
    record = False

    # output config
    output_path = "results/"
    model_output = output_path + "model.weights"
    log_path = output_path + "log.txt"
    plot_output = output_path + "scores.png"

    # model and training config
    num_episodes_test = 50
    grad_clip = True
    clip_val = 100
    saving_freq = 250000
    log_freq = 50
    eval_freq = 25000
#    record_freq = 10000
    soft_epsilon = 0.05

    # hyper params
    nsteps_train = 1000000  # 10000
    batch_size = 128
    buffer_size = 100000
    target_update_freq = 1000
    gamma = 0.99
    learning_freq = 1
    state_history = 1
    lr_begin = 0.0025
    lr_end = 0.0005
    lr_nsteps = 500000
    eps_begin = 1
    eps_end = 0.01
    eps_nsteps = 800000
    learning_start = 5000
