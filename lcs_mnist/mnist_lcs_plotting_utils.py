# 2024-05-17 
# This script contains various utility functions for plotting the MNIST LCS results

# %% LIBRARY IMPORT

from lcs.plotting_utils import *
import matplotlib.pyplot as plt
import os, pickle
from functools import partial

# %% READ IN DATA

def read_in_data(run_timestamp, results_folder='../results/mnist/', label=None):
    if label is not None:
        modifier = '_' + label
    else:
        modifier = ''
        
    with open(os.path.join(results_folder, f'{run_timestamp}{modifier}_run_info.pkl'), 'rb') as f:
        run_info = pickle.load(f)

        t = run_info['t']
        loss = run_info['loss']
        try:
            c = run_info['c1']
        except KeyError:
            print("This model does not represent contexts.")
            c = None
        norm_W1 = run_info['norm_W1']
        acc = run_info['context_alignment1']
        context = run_info['c_gt1']

        cfg = run_info['cfg']

    return t, loss, c, norm_W1, acc, context, cfg


# %% PLOTTING

def indicate_contexts(ax, t, cfg, ):
    for i, t_switch in enumerate(np.arange(0, cfg.block_duration*cfg.num_blocks, cfg.block_duration)):
                switch_point = int(0.75*cfg.num_blocks)
                if cfg.shared_teachers:
                    if cfg.c_gt_curriculum == 'B_AB__A_B_AB__':
                        if i < switch_point:
                            c = f'C{(i % cfg.num_contexts) + 1}'
                        else:
                            c = f'C{(i % (cfg.num_contexts+cfg.num_shared_contexts))}'
                    elif cfg.c_gt_curriculum == 'B_AB__A_B__':
                        if i < switch_point:
                            c = f'C{(i % cfg.num_contexts) + 1}'
                        else:
                            c = f'C{(i % cfg.num_contexts)}'
                    elif cfg.c_gt_curriculum == 'A_B__AB__':
                        if i < switch_point:
                            c = f'C{(i % cfg.num_contexts)}'
                        else:
                            c = 'C2'
                    elif cfg.c_gt_curriculum == 'AB_BC__':
                        c = f'C{((i % (cfg.num_contexts-1)) + cfg.num_contexts)}'
                    elif cfg.c_gt_curriculum == 'A_B_AB__':
                        c = f'C{i % (cfg.num_contexts + cfg.num_shared_contexts)}'
                    elif cfg.c_gt_curriculum == 'A_B_C__AB_BC_CA__':
                        if i < switch_point:
                            c = f'C{(i % cfg.num_contexts)}'
                        else:
                            c = f'C{(i % cfg.num_contexts)+3}'
                    elif cfg.c_gt_curriculum == 'AB_BC_CA__':
                        c = f'C{((i % (cfg.num_contexts)) + cfg.num_contexts)}'
                    elif cfg.c_gt_curriculum == 'AB_CD__AD__': # hard-coded
                        if i < switch_point:
                            if i % 2 == 0:
                                c = 'C4'
                            else:
                                c = 'C6'
                        else:
                            c = 'C7'
                if cfg.c_gt_curriculum == 'A_B__':
                    c = f'C{i % cfg.num_contexts}'
                ax.axvspan(t_switch, t_switch+cfg.block_duration, color=c, alpha=0.1, zorder=-10)


# %%
 
def calc_block_intervals(context):

    block_intervals = []

    current_context = 0
    current_start = 0
    for i in range(1, len(context)):
        if context[i] != current_context:
            block_intervals.append((current_start, i))
            current_start = i
            current_context = context[i]

    return block_intervals

# %% FOR EACH CONTEXT, DETERMINE WHEN WE CROSS THRESHOLD

def determine_steps_to_threshold(acc, block_intervals, threshold_acc = 0.85):

    times_to_threshold = []

    for block_interval in block_intervals:
        current_block = acc[block_interval[0]:block_interval[1]]

        ## find first time we cross threshold
        for i in range(len(current_block)):
            if current_block[i] > threshold_acc:
                times_to_threshold.append(i)
                break
            
            if i == len(current_block) - 1:
                times_to_threshold.append(i)

    return times_to_threshold

# %% 

def calc_specificity_at_threshold(c, times_to_threshold, block_intervals, sim_metric = 'euclidean'):
    specificity_at_threshold = []

    for i in range(1, len(times_to_threshold)):
        c_at_threshold = c[block_intervals[i][0] + times_to_threshold[i]]
        c_at_threshold_sorted = np.sort(c_at_threshold)

        print(c_at_threshold_sorted)

        ## specificity at threshold
        if sim_metric == 'euclidean':
            #sim = 1 - np.linalg.norm(c_at_threshold_sorted - [1,0])
            sim = np.linalg.norm(c_at_threshold_sorted - [1,0])
        if sim_metric == 'cossim':
            sim = np.dot(c_at_threshold_sorted, [1,0]) / (np.linalg.norm(c_at_threshold_sorted) * np.linalg.norm([1,0]))

        specificity_at_threshold.append(sim)

    return specificity_at_threshold

@partial(np.vectorize, signature='(t),(t),(),()->(n),(n)')
def get_first_crossing_times(loss, c_gt1, cfg, thsd=0.1):
    ti_switches = np.where(np.abs(np.gradient(c_gt1) / cfg.dt_tape) > 0.2)[0] + 1

    losses_blocks = np.split(loss, ti_switches)
    losses_blocks = [loss_block for loss_block in losses_blocks if len(loss_block) > 2]
    blocks = np.arange(1, len(losses_blocks)+1)

    idcs_first = []
    for loss_block in losses_blocks:
        idx = np.where(loss_block < thsd)[0] # gets first index where loss is below threshold
        idcs_first.append(idx[0] if len(idx) > 0 else len(loss_block)-1)

    idcs_first = np.array(idcs_first)
    return blocks, idcs_first


# %% PLOT WCST

def plot_wcst(ax, accs, contexts, labels = ['LCS', 'Control'], model_colors = ['black', 'red'], start_block = 1, threshold_acc = 0.85):
    
    for i in range(len(accs)):
        context = np.array(contexts)[i]
        print(context)
        block_intervals = calc_block_intervals(context)
        n_blocks = len(block_intervals)
        print(block_intervals)

        timess_to_threshold = []
        for j in range(len(accs[i])):
            timess_to_threshold.append(determine_steps_to_threshold(accs[i][j], block_intervals, threshold_acc))
        print(timess_to_threshold)

        timess_to_threshold = np.array(timess_to_threshold)
        means = np.mean(timess_to_threshold, axis=0)
        stderrs = np.std(timess_to_threshold, axis=0) / np.sqrt(len(timess_to_threshold))

        ax.plot(np.arange(start_block, n_blocks), means[start_block:], label=labels[i], c=model_colors[i])
        ax.fill_between(np.arange(start_block, n_blocks), means[start_block:] - stderrs[start_block:], means[start_block:] + stderrs[start_block:], alpha=0.3, color=model_colors[i])

    ax.set_xlabel('Block (starting from %d)'%start_block)
    ax.set_ylabel('T to threshold')

    ax.legend()

    return ax

# %%

def plot_specificity(ax, c, accs, contexts, labels = ['LCS', 'Control'], model_colors = ['black', 'red'], sim_metric = 'euclidean', threshold_acc = 0.85):
    
    for i in range(len(accs)):
        context = np.array(contexts)[i]
        block_intervals = calc_block_intervals(context)
        n_blocks = len(block_intervals)
        
        specificities = []
        for j in range(len(accs[i])):
            times_to_threshold = determine_steps_to_threshold(accs[i][j], block_intervals, threshold_acc)
            specificities.append(calc_specificity_at_threshold(c, times_to_threshold, block_intervals, sim_metric = sim_metric))

        specificities = np.array(specificities)
        means = np.mean(specificities, axis=0)
        stderrs = np.std(specificities, axis=0) / np.sqrt(len(specificities))

        ax.plot(np.arange(1, n_blocks), means, label=labels[i], c=model_colors[i])
        ax.fill_between(np.arange(1, n_blocks), means - stderrs, means + stderrs, alpha=0.3, color=model_colors[i])
        # ax.plot(np.arange(1, n_blocks), specificity_at_threshold, label=labels[i], c=model_colors[i])
        # times_to_threshold = determine_steps_to_threshold(accs[i], block_intervals, threshold_acc)
        # specificity_at_threshold = calc_specificity_at_threshold(c, times_to_threshold, block_intervals, sim_metric = sim_metric)
        # ax.plot(np.arange(1, n_blocks), specificity_at_threshold, label=labels[i], c=model_colors[i])

    ax.set_xlabel('Block')
    #ax.set_ylabel('Specificity')
    ax.set_ylabel('Inspecificity')

    ax.legend()

    return ax
# %%
