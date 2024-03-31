import matplotlib.pyplot as plt

def vis_batch_elem(batch_element, savedir='tmp'):
    plt.figure()
    for state in batch_element.agent_future_np:
        plt.scatter(state[0], state[1], c='r')
    for state in batch_element.agent_history_np:
        plt.scatter(state[0], state[1], c='pink')
    for neigh in batch_element.neighbor_futures:
        for state in neigh:
            plt.scatter(state[0], state[1], c='b')
    for neigh in batch_element.neighbor_histories:
        for state in neigh:
            plt.scatter(state[0], state[1], c='cyan')
    for li in range(batch_element.lane_positions.shape[0]):
        pos = batch_element.lane_positions[li]
        pad = batch_element.lane_paddings[li]
        padpos = pos[~pad]
        plt.plot(padpos[:,0], padpos[:,1], c='grey')
    plt.xlim([-20,80])
    plt.ylim([-50,50])
    plt.savefig(f'{savedir}/{batch_element.scene_id}_{batch_element.scene_ts}.jpg')
    plt.close()

def vis_fmae_data(data, savedir='tmp'):
    plt.figure()
    for state in data['x_positions']:
        plt.scatter(state[:,0], state[:,1], c='r', s=10)
    for li in range(data['lane_positions'].shape[0]):
        pos = data['lane_positions'][li]
        pad = data['lane_padding_mask'][li]
        padpos = pos[~pad]
        plt.plot(padpos[:,0], padpos[:,1], c='grey')
    plt.xlim([-20,80])
    plt.ylim([-50,50])
    scene_id = data['scenario_id']
    plt.savefig(f'{savedir}/{scene_id}.jpg')
    plt.close()