import torch
from torch.nn.utils.rnn import pad_sequence

def padNadd_output_batch(output_mae_cat, output_mae, test_batch_cat, test_batch, max_fut_len, historical_steps):

    output_mae['y_mae_hat'] = output_mae['y_mae_hat'][output_mae['future_pred_mask']].unsqueeze(0)
    output_mae['x_mae_hat'] = output_mae['x_mae_hat'][output_mae['hist_pred_mask']].unsqueeze(0)
    output_mae['lane_mae_pred'] = output_mae['lane_mae_pred'][output_mae['lane_pred_mask']].unsqueeze(0)

    test_batch['y_agent'] = test_batch['y'][:,:1]
    test_batch['y_4_mae'] = test_batch['y'][output_mae['future_pred_mask']].unsqueeze(0)
    test_batch['x_positions_4_mae'] = test_batch['x_positions'][output_mae['hist_pred_mask']].unsqueeze(0)
    test_batch['x_centers_4_mae'] = test_batch['x_centers'][output_mae['hist_pred_mask']].unsqueeze(0)

    test_batch['x_padding_hist_4_mae'] = test_batch["x_padding_mask"][:, :, :historical_steps][output_mae['hist_pred_mask']].unsqueeze(0)
    test_batch['x_padding_fut_4_mae'] = test_batch["x_padding_mask"][:, :, historical_steps:][output_mae['future_pred_mask']].unsqueeze(0)
    
    output_mae['future_pred_mask'] = output_mae['future_pred_mask'][output_mae['future_pred_mask']].unsqueeze(0)
    output_mae['hist_pred_mask'] = output_mae['hist_pred_mask'][output_mae['hist_pred_mask']].unsqueeze(0)
    output_mae['lane_pred_mask'] = output_mae['lane_pred_mask'][output_mae['lane_pred_mask']].unsqueeze(0)

    if len(output_mae_cat) == 0:
        output_mae_cat.update(output_mae)
        test_batch_cat.update(test_batch)
    else:
        for key in ['lane_mae_pred', 'lane_pred_mask', 'x_mae_hat', 'hist_pred_mask', 'y_mae_hat', 'future_pred_mask', 'y_hat']:
            if output_mae_cat[key].size(0) < max_fut_len:
                output_mae_cat[key] = pad_sequence([*output_mae_cat[key],output_mae[key][0]], batch_first=True)
            else:
                output_mae_cat[key] = pad_sequence([*output_mae_cat[key][1:],output_mae[key][0]], batch_first=True)
        for key in ['y_agent', 'y_4_mae', 'x_positions_4_mae', 'x_centers_4_mae', 'x_padding_hist_4_mae', 'x_padding_fut_4_mae']:
            if test_batch_cat[key].size(0) < max_fut_len:
                test_batch_cat[key] = pad_sequence([*test_batch_cat[key],test_batch[key][0]], batch_first=True)
            else:
                test_batch_cat[key] = pad_sequence([*test_batch_cat[key][1:],test_batch[key][0]], batch_first=True)


    return output_mae_cat
    


def padNadd_batch(test_batch_cat, test_batch, max_fut_len, padding_mask, historical_steps):
    if len(test_batch_cat) == 0:
        test_batch_cat.update(test_batch)
    else:
        for key in test_batch_cat.keys():
            if key in ['num_actors', 'num_lanes', 'scenario_id', 'scene_ts', 'track_id', 'origin', 'theta', 'actor_names']:
                continue
            else:
                if test_batch_cat[key].size(0) < max_fut_len:
                    test_batch_cat[key] = pad_sequence([*test_batch_cat[key],test_batch[key][0]], batch_first=True)
                else:
                    test_batch_cat[key] = pad_sequence([*test_batch_cat[key][1:],test_batch[key][0]], batch_first=True)
    return test_batch_cat