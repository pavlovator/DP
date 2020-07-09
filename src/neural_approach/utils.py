def update_history_RV1(history, ep, t_l, v_l, t_mae, v_mae):
    history["epoch"].append(ep)
    history["train_loss"].append(t_l)
    history["validation_loss"].append(v_l)
    history["train_MAE"].append(t_mae)
    history["validation_MAE"].append(v_mae)


def update_history_CV1(history, ep, t_l, v_l, t_a_a, v_a_a, t_a_i, v_a_i):
    history["epoch"].append(ep)
    history["train_loss"].append(t_l)
    history["validation_loss"].append(v_l)
    history["train_acc_arrow"].append(t_a_a)
    history["validation_acc_arrow"].append(v_a_a)
    history["train_acc_image"].append(t_a_i)
    history["validation_acc_image"].append(v_a_i)