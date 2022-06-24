import numpy as np
import matplotlib.pyplot as plt



setting = 'informer_ETTh1_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0'
preds = np.load('./results/'+setting+'/pred.npy')
trues = np.load('./results/'+setting+'/true.npy')
metrics = np.load('./results/'+setting+'/metrics.npy')
real_pred = np.load('./results/'+setting+'/real_prediction.npy')
print(metrics)

# draw OT prediction 油温预测和gd
plt.figure()
plt.plot(trues[0,:,-1], label='GroundTruth')
plt.plot(preds[0,:,-1], label='Prediction')
# plt.plot(real_pred[0,:,-1],label = 'real-pred')   # 真实预测同样也是很小？不知道为啥
plt.legend()
plt.savefig('./img/OT_gd_pred_ETTm1_test0.png')
plt.show()


# # draw HUFL prediction
# plt.figure()
# #plt.plot(trues[0,:,0], label='GroundTruth')
# plt.plot(preds[0,:,0], label='Prediction')
# plt.legend()
# plt.savefig('./img/HUFL_gd_pred_ETTm1_test0.png')
# plt.show()


