import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree
from sklearn.metrics import f1_score
import graphviz

train_df = pd.read_csv('data/train.csv')
print(train_df.head(5))
train_time = train_df['time'].values
train_signal = train_df['signal'].values
train_channel = train_df['open_channels'].values
train_length = train_time.shape[0]
test_df = pd.read_csv('data/test.csv')
test_time = test_df['time'].values
test_signal = test_df['signal'].values
test_length = test_time.shape[0]

# ===========train signal================
# fig = plt.figure(figsize=(20, 5))
# step = 1000
# plt.plot(range(0, train_length, step), train_signal[0::step])
# for i in range(11):
#     plt.plot([i * 500000, i * 500000], [-5, 12.5], 'r')
# for i in range(10):
#     plt.text(i * 500000 + 200000, 10, str(i + 1), size=20)
# plt.xlabel('Row',size=16)
# plt.ylabel('Signal',size=16)
# plt.title('Training Data Signal - 10 batches',size=20)
# plt.savefig('EDA/time_signal.png')
# plt.show()

# ===========train channel================
# fig = plt.figure(figsize=(20, 5))
# step = 1000
# plt.plot(range(0, train_length, step), train_channel[0::step])
# for i in range(11):
#     plt.plot([i * 500000, i * 500000], [-5, 12.5], 'r')
# for i in range(10):
#     plt.text(i * 500000 + 200000, 10, str(i + 1), size=20)
# plt.xlabel('Row', size=16)
# plt.ylabel('Channels Open', size=16)
# plt.title('Training Data Open Channels - 10 batches', size=20)
# plt.savefig('EDA/time_channel.png')
# plt.show()

# ===========Correlation between signal and open channels================
# for k in range(5):
#     start = int(np.random.uniform(0, train_length - 5000))
#     end = start + 5000
#     step = 10
#     print('#' * 25)
#     print('### Random {} to {} '.format(start, end))
#     print('#' * 25)
#     plt.figure(figsize=(20, 5))
#     plt.plot(range(start, end, step), train_signal[start:end][0::step])
#     plt.plot(range(start, end, step), train_channel[start:end][0::step])
#     # plt.savefig('EDA/signal_channel.png')
# plt.show()

# ==========Test Data=============
# plt.figure(figsize=(20, 5))
# step = 1000
# label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
# plt.plot(range(0, test_length, step), test_signal[0::step])
# for i in range(5):
#     plt.plot([i * 500000, i * 500000], [-5, 12.5], 'r')
# for j in range(21):
#     plt.plot([j * 100000, j * 100000], [-5, 12.5], 'r:')
# for k in range(4):
#     plt.text(k * 500000 + 200000, 10, str(k + 1), size=20)
# for k in range(10):
#     plt.text(k * 100000 + 40000, 7, label[k], size=16)
# plt.xlabel('Row', size=16)
# plt.ylabel('Channels Open', size=16)
# plt.title('Test Data Signal - 4 batches - 10 subsamples', size=20)
# plt.savefig('EDA/test_data.png')
# plt.show()

# ==========Remove Train Data Drift=============
train_copy = train_df.copy()

a = 500000
b = 600000  # CLEAN TRAIN BATCH 2
signal = train_copy['signal'][a:b].values
time = train_copy['time'][a:b].values
train_copy.loc[train_copy.index[a:b], 'signal'] = signal - 3 * (time - 50) / 10.


# batch = 2
# a = 500000 * (batch - 1)
# b = 500000 * batch
# res = 50
# plt.figure(figsize=(20, 5))
# plt.plot(range(a, b, res), train_copy['signal'][a:b][0::res])
# plt.title('Training Batch 2 without Slant Drift', size=16)
# plt.savefig('EDA/without Slant Drift.png')
# plt.figure(figsize=(20, 5))
# plt.plot(range(a, b, res), train_signal[a:b][0::res])
# plt.title('Training Batch 2 with Slant Drift', size=16)
# plt.savefig('EDA/with Slant Drift.png')
# plt.show()

# 一元二次方程
def f(x, low, high, mid):
    return -((-low + high) / 625) * (x - mid) ** 2 + high - low


# CLEAN TRAIN BATCH 7
batch = 7
a = 500000 * (batch - 1)
b = 500000 * batch
train_copy.loc[train_copy.index[a:b], 'signal'] = train_signal[a:b] - f(train_time[a:b], -1.817, 3.186, 325)
# CLEAN TRAIN BATCH 8
batch = 8
a = 500000 * (batch - 1)
b = 500000 * batch
train_copy.loc[train_copy.index[a:b], 'signal'] = train_signal[a:b] - f(train_time[a:b], -0.094, 4.936, 375)
# CLEAN TRAIN BATCH 9
batch = 9
a = 500000 * (batch - 1)
b = 500000 * batch
train_copy.loc[train_copy.index[a:b], 'signal'] = train_signal[a:b] - f(train_time[a:b], 1.715, 6.689, 425)
# CLEAN TRAIN BATCH 10
batch = 10
a = 500000 * (batch - 1)
b = 500000 * batch
train_copy.loc[train_copy.index[a:b], 'signal'] = train_signal[a:b] - f(train_time[a:b], 3.361, 8.45, 475)

#可视化
# plt.figure(figsize=(20, 5))
# plt.plot(train_time[::1000], train_signal[::1000])
# plt.title('Training Batches 7-10 with Parabolic Drift', size=16)
# plt.savefig('EDA/with Parabolic Drift.png')
# plt.figure(figsize=(20, 5))
# plt.plot(train_copy['time'][::1000], train_copy['signal'][::1000])
# plt.title('Training Batches 7-10 without Parabolic Drift', size=16)
# plt.savefig('EDA/without Parabolic Drift.png')
# plt.show()

# =============================Make Five Simple Models======================
# ================batch 1和2使用Slow Open Channel======================
batch = 1
a = 500000 * (batch - 1);
b = 500000 * batch
batch = 2
c = 500000 * (batch - 1)
d = 500000 * batch
temp = np.concatenate([train_copy['signal'].values[a:b], train_copy['signal'].values[c:d]])
X_train = np.concatenate([train_copy['signal'].values[a:b], train_copy['signal'].values[c:d]]).reshape((-1, 1))
y_train = np.concatenate([train_copy['open_channels'].values[a:b], train_copy['open_channels'].values[c:d]]).reshape(
    (-1, 1))

clf1s = tree.DecisionTreeClassifier(max_depth=1)
clf1s = clf1s.fit(X_train, y_train)
print('Training model 1s channel')
preds = clf1s.predict(X_train)
print('has f1 validation score =', f1_score(y_train, preds, average='macro'))

# tree_graph = tree.export_graphviz(clf1s, out_file=None, max_depth=10,
#                                   impurity=False, feature_names=['signal'], class_names=['0', '1'],
#                                   rounded=True, filled=True)
# graph = graphviz.Source(tree_graph)
# graph.view('Slow Open Channel')

# ================batch 3和7使用Fast Open Channel======================
batch = 3
a = 500000 * (batch - 1)
b = 500000 * batch
batch = 7
c = 500000 * (batch - 1)
d = 500000 * batch
X_train = np.concatenate([train_copy['signal'].values[a:b], train_copy['signal'].values[c:d]]).reshape((-1, 1))
y_train = np.concatenate([train_copy['open_channels'].values[a:b], train_copy['open_channels'].values[c:d]]).reshape(
    (-1, 1))

clf1f = tree.DecisionTreeClassifier(max_depth=1)
clf1f = clf1f.fit(X_train, y_train)
print('Training model 1f channel')
preds = clf1f.predict(X_train)
print('has f1 validation score =', f1_score(y_train, preds, average='macro'))

# tree_graph = tree.export_graphviz(clf1f, out_file=None, max_depth=10,
#                                   impurity=False, feature_names=['signal'], class_names=['0', '1'],
#                                   rounded=True, filled=True)
# graph = graphviz.Source(tree_graph)
# graph.view('Fast Open Channel')

# ================batch 4和8使用 Maximum 3 Open Channel======================
batch = 4
a = 500000 * (batch - 1)
b = 500000 * batch
batch = 8
c = 500000 * (batch - 1)
d = 500000 * batch
X_train = np.concatenate([train_copy['signal'].values[a:b], train_copy['signal'].values[c:d]]).reshape((-1, 1))
y_train = np.concatenate([train_copy['open_channels'].values[a:b], train_copy['open_channels'].values[c:d]]).reshape(
    (-1, 1))

clf3 = tree.DecisionTreeClassifier(max_leaf_nodes=4)
clf3 = clf3.fit(X_train, y_train)
print('Training model 3 channel')
preds = clf3.predict(X_train)
print('has f1 validation score =', f1_score(y_train, preds, average='macro'))

# tree_graph = tree.export_graphviz(clf3, out_file=None, max_depth=10,
#                                   impurity=False, feature_names=['signal'], class_names=['0', '1', '2', '3'],
#                                   rounded=True, filled=True)
# graph = graphviz.Source(tree_graph)
# graph.view('Maximum 3 Open Channel')

# ================batch 6和9使用 Maximum 5 Open Channel======================
batch = 6
a = 500000 * (batch - 1)
b = 500000 * batch
batch = 9
c = 500000 * (batch - 1)
d = 500000 * batch
X_train = np.concatenate([train_copy['signal'].values[a:b], train_copy['signal'].values[c:d]]).reshape((-1, 1))
y_train = np.concatenate([train_copy['open_channels'].values[a:b], train_copy['open_channels'].values[c:d]]).reshape(
    (-1, 1))

clf5 = tree.DecisionTreeClassifier(max_leaf_nodes=6)
clf5 = clf5.fit(X_train, y_train)
print('Trained model 5 channel')
preds = clf5.predict(X_train)
print('has f1 validation score =', f1_score(y_train, preds, average='macro'))

# tree_graph = tree.export_graphviz(clf5, out_file=None, max_depth=10,
#                                   impurity=False, feature_names=['signal'], class_names=['0', '1', '2', '3', '4', '5'],
#                                   rounded=True, filled=True)
# graph = graphviz.Source(tree_graph)
# graph.view('Maximum 5 Open Channel')

# ================batch 5和10使用 Maximum 10 Open Channel======================
batch = 5
a = 500000 * (batch - 1)
b = 500000 * batch
batch = 10
c = 500000 * (batch - 1)
d = 500000 * batch
X_train = np.concatenate([train_copy['signal'].values[a:b], train_copy['signal'].values[c:d]]).reshape((-1, 1))
y_train = np.concatenate([train_copy['open_channels'].values[a:b], train_copy['open_channels'].values[c:d]]).reshape(
    (-1, 1))

clf10 = tree.DecisionTreeClassifier(max_leaf_nodes=8)
clf10 = clf10.fit(X_train, y_train)
print('Trained model 10 channel')
preds = clf10.predict(X_train)
print('has f1 validation score =', f1_score(y_train, preds, average='macro'))

# tree_graph = tree.export_graphviz(clf10, out_file=None, max_depth=10,
#                                   impurity=False, feature_names=['signal'], class_names=[str(x) for x in range(11)],
#                                   rounded=True, filled=True)
# graph = graphviz.Source(tree_graph)
# graph.view('Maximum 10 Open Channel')

# ==================================Analyze Test Data Drift===========================
# ORIGINAL TRAIN DATA
# plt.figure(figsize=(20, 5))
# r = train_df['signal'].rolling(30000).mean()
# plt.plot(train_time, r)
# for i in range(11):
#     plt.plot([i * 50, i * 50], [-3, 8], 'r:')
# for j in range(10):
#     plt.text(j * 50 + 20, 6, str(j + 1), size=20)
# plt.title('Training Signal Rolling Mean. Has Drift wherever plot is not horizontal line', size=16)
# plt.savefig('EDA/Training Signal Rolling Mean with Drift.png')

# TRAIN DATA WITHOUT DRIFT
# plt.figure(figsize=(20, 5))
# r = train_copy['signal'].rolling(30000).mean()
# plt.plot(train_copy['time'].values, r)
# for i in range(11):
#     plt.plot([i * 50, i * 50], [-3, 8], 'r:')
# for j in range(10):
#     plt.text(j * 50 + 20, 6, str(j + 1), size=20)
# plt.title('Training Signal Rolling Mean without Drift', size=16)
# plt.savefig('EDA/Training Signal Rolling Mean without Drift.png')
# plt.show()

# Test Data Drift
# plt.figure(figsize=(20, 5))
# let = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
# r = test_df['signal'].rolling(30000).mean()
# plt.plot(test_df['time'].values, r)
# for i in range(21):
#     plt.plot([500 + i * 10, 500 + i * 10], [-3, 6], 'r:')
# for i in range(5):
#     plt.plot([500 + i * 50, 500 + i * 50], [-3, 6], 'r')
# for k in range(4):
#     plt.text(525 + k * 50, 5.5, str(k + 1), size=20)
# for k in range(10):
#     plt.text(505 + k * 10, 4, let[k], size=16)
# plt.title('Test Signal Rolling Mean. Has Drift wherever plot is not horizontal line', size=16)
# plt.savefig('Test Signal Rolling Mean.png')
# plt.show()

# Remove Test Data Drift
test_copy = test_df.copy()

# REMOVE BATCH 1 DRIFT
start = 500
a = 0
b = 100000

test_copy.loc[test_copy.index[a:b], 'signal'] = test_signal[a:b] - 3 * (test_time[a:b] - start) / 10.
start = 510
a = 100000
b = 200000
test_copy.loc[test_copy.index[a:b], 'signal'] = test_signal[a:b] - 3 * (test_time[a:b] - start) / 10.
start = 540
a = 400000
b = 500000
test_copy.loc[test_copy.index[a:b], 'signal'] = test_signal[a:b] - 3 * (test_time[a:b] - start) / 10.

# REMOVE BATCH 2 DRIFT
start = 560
a = 600000
b = 700000
test_copy.loc[test_copy.index[a:b], 'signal'] = test_signal[a:b] - 3 * (test_time[a:b] - start) / 10.
start = 570
a = 700000
b = 800000
test_copy.loc[test_copy.index[a:b], 'signal'] = test_signal[a:b] - 3 * (test_time[a:b] - start) / 10.
start = 580
a = 800000
b = 900000
test_copy.loc[test_copy.index[a:b], 'signal'] = test_signal[a:b] - 3 * (test_time[a:b] - start) / 10.


# REMOVE BATCH 3 DRIFT
def f(x):
    return -(0.00788) * (x - 625) ** 2 + 2.345 + 2.58


a = 1000000
b = 1500000
test_copy.loc[test_copy.index[a:b], 'signal'] = test_signal[a:b] - f(test_time[a:b])

# let = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
# plt.figure(figsize=(20, 5))
# res = 1000
# plt.plot(range(0, test_copy.shape[0], res), test_copy['signal'][0::res])
# for i in range(5):
#     plt.plot([i * 500000, i * 500000], [-5, 12.5], 'r')
# for i in range(21):
#     plt.plot([i * 100000, i * 100000], [-5, 12.5], 'r:')
# for k in range(4):
#     plt.text(k * 500000 + 250000, 10, str(k + 1), size=20)
# for k in range(10):
#     plt.text(k * 100000 + 40000, 7.5, let[k], size=16)
# plt.title('Test Signal without Drift', size=16)
#
# plt.figure(figsize=(20, 5))
# r = test_df['signal'].rolling(30000).mean()
# plt.plot(test_time, r)
# for i in range(21):
#     plt.plot([500 + i * 10, 500 + i * 10], [-2, 6], 'r:')
# for i in range(5):
#     plt.plot([500 + i * 50, 500 + i * 50], [-2, 6], 'r')
# for k in range(4):
#     plt.text(525 + k * 50, 5.5, str(k + 1), size=20)
# for k in range(10):
#     plt.text(505 + k * 10, 4, let[k], size=16)
# plt.title('Test Signal Rolling Mean without Drift', size=16)
# plt.show()

# =====================Predict Test==================
sub = pd.read_csv('data/sample_submission.csv')

a = 0  # SUBSAMPLE A, Model 1s
sub.iloc[100000 * a:100000 * (a + 1), 1] = clf1s.predict(
    test_copy['signal'].values[100000 * a:100000 * (a + 1)].reshape((-1, 1)))

a = 1  # SUBSAMPLE B, Model 3
sub.iloc[100000 * a:100000 * (a + 1), 1] = clf3.predict(
    test_copy['signal'].values[100000 * a:100000 * (a + 1)].reshape((-1, 1)))

a = 2  # SUBSAMPLE C, Model 5
sub.iloc[100000 * a:100000 * (a + 1), 1] = clf5.predict(
    test_copy['signal'].values[100000 * a:100000 * (a + 1)].reshape((-1, 1)))

a = 3  # SUBSAMPLE D, Model 1s
sub.iloc[100000 * a:100000 * (a + 1), 1] = clf1s.predict(
    test_copy['signal'].values[100000 * a:100000 * (a + 1)].reshape((-1, 1)))

a = 4  # SUBSAMPLE E, Model 1f
sub.iloc[100000 * a:100000 * (a + 1), 1] = clf1f.predict(
    test_copy['signal'].values[100000 * a:100000 * (a + 1)].reshape((-1, 1)))

a = 5  # SUBSAMPLE F, Model 10
sub.iloc[100000 * a:100000 * (a + 1), 1] = clf10.predict(
    test_copy['signal'].values[100000 * a:100000 * (a + 1)].reshape((-1, 1)))

a = 6  # SUBSAMPLE G, Model 5
sub.iloc[100000 * a:100000 * (a + 1), 1] = clf5.predict(
    test_copy['signal'].values[100000 * a:100000 * (a + 1)].reshape((-1, 1)))

a = 7  # SUBSAMPLE H, Model 10
sub.iloc[100000 * a:100000 * (a + 1), 1] = clf10.predict(
    test_copy['signal'].values[100000 * a:100000 * (a + 1)].reshape((-1, 1)))

a = 8  # SUBSAMPLE I, Model 1s
sub.iloc[100000 * a:100000 * (a + 1), 1] = clf1s.predict(
    test_copy['signal'].values[100000 * a:100000 * (a + 1)].reshape((-1, 1)))

a = 9  # SUBSAMPLE J, Model 3
sub.iloc[100000 * a:100000 * (a + 1), 1] = clf3.predict(
    test_copy['signal'].values[100000 * a:100000 * (a + 1)].reshape((-1, 1)))

# BATCHES 3 AND 4, Model 1s
sub.iloc[1000000:2000000, 1] = clf1s.predict(test_copy['signal'].values[1000000:2000000].reshape((-1, 1)))

# ===============Display Test Predictions===============
# let = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
# plt.figure(figsize=(20, 5))
# res = 1000
# plt.plot(range(0, test_df.shape[0], res), sub.open_channels[0::res])
# for i in range(5):
#     plt.plot([i * 500000, i * 500000], [-5, 12.5], 'r')
# for i in range(21):
#     plt.plot([i * 100000, i * 100000], [-5, 12.5], 'r:')
# for k in range(4):
#     plt.text(k * 500000 + 250000, 10, str(k + 1), size=20)
# for k in range(10):
#     plt.text(k * 100000 + 40000, 7.5, let[k], size=16)
# plt.title('Test Data Predictions', size=16)
# plt.show()

sub.to_csv('submission.csv', index=False, float_format='%.4f')
print('finish')
