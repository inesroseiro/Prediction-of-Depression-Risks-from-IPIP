load('target.csv')
load('gb_pred.csv')
load('kn_pred.csv')
load('lr_pred.csv')
load('rf_pred.csv')
load('svr_pred.csv')


range = 1:numel(target(:,2));
line_width = 1;

target = target(range, 2);
gb_pred = gb_pred(range, 2);
kn_pred = kn_pred(range, 2);
lr_pred = lr_pred(range, 2);
rf_pred = rf_pred(range, 2);
svr_pred = svr_pred(range,2);

[target, indices] = sort(target);

gb_pred = gb_pred(indices);
kn_pred = kn_pred(indices);
lr_pred = lr_pred(indices);
rf_pred = rf_pred(indices);
svr_pred = svr_pred(indices);


figure
hold on
plot(target, 'LineWidth', line_width)
plot(gb_pred, 'LineWidth', line_width)
legend('target', 'Gradient boosting')
axis([-inf inf 0 8])
title('Prediction vs. target')
hold off

figure
hold on
plot(target, 'LineWidth', line_width)
plot(kn_pred, 'LineWidth', line_width)
legend('target', 'K-nearest neighbours')
axis([-inf inf 0 8])
title('Prediction vs. target')

hold off
figure
hold on
plot(target, 'LineWidth', line_width)
plot(lr_pred, 'LineWidth', line_width)
legend('target','Linear regression')
axis([-inf inf 0 8])
title('Prediction vs. target')

hold off
figure
hold on
plot(target, 'LineWidth', line_width)
plot(rf_pred, 'LineWidth', line_width)
legend('target','Random Forest')
axis([-inf inf 0 8])
title('Prediction vs. target')

hold off
figure
hold on
plot(target, 'LineWidth', line_width)
plot(svr_pred, 'LineWidth', line_width)
legend('target','Support-vector machine')
axis([-inf inf 0 inf])
title('Prediction vs. target')

hold off
figure
hold on
plot(target-lr_pred, 'LineWidth', line_width)
plot(target-gb_pred, 'LineWidth', line_width)
plot(target-rf_pred, 'LineWidth', line_width)
plot(target-kn_pred, 'LineWidth', line_width)
plot(target-svr_pred, 'LineWidth', line_width)
legend('lr','gb','rf','kn','svr')
title('Residues of prediction and target values')

figure
X = categorical({'Baseline', 'Gradient boosting', 'Support-vector machine', 'K-nearest neighbours','Random forest', 'Linear regression'});
X = reordercats(X,{'Baseline', 'Gradient boosting', 'Support-vector machine', 'K-nearest neighbours','Random forest', 'Linear regression'});
Y = [1.2715 0.9433 0.9810 0.9861 1.0076 1.0543];

bar(X,Y)
grid on
title('Initial Mean Absolute Error')

figure
X = categorical({'Random forest', 'Support-vector machine', 'Gradient boosting', 'K-nearest neighbours', 'Linear regression'});
X = reordercats(X,{'Random forest','Support-vector machine', 'Gradient boosting', 'K-nearest neighbours', 'Linear regression'});
Y = [mean(abs(target - rf_pred)) mean(abs(target - svr_pred)) mean(abs(target - gb_pred)) mean(abs(target - kn_pred)) mean(abs(target - lr_pred))];

bar(X,Y)
grid on

title('Tuned Mean Absolute Error')