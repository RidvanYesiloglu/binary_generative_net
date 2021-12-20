K=5;
N=64;
batch_size=10;
codesets = randi([0 1], batch_size, K, N);
pyrun("import os");
pyrun("os.chdir('../../neural_networks/functions')");
pyrun("import verify_correlations as vc");
auto_simple = double(pyrun("auto_simple = vc.autocorr_even_simple(codesets)", "auto_simple", codesets=(-1+2*codesets)));
auto_mtlb = calc_autocorr_K_x_N(codesets, true);
fprintf('Sum of sq. difference between MATLAB autocorrelation and simple python autocorrelation is %.5f.\n', sum((auto_simple-auto_mtlb).^2,'all'));
cross_simple = double(pyrun("cross_simple = vc.crosscorr_even_simple(codesets)", "cross_simple", codesets=(-1+2*codesets)));
cross_mtlb = calc_crosscorr_K_x_N(codesets, true);

fprintf('Sum of sq. difference between MATLAB crosscorrelation and simple python crosscorrelation is %.5f.\n', sum((cross_simple-cross_mtlb).^2,'all'));
